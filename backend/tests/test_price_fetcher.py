"""Test price_fetcher: transform, validate, và upsert idempotent (mock vnstock)."""

from datetime import date

import pandas as pd
from sqlalchemy import func, select

from models.database import PriceHistory, Stock
from services import price_fetcher, stock_seed


def _df(close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-03"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.0, 99.0],
            "close": [close, close + 1],
            "volume": [1_000, 1_100],
        }
    )


def test_transform_maps_columns():
    rows = price_fetcher.transform(_df(), stock_id=7)
    assert len(rows) == 2
    assert rows[0]["stock_id"] == 7
    assert rows[0]["date"] == date(2024, 1, 2)
    assert rows[0]["close"] == 100.0
    assert rows[0]["volume"] == 1_000


def test_transform_empty():
    assert price_fetcher.transform(pd.DataFrame(), stock_id=1) == []


def _row(**over):
    base = {
        "stock_id": 1,
        "date": date(2024, 1, 2),
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 10,
    }
    base.update(over)
    return base


def test_validate_drops_dirty_rows():
    rows = [
        _row(),  # ok
        _row(date=None),  # date None
        _row(open=None),  # null OHLC
        _row(high=0.5, low=2.0),  # high < low
        _row(open=-1.0),  # giá âm
    ]
    clean, dropped = price_fetcher.validate(rows)
    assert len(clean) == 1
    assert dropped == 4


async def test_fetch_idempotent_and_updates(session_factory, monkeypatch):
    # Trỏ SessionLocal của 2 service vào DB test
    monkeypatch.setattr(price_fetcher, "SessionLocal", session_factory)
    monkeypatch.setattr(stock_seed, "SessionLocal", session_factory)
    # Mock metadata (tránh gọi mạng vnstock)
    monkeypatch.setattr(
        stock_seed,
        "_fetch_metadata",
        lambda symbol, source="vci": {
            "symbol": symbol.upper(),
            "name": "Test Corp",
            "exchange": "HOSE",
            "sector": None,
        },
    )

    # Lần 1: close=100
    monkeypatch.setattr(price_fetcher, "_fetch_raw", lambda *a, **k: _df(100.0))
    n1 = await price_fetcher.fetch_price_history("vcb")
    assert n1 == 2

    # Lần 2: close=200 (giá adjusted thay đổi hồi tố)
    monkeypatch.setattr(price_fetcher, "_fetch_raw", lambda *a, **k: _df(200.0))
    await price_fetcher.fetch_price_history("vcb")

    async with session_factory() as s:
        n_price = (await s.execute(select(func.count()).select_from(PriceHistory))).scalar_one()
        n_stock = (await s.execute(select(func.count()).select_from(Stock))).scalar_one()
        first_close = (
            await s.execute(select(PriceHistory.close).order_by(PriceHistory.date))
        ).scalars().first()

    assert n_price == 2  # idempotent — không nhân đôi
    assert n_stock == 1  # seed_stock cũng idempotent theo symbol
    assert first_close == 200.0  # giá được UPDATE
