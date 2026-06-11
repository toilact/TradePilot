"""Fetch OHLCV (giá điều chỉnh) qua vnstock → upsert price_history.

Mắt xích dữ liệu đầu tiên của walking skeleton.
- Nguồn: vnstock source `vci` (giá điều chỉnh mặc định).
- Lịch sử tối đa (start mặc định 2009-01-01).
- Upsert theo (stock_id, date) → idempotent, cập nhật giá adjusted hồi tố.
- Validate trọng điểm: bỏ dòng bẩn để không đầu độc nhãn ±1%.

Chạy thủ công: python -m services.price_fetcher --symbol VCB
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime

import pandas as pd
import structlog

from db.upsert import upsert
from models.database import PriceHistory, SessionLocal
from services.stock_seed import seed_stock

logger = structlog.get_logger(__name__)

DEFAULT_START = "2009-01-01"
SOURCE = "vci"

_OHLCV = ("open", "high", "low", "close", "volume")


def transform(df: pd.DataFrame, stock_id: int) -> list[dict]:
    """DataFrame vnstock (cột time, open, high, low, close, volume) → list dict cho model."""
    if df is None or df.empty:
        return []
    out: list[dict] = []
    for _, r in df.iterrows():
        raw = r.get("time")
        d = pd.to_datetime(raw).date() if raw is not None else None
        out.append(
            {
                "stock_id": stock_id,
                "date": d,
                "open": _to_float(r.get("open")),
                "high": _to_float(r.get("high")),
                "low": _to_float(r.get("low")),
                "close": _to_float(r.get("close")),
                "volume": _to_int(r.get("volume")),
            }
        )
    return out


def validate(rows: list[dict]) -> tuple[list[dict], int]:
    """Giữ dòng hợp lệ, đếm dòng bỏ. Bảo vệ bất biến nhãn khỏi dữ liệu bẩn."""
    clean: list[dict] = []
    for row in rows:
        if not isinstance(row.get("date"), date):
            continue
        ohlc = [row["open"], row["high"], row["low"], row["close"]]
        if any(v is None for v in ohlc) or row["volume"] is None:
            continue
        if any(v <= 0 for v in ohlc):
            continue
        if row["high"] < row["low"]:
            continue
        if row["volume"] < 0:
            continue
        clean.append(row)
    return clean, len(rows) - len(clean)


def _fetch_raw(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    """Gọi vnstock (blocking) — bọc qua asyncio.to_thread ở caller."""
    from vnstock import Quote

    return Quote(source=source, symbol=symbol).history(start=start, end=end, interval="1D")


async def fetch_price_history(
    symbol: str,
    start: str = DEFAULT_START,
    end: str | None = None,
    source: str = SOURCE,
) -> int:
    """Lấy & upsert OHLCV cho `symbol`. Trả về số dòng đã ghi."""
    symbol = symbol.upper()
    end = end or date.today().isoformat()

    stock_id = await seed_stock(symbol, source)
    df = await asyncio.to_thread(_fetch_raw, symbol, start, end, source)

    rows = transform(df, stock_id)
    clean, dropped = validate(rows)

    async with SessionLocal() as session:
        written = await upsert(
            session,
            PriceHistory,
            clean,
            index_elements=["stock_id", "date"],
            update_cols=list(_OHLCV),
        )

    logger.info(
        "price_fetched", symbol=symbol, start=start, end=end, written=written, dropped=dropped
    )
    return written


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # loại NaN
    except (TypeError, ValueError):
        return None


def _to_int(v) -> int | None:
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _main() -> None:
    from logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Fetch OHLCV 1 mã CK qua vnstock")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    args = parser.parse_args()
    n = asyncio.run(fetch_price_history(args.symbol, args.start, args.end))
    print(f"✓ {args.symbol}: ghi {n} dòng price_history")


if __name__ == "__main__":
    _main()
