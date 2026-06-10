"""Test fill_actual_results: chấm nhãn thực tế từ close T vs close phiên kế tiếp.

SQLite in-memory (conftest). Kiểm: chấm đúng, bỏ qua khi chưa có phiên kế tiếp, phiên kế tiếp
= ngày>T nhỏ nhất (không +1 lịch), idempotent, tích hợp get_accuracy.
"""

from datetime import date

from sqlalchemy import func, select

from models.database import ActualResult, Prediction, PriceHistory, Stock
from services import read_api
from services.actual_results import fill_actual_results


async def _add_stock(s, symbol="VCB"):
    stock = Stock(symbol=symbol, name=symbol, exchange="HSX")
    s.add(stock)
    await s.flush()
    return stock.id


def _price(sid, d, close):
    return PriceHistory(
        stock_id=sid, date=d, open=close, high=close, low=close, close=close, volume=100
    )


def _pred(sid, t, label="tang"):
    return Prediction(
        stock_id=sid,
        prediction_date=t,
        target_date=t,
        label=label,
        confidence=0.5,
        model_version="stub_v0",
    )


async def test_fill_scores_correct_label(session_factory):
    async with session_factory() as s:
        sid = await _add_stock(s)
        # close[T=8/6]=100, close[phiên kế tiếp 9/6]=102 → +2% → tang
        s.add_all(
            [
                _price(sid, date(2026, 6, 8), 100.0),
                _price(sid, date(2026, 6, 9), 102.0),
                _pred(sid, date(2026, 6, 8)),
            ]
        )
        await s.commit()
        n = await fill_actual_results(s)
        rows = (await s.execute(select(ActualResult.date, ActualResult.label))).all()
    assert n == 1
    assert rows == [(date(2026, 6, 8), "tang")]


async def test_fill_skips_when_no_next_session(session_factory):
    async with session_factory() as s:
        sid = await _add_stock(s)
        # chỉ có giá tới ngày T → chưa có phiên kế tiếp → bỏ qua, không lỗi
        s.add_all([_price(sid, date(2026, 6, 8), 100.0), _pred(sid, date(2026, 6, 8))])
        await s.commit()
        n = await fill_actual_results(s)
        cnt = (await s.execute(select(func.count()).select_from(ActualResult))).scalar_one()
    assert n == 0
    assert cnt == 0


async def test_next_session_is_not_calendar_plus_one(session_factory):
    async with session_factory() as s:
        sid = await _add_stock(s)
        # T = thứ Sáu 5/6; phiên kế tiếp là thứ Hai 8/6 (KHÔNG phải thứ Bảy 6/6).
        # close 100 → 95 = -5% → giam. Chứng minh dùng "ngày>T nhỏ nhất" trong price_history.
        s.add_all(
            [
                _price(sid, date(2026, 6, 5), 100.0),
                _price(sid, date(2026, 6, 8), 95.0),
                _pred(sid, date(2026, 6, 5)),
            ]
        )
        await s.commit()
        await fill_actual_results(s)
        rows = (await s.execute(select(ActualResult.date, ActualResult.label))).all()
    assert rows == [(date(2026, 6, 5), "giam")]


async def test_fill_idempotent(session_factory):
    async with session_factory() as s:
        sid = await _add_stock(s)
        s.add_all(
            [
                _price(sid, date(2026, 6, 8), 100.0),
                _price(sid, date(2026, 6, 9), 102.0),
                _pred(sid, date(2026, 6, 8)),
            ]
        )
        await s.commit()
        await fill_actual_results(s)
        await fill_actual_results(s)  # chạy lần 2
        cnt = (await s.execute(select(func.count()).select_from(ActualResult))).scalar_one()
    assert cnt == 1  # không nhân đôi


async def test_integration_with_get_accuracy(session_factory):
    async with session_factory() as s:
        sid = await _add_stock(s)
        # prediction label=tang; thực tế +2% (tang) → đúng → accuracy 1.0
        s.add_all(
            [
                _price(sid, date(2026, 6, 8), 100.0),
                _price(sid, date(2026, 6, 9), 102.0),
                _pred(sid, date(2026, 6, 8), label="tang"),
            ]
        )
        await s.commit()
        await fill_actual_results(s)
        acc = await read_api.get_accuracy(s)
    assert acc["overall"] == 1.0
    assert acc["byLabel"]["tang"] == 1.0
