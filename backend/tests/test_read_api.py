"""Test read_api: shape + logic của các endpoint Phase 1.4 (predictions/history/accuracy).

Dùng SQLite in-memory (conftest). Kiểm: exchange map HSX→HOSE, changePct, sentiment fallback 0,
accuracy stub-aware (rỗng khi chưa có actual_results), accuracy tính đúng khi có actual.
"""

from datetime import date

from models.database import (
    ActualResult,
    DailySentiment,
    Prediction,
    PriceHistory,
    Stock,
)
from services import read_api


async def _seed(factory):
    """Tạo 1 mã VCB: 2 phiên giá, 1 daily_sentiment, 1 prediction. Trả stock_id."""
    async with factory() as s:
        stock = Stock(symbol="VCB", name="Vietcombank", exchange="HSX", sector="Ngân hàng")
        s.add(stock)
        await s.flush()
        sid = stock.id
        s.add_all(
            [
                PriceHistory(stock_id=sid, date=date(2026, 6, 8), open=60, high=61,
                             low=59, close=60.0, volume=1000),
                PriceHistory(stock_id=sid, date=date(2026, 6, 9), open=60, high=62,
                             low=60, close=61.2, volume=1200),
                DailySentiment(stock_id=sid, date=date(2026, 6, 9),
                               sentiment_agg=0.3, news_count=2),
                Prediction(stock_id=sid, prediction_date=date(2026, 6, 9),
                           target_date=date(2026, 6, 10), label="tang", confidence=0.7,
                           model_version="stub_v0"),
            ]
        )
        await s.commit()
        return sid


async def test_get_prediction_shape_and_exchange_map(session_factory):
    await _seed(session_factory)
    async with session_factory() as s:
        p = await read_api.get_prediction(s, "vcb")  # lowercase → upper bên trong
    assert p["symbol"] == "VCB"
    assert p["exchange"] == "HOSE"  # HSX → HOSE
    assert p["label"] == "tang"
    assert p["confidence"] == 0.7
    assert p["modelVersion"] == "stub_v0"
    assert p["sentiment"] == 0.3  # daily_sentiment gần nhất
    assert p["close"] == 61.2
    # changePct = (61.2 - 60.0)/60.0 * 100 = 2.0
    assert p["changePct"] == 2.0


async def test_get_prediction_missing_symbol_returns_none(session_factory):
    await _seed(session_factory)
    async with session_factory() as s:
        assert await read_api.get_prediction(s, "XXX") is None


async def test_list_predictions_only_with_label(session_factory):
    await _seed(session_factory)
    async with session_factory() as s:
        # thêm 1 mã KHÔNG có prediction → không xuất hiện trong list
        s2 = Stock(symbol="FPT", name="FPT", exchange="HSX")
        s.add(s2)
        await s.commit()
        lst = await read_api.list_predictions(s)
    assert [p["symbol"] for p in lst] == ["VCB"]  # FPT bị loại (chưa có dự đoán)


async def test_get_history_with_sentiment_fallback(session_factory):
    await _seed(session_factory)
    async with session_factory() as s:
        h = await read_api.get_history(s, "VCB")
    assert len(h) == 2
    assert h[0] == {"date": "2026-06-08", "close": 60.0, "sentiment": 0.0}  # ngày không tin → 0
    assert h[1] == {"date": "2026-06-09", "close": 61.2, "sentiment": 0.3}


async def test_accuracy_empty_when_no_actuals(session_factory):
    await _seed(session_factory)
    async with session_factory() as s:
        acc = await read_api.get_accuracy(s)
    assert acc["overall"] == 0.0
    assert acc["series"] == []
    assert "detail" in acc  # giải thích chưa có actual_results


async def test_accuracy_computed_with_actuals(session_factory):
    sid = await _seed(session_factory)
    async with session_factory() as s:
        # prediction target_date=10/6 label=tang. actual ngày 10/6 = tang → đúng.
        s.add(ActualResult(stock_id=sid, date=date(2026, 6, 10), label="tang"))
        await s.commit()
        acc = await read_api.get_accuracy(s)
    assert acc["overall"] == 1.0
    assert acc["byLabel"]["tang"] == 1.0
    assert len(acc["series"]) == 1
