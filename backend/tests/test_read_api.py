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
                PriceHistory(
                    stock_id=sid,
                    date=date(2026, 6, 8),
                    open=60,
                    high=61,
                    low=59,
                    close=60.0,
                    volume=1000,
                ),
                PriceHistory(
                    stock_id=sid,
                    date=date(2026, 6, 9),
                    open=60,
                    high=62,
                    low=60,
                    close=61.2,
                    volume=1200,
                ),
                DailySentiment(
                    stock_id=sid, date=date(2026, 6, 9), sentiment_agg=0.3, news_count=2
                ),
                Prediction(
                    stock_id=sid,
                    prediction_date=date(2026, 6, 9),
                    target_date=date(2026, 6, 10),
                    label="tang",
                    confidence=0.7,
                    model_version="stub_v0",
                ),
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


async def test_list_predictions_multi_stock_no_mixup(session_factory):
    # Bảo vệ N+1 refactor (window function): mỗi mã phải gom ĐÚNG close/changePct/sentiment/label,
    # không lẫn sang mã khác khi query gom toàn bộ.
    await _seed(session_factory)  # VCB
    async with session_factory() as s:
        fpt = Stock(symbol="FPT", name="FPT", exchange="HSX")
        s.add(fpt)
        await s.flush()
        s.add_all(
            [
                PriceHistory(
                    stock_id=fpt.id,
                    date=date(2026, 6, 8),
                    open=100,
                    high=101,
                    low=99,
                    close=100.0,
                    volume=500,
                ),
                PriceHistory(
                    stock_id=fpt.id,
                    date=date(2026, 6, 9),
                    open=100,
                    high=105,
                    low=100,
                    close=110.0,
                    volume=600,
                ),  # +10%
                DailySentiment(
                    stock_id=fpt.id, date=date(2026, 6, 9), sentiment_agg=-0.5, news_count=1
                ),
                Prediction(
                    stock_id=fpt.id,
                    prediction_date=date(2026, 6, 9),
                    target_date=date(2026, 6, 10),
                    label="giam",
                    confidence=0.6,
                    model_version="stub_v0",
                ),
            ]
        )
        await s.commit()
        lst = await read_api.list_predictions(s)
    by_sym = {p["symbol"]: p for p in lst}
    assert by_sym["VCB"]["close"] == 61.2 and by_sym["VCB"]["changePct"] == 2.0
    assert by_sym["VCB"]["label"] == "tang" and by_sym["VCB"]["sentiment"] == 0.3
    assert by_sym["FPT"]["close"] == 110.0 and by_sym["FPT"]["changePct"] == 10.0
    assert by_sym["FPT"]["label"] == "giam" and by_sym["FPT"]["sentiment"] == -0.5


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
        # CONTRACT: actual.date = prediction_date (ngày T = 9/6), KHÔNG phải target_date.
        # prediction ngày T=9/6 label=tang; actual ngày 9/6 = tang → đúng.
        s.add(ActualResult(stock_id=sid, date=date(2026, 6, 9), label="tang"))
        await s.commit()
        acc = await read_api.get_accuracy(s)
    assert acc["overall"] == 1.0
    assert acc["byLabel"]["tang"] == 1.0
    assert len(acc["series"]) == 1


async def test_accuracy_join_by_prediction_date_not_target(session_factory):
    # Bảo vệ contract: actual khớp prediction_date (T), không phải target_date (T+1).
    sid = await _seed(session_factory)  # prediction_date=9/6, target_date=10/6, label=tang
    async with session_factory() as s:
        # Fill actual theo target_date (10/6) — KHÔNG được khớp → vẫn rỗng.
        s.add(ActualResult(stock_id=sid, date=date(2026, 6, 10), label="tang"))
        await s.commit()
        acc = await read_api.get_accuracy(s)
    assert acc["series"] == []  # không khớp vì 10/6 ≠ prediction_date 9/6


# --- /api/news (M6) ---


async def _seed_news(factory):
    """2 mã (VCB, FPT) + 2 bài: bài 1 nhắc CẢ HAI mã (N-N), bài 2 chỉ VCB, sentiment NULL."""
    from datetime import datetime

    from models.database import News, NewsStock

    async with factory() as s:
        vcb = Stock(symbol="VCB", name="Vietcombank", exchange="HSX", sector="Ngân hàng")
        fpt = Stock(symbol="FPT", name="FPT Corp", exchange="HSX", sector="Công nghệ")
        s.add_all([vcb, fpt])
        await s.flush()
        n1 = News(
            title="VCB và FPT cùng tăng",
            url="https://cafef.vn/n1",
            source="cafef",
            published_at=datetime(2026, 6, 10, 9, 0),
            sentiment_score=0.5,
        )
        n2 = News(
            title="VCB họp cổ đông",
            url="https://cafef.vn/n2",
            source="cafef",
            published_at=datetime(2026, 6, 11, 9, 0),
            sentiment_score=None,  # stub chưa chấm — API phải trả 0.0
        )
        s.add_all([n1, n2])
        await s.flush()
        s.add_all(
            [
                NewsStock(news_id=n1.id, stock_id=vcb.id),
                NewsStock(news_id=n1.id, stock_id=fpt.id),
                NewsStock(news_id=n2.id, stock_id=vcb.id),
            ]
        )
        await s.commit()


async def test_get_news_n_n_and_order(session_factory):
    """1 bài nhiều mã qua news_stocks; sắp published_at desc; shape khớp NewsItem."""
    await _seed_news(session_factory)
    async with session_factory() as s:
        vcb_news = await read_api.get_news(s, "vcb")  # lowercase → vẫn khớp
        fpt_news = await read_api.get_news(s, "FPT")
    assert [n["title"] for n in vcb_news] == ["VCB họp cổ đông", "VCB và FPT cùng tăng"]
    assert [n["title"] for n in fpt_news] == ["VCB và FPT cùng tăng"]
    item = vcb_news[1]
    assert set(item) == {"title", "source", "publishedAt", "sentiment", "url"}
    assert item["sentiment"] == 0.5
    assert item["publishedAt"] == "2026-06-10T09:00:00"


async def test_get_news_null_sentiment_is_zero(session_factory):
    await _seed_news(session_factory)
    async with session_factory() as s:
        vcb_news = await read_api.get_news(s, "VCB")
    assert vcb_news[0]["sentiment"] == 0.0  # NULL → 0.0, quy ước "không tin → 0"


async def test_get_news_limit(session_factory):
    await _seed_news(session_factory)
    async with session_factory() as s:
        vcb_news = await read_api.get_news(s, "VCB", limit=1)
    assert len(vcb_news) == 1
    assert vcb_news[0]["title"] == "VCB họp cổ đông"  # limit cắt SAU khi sắp desc


async def test_get_news_unknown_symbol_none_vs_empty(session_factory):
    """Mã không tồn tại → None (route 404); mã có nhưng chưa tin → []."""
    await _seed_news(session_factory)
    async with session_factory() as s:
        assert await read_api.get_news(s, "XXX") is None
        stock = Stock(symbol="HPG", name="Hòa Phát", exchange="HSX", sector="Thép")
        s.add(stock)
        await s.commit()
        assert await read_api.get_news(s, "HPG") == []
