"""Test sentiment: score_news (stub) + build_daily_sentiment aggregation + idempotent."""

from datetime import datetime

from sqlalchemy import select

from models.database import DailySentiment, News, NewsStock, Stock
from services import sentiment


def test_score_text_stub_returns_neutral():
    # Stub Phase 1.2: trung tính 0.0 cho tới khi có PhoBERT.
    assert sentiment.score_text("Cổ phiếu tăng trần") == 0.0
    assert sentiment.score_text("") == 0.0


async def _seed_news(factory, stock_symbol, items):
    """items: list (published_at, sentiment_score). Tạo stock + news + news_stocks."""
    async with factory() as s:
        stock = Stock(symbol=stock_symbol, name=stock_symbol, exchange="HOSE")
        s.add(stock)
        await s.flush()
        for i, (pub, score) in enumerate(items):
            n = News(
                title=f"tin {i}",
                url=f"https://x/{stock_symbol}/{i}",
                source="cafef",
                published_at=pub,
                sentiment_score=score,
            )
            s.add(n)
            await s.flush()
            s.add(NewsStock(news_id=n.id, stock_id=stock.id))
        await s.commit()
        return stock.id


async def test_build_daily_sentiment_aggregates_by_day(session_factory, monkeypatch):
    monkeypatch.setattr(sentiment, "SessionLocal", session_factory)
    # 2 tin ngày 10/6 (0.4, 0.8 → tb 0.6), 1 tin ngày 11/6 (-0.2)
    await _seed_news(
        session_factory,
        "VCB",
        [
            (datetime(2026, 6, 10, 9, 0), 0.4),
            (datetime(2026, 6, 10, 15, 0), 0.8),
            (datetime(2026, 6, 11, 9, 0), -0.2),
        ],
    )
    n = await sentiment.build_daily_sentiment("VCB")
    assert n == 2  # 2 ngày có tin

    async with session_factory() as s:
        rows = (
            (await s.execute(select(DailySentiment).order_by(DailySentiment.date))).scalars().all()
        )
    assert [r.date.isoformat() for r in rows] == ["2026-06-10", "2026-06-11"]
    assert abs(rows[0].sentiment_agg - 0.6) < 1e-9
    assert rows[0].news_count == 2
    assert rows[1].sentiment_agg == -0.2
    assert rows[1].news_count == 1


async def test_build_daily_sentiment_idempotent(session_factory, monkeypatch):
    monkeypatch.setattr(sentiment, "SessionLocal", session_factory)
    await _seed_news(session_factory, "FPT", [(datetime(2026, 6, 10, 9, 0), 0.5)])
    await sentiment.build_daily_sentiment("FPT")
    await sentiment.build_daily_sentiment("FPT")  # chạy lại
    async with session_factory() as s:
        cnt = len((await s.execute(select(DailySentiment))).scalars().all())
    assert cnt == 1  # không nhân đôi


async def test_build_daily_sentiment_unseeded_symbol(session_factory, monkeypatch):
    monkeypatch.setattr(sentiment, "SessionLocal", session_factory)
    assert await sentiment.build_daily_sentiment("ZZZ") == 0  # mã chưa seed → 0, không lỗi


async def test_score_news_fills_null_scores(session_factory, monkeypatch):
    monkeypatch.setattr(sentiment, "SessionLocal", session_factory)
    # tin chưa có điểm (None) → score_news chấm bằng stub (0.0)
    await _seed_news(session_factory, "HPG", [(datetime(2026, 6, 10, 9, 0), None)])
    n = await sentiment.score_news("HPG")
    assert n == 1
    async with session_factory() as s:
        scores = (await s.execute(select(News.sentiment_score))).scalars().all()
    assert scores == [0.0]
    # chạy lại: không còn tin None để chấm
    assert await sentiment.score_news("HPG") == 0
