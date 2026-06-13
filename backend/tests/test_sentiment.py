"""Test sentiment: score_texts (PhoBERT/degrade) + score_news + aggregation + idempotent."""

from datetime import datetime

import pytest
from sqlalchemy import select

from models.database import DailySentiment, News, NewsStock, Stock
from services import sentiment


@pytest.mark.skipif(
    not (sentiment.MODEL_DIR / "config.json").exists(),
    reason="chưa có artifact PhoBERT (ml/artifacts/phobert_sentiment) — chạy sau khi fine-tune",
)
def test_phobert_golden_sentences_directionality():
    # 5 câu vàng: model thật phải đoán ĐÚNG CHIỀU (pos > 0, neg < 0). Skip nếu chưa có artifact.
    sentiment._default_scorer_loaded = False  # ép load lại artifact thật
    sentiment._default_scorer_cache = None
    pos = sentiment.score_text("Vietcombank báo lãi kỷ lục, cổ phiếu lập đỉnh")
    pos2 = sentiment.score_text("HPG tăng trần, khối ngoại mua ròng mạnh")
    neg = sentiment.score_text("Cổ phiếu BIDV giảm sàn, nhà đầu tư bán tháo")
    neg2 = sentiment.score_text("Doanh nghiệp thua lỗ nặng, nợ xấu phình to")
    neu = sentiment.score_text("FPT chốt danh sách cổ đông trả cổ tức")
    assert pos > 0 and pos2 > 0, f"câu tích cực phải > 0 (được {pos}, {pos2})"
    assert neg < 0 and neg2 < 0, f"câu tiêu cực phải < 0 (được {neg}, {neg2})"
    assert -1.0 <= neu <= 1.0  # neu chỉ cần trong khoảng hợp lệ


def test_score_text_degrades_to_neutral_without_artifact(monkeypatch):
    # Thiếu artifact PhoBERT → degrade 0.0, pipeline vẫn chạy (M8). scorer=None ép load mặc định.
    monkeypatch.setattr(sentiment, "_default_scorer_loaded", False)
    monkeypatch.setattr(sentiment, "_default_scorer_cache", None)
    monkeypatch.setattr(sentiment, "_phobert_scorer", lambda: None)
    assert sentiment.score_text("Cổ phiếu tăng trần") == 0.0
    assert sentiment.score_text("") == 0.0


def test_score_texts_uses_injected_scorer():
    # scorer inject → score_texts trả đúng output, không cần torch.
    def fake(texts):
        return [0.7 if "lãi" in t else -0.3 for t in texts]

    out = sentiment.score_texts(["VCB lãi lớn", "HPG giảm sàn"], scorer=fake)
    assert out == [0.7, -0.3]
    assert sentiment.score_text("VCB lãi lớn", scorer=fake) == 0.7


def test_score_texts_empty_returns_empty():
    assert sentiment.score_texts([], scorer=lambda t: [1.0]) == []


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
    # tin chưa có điểm (None) → score_news chấm bằng scorer inject (+0.5)
    await _seed_news(session_factory, "HPG", [(datetime(2026, 6, 10, 9, 0), None)])
    n = await sentiment.score_news("HPG", scorer=lambda titles: [0.5] * len(titles))
    assert n == 1
    async with session_factory() as s:
        scores = (await s.execute(select(News.sentiment_score))).scalars().all()
    assert scores == [0.5]
    # chạy lại (không rescore): không còn tin NULL để chấm
    assert await sentiment.score_news("HPG", scorer=lambda titles: [0.5] * len(titles)) == 0


async def test_score_news_rescore_overwrites_existing(session_factory, monkeypatch):
    monkeypatch.setattr(sentiment, "SessionLocal", session_factory)
    # tin đã mang điểm 0.0 của stub → rescore=True chấm lại bằng PhoBERT (mô phỏng -0.8)
    await _seed_news(session_factory, "VIC", [(datetime(2026, 6, 10, 9, 0), 0.0)])
    n = await sentiment.score_news("VIC", rescore=True, scorer=lambda titles: [-0.8] * len(titles))
    assert n == 1
    async with session_factory() as s:
        scores = (await s.execute(select(News.sentiment_score))).scalars().all()
    assert scores == [-0.8]
