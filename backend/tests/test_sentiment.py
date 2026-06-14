"""Test sentiment: score_texts (PhoBERT/degrade) + score_news + aggregation + idempotent.

Gồm test thuần effective_trading_day / aggregate_daily (cutoff 16:00 + dồn ngày nghỉ +
sentiment_extreme) — không cần DB.
"""

from datetime import date, datetime

import pytest
from sqlalchemy import select

from models.database import DailySentiment, News, NewsStock, PriceHistory, Stock
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


# --- effective_trading_day / aggregate_daily (thuần, không DB) ------------------------------

# Tuần 8-12/6/2026 (T2-T6) + 15/6 (T2 kế); 13-14/6 là T7/CN (nghỉ).
_TDAYS = [
    date(2026, 6, 8),
    date(2026, 6, 9),
    date(2026, 6, 10),
    date(2026, 6, 11),
    date(2026, 6, 12),
    date(2026, 6, 15),
]


def test_effective_day_before_close_same_session():
    # Tin 14:00 phiên giao dịch → dùng ngay phiên đó (biết trước 16:00).
    assert sentiment.effective_trading_day(datetime(2026, 6, 10, 14, 0), _TDAYS) == date(
        2026, 6, 10
    )


def test_effective_day_at_or_after_close_rolls_next():
    # >= 16:00 → sau giờ chốt → phiên giao dịch kế tiếp (chống leakage serve-time).
    assert sentiment.effective_trading_day(datetime(2026, 6, 10, 16, 0), _TDAYS) == date(
        2026, 6, 11
    )
    assert sentiment.effective_trading_day(datetime(2026, 6, 10, 17, 30), _TDAYS) == date(
        2026, 6, 11
    )


def test_effective_day_friday_evening_and_weekend_roll_to_monday():
    # Tin tối T6 + tin T7/CN đều dồn vào phiên T2 kế (không bị left-join drop ở builder).
    assert sentiment.effective_trading_day(datetime(2026, 6, 12, 19, 0), _TDAYS) == date(
        2026, 6, 15
    )
    assert sentiment.effective_trading_day(datetime(2026, 6, 13, 10, 0), _TDAYS) == date(
        2026, 6, 15
    )
    assert sentiment.effective_trading_day(datetime(2026, 6, 14, 8, 0), _TDAYS) == date(2026, 6, 15)


def test_effective_day_after_latest_session_waits():
    # Tin sau phiên mới nhất → chưa có phiên để gắn → None ("chờ").
    assert sentiment.effective_trading_day(datetime(2026, 6, 15, 17, 0), _TDAYS) is None


def test_aggregate_daily_extreme_is_signed_max_abs():
    rows = [
        (datetime(2026, 6, 10, 9, 0), 0.4),
        (datetime(2026, 6, 10, 10, 0), -0.9),  # |max| nhưng âm → extreme giữ dấu
        (datetime(2026, 6, 10, 11, 0), 0.2),
    ]
    out = sentiment.aggregate_daily(rows, _TDAYS)
    assert len(out) == 1
    rec = out[0]
    assert rec["date"] == date(2026, 6, 10)
    assert rec["news_count"] == 3
    assert abs(rec["sentiment_agg"] - (0.4 - 0.9 + 0.2) / 3) < 1e-9
    assert rec["sentiment_extreme"] == -0.9


def test_aggregate_daily_consolidates_weekend_into_monday():
    rows = [
        (datetime(2026, 6, 12, 19, 0), 0.5),  # tối T6
        (datetime(2026, 6, 13, 9, 0), 0.9),  # T7
        (datetime(2026, 6, 15, 9, 0), -0.1),  # phiên T2 (trước 16:00)
    ]
    out = {r["date"]: r for r in sentiment.aggregate_daily(rows, _TDAYS)}
    assert set(out) == {date(2026, 6, 15)}  # cả 3 dồn về T2
    assert out[date(2026, 6, 15)]["news_count"] == 3
    assert out[date(2026, 6, 15)]["sentiment_extreme"] == 0.9


async def _seed_news(factory, stock_symbol, items, trading_days=None):
    """items: list (published_at, sentiment_score). Tạo stock + news + news_stocks.

    `trading_days`: list date có giá (price_history) để build_daily_sentiment ánh xạ tin về
    phiên giao dịch hiệu lực. None → suy ra từ ngày lịch của các tin (mọi tin trước 16:00).
    """
    if trading_days is None:
        trading_days = sorted({pub.date() for pub, _ in items})
    async with factory() as s:
        stock = Stock(symbol=stock_symbol, name=stock_symbol, exchange="HOSE")
        s.add(stock)
        await s.flush()
        for d in trading_days:
            s.add(PriceHistory(stock_id=stock.id, date=d, open=1, high=1, low=1, close=1, volume=1))
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


async def test_build_daily_sentiment_weekend_rolls_into_next_session(session_factory, monkeypatch):
    monkeypatch.setattr(sentiment, "SessionLocal", session_factory)
    # Phiên giao dịch: T6 12/6 + T2 15/6 (nghỉ T7/CN). Tin tối T6 (>=16:00) + tin T7 phải dồn
    # vào phiên T2 15/6, KHÔNG mất (bug left-join cũ) — và sentiment_extreme = signed max-abs.
    await _seed_news(
        session_factory,
        "VCB",
        [
            (datetime(2026, 6, 12, 11, 0), 0.3),  # T6 trước 16:00 → phiên 12/6
            (datetime(2026, 6, 12, 18, 0), -0.9),  # T6 sau 16:00 → dồn 15/6
            (datetime(2026, 6, 13, 9, 0), 0.5),  # T7 → dồn 15/6
        ],
        trading_days=[date(2026, 6, 12), date(2026, 6, 15)],
    )
    n = await sentiment.build_daily_sentiment("VCB")
    assert n == 2  # 12/6 và 15/6 (không có 13/6)
    async with session_factory() as s:
        rows = (
            (await s.execute(select(DailySentiment).order_by(DailySentiment.date))).scalars().all()
        )
    by_date = {r.date.isoformat(): r for r in rows}
    assert set(by_date) == {"2026-06-12", "2026-06-15"}
    assert by_date["2026-06-12"].news_count == 1
    mon = by_date["2026-06-15"]
    assert mon.news_count == 2  # tin tối T6 + tin T7
    assert abs(mon.sentiment_agg - (-0.9 + 0.5) / 2) < 1e-9
    assert mon.sentiment_extreme == -0.9


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
