"""PhoBERT sentiment inference + tổng hợp daily_sentiment.

Backend CHỈ inference — model train ở ml/ rồi export sang ml/artifacts/phobert_sentiment/.

Mắt xích Phase 1.2 (thứ tự): score_news (chấm điểm tin) → build_daily_sentiment (gộp ngày/mã).
- score_text hiện là STUB trả 0.0 (PhoBERT làm ở Phase 1.2-B trên Kaggle). Pipeline chạy được
  ngay; news_count vẫn là tín hiệu thật cho TFT, sentiment_agg sẽ có tín hiệu khi thay model thật.

Bất biến:
  - Ngày không tin → sentiment_agg = 0 (feature builder TFT điền; aggregate chỉ ghi ngày CÓ tin).
  - Chống leakage: gộp theo NGÀY của published_at (đã chuẩn hoá giờ VN ở crawler).
  - sentiment_agg ∈ [-1, 1]; chấm trên TITLE (content=None theo governance).
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import select

from db.upsert import upsert
from models.database import DailySentiment, News, NewsStock, SessionLocal, Stock

logger = logging.getLogger(__name__)

MODEL_PATH = "../ml/artifacts/phobert_sentiment"


def score_text(text: str) -> float:
    """Sentiment score [-1, 1] cho 1 đoạn tin.

    STUB Phase 1.2: trả 0.0 (trung tính) cho tới khi PhoBERT fine-tuned sẵn sàng
    (Phase 1.2-B, train Kaggle → ml/artifacts/phobert_sentiment/). Khi thay model thật,
    chỉ cần đổi thân hàm này — score_news/build_daily_sentiment giữ nguyên.
    """
    return 0.0


async def score_news(symbol: str | None = None) -> int:
    """Chấm sentiment_score cho tin CHƯA có điểm (theo title). Trả số tin đã chấm.

    `symbol=None` → chấm mọi tin chưa điểm; có symbol → chỉ tin map tới mã đó.
    """
    async with SessionLocal() as session:
        stmt = select(News.id, News.title).where(News.sentiment_score.is_(None))
        if symbol:
            stmt = (
                stmt.join(NewsStock, NewsStock.news_id == News.id)
                .join(Stock, Stock.id == NewsStock.stock_id)
                .where(Stock.symbol == symbol.upper())
            )
        rows = (await session.execute(stmt)).all()
        for news_id, title in rows:
            score = score_text(title or "")
            await session.execute(
                News.__table__.update().where(News.id == news_id).values(sentiment_score=score)
            )
        await session.commit()
    logger.info("score_news %s: chấm %d tin", symbol or "(tất cả)", len(rows))
    return len(rows)


async def build_daily_sentiment(symbol: str) -> int:
    """Gộp news.sentiment_score → daily_sentiment theo (stock_id, ngày). Trả số ngày đã ghi.

    sentiment_agg = trung bình điểm các tin trong ngày; news_count = số tin. Chỉ ghi ngày CÓ tin
    (ngày không tin → 0, để feature builder TFT điền). Idempotent theo (stock_id, date).
    """
    symbol = symbol.upper()
    async with SessionLocal() as session:
        stock_id = (
            await session.execute(select(Stock.id).where(Stock.symbol == symbol))
        ).scalar_one_or_none()
        if stock_id is None:
            logger.warning("build_daily_sentiment: chưa seed mã %s", symbol)
            return 0

        rows = (
            await session.execute(
                select(News.published_at, News.sentiment_score)
                .join(NewsStock, NewsStock.news_id == News.id)
                .where(NewsStock.stock_id == stock_id)
                .where(News.sentiment_score.is_not(None))
            )
        ).all()

        # Gộp theo ngày trong Python (trung lập dialect SQLite/PG).
        by_day: dict[object, list[float]] = {}
        for published_at, score in rows:
            by_day.setdefault(published_at.date(), []).append(score)

        records = [
            {
                "stock_id": stock_id,
                "date": day,
                "sentiment_agg": sum(scores) / len(scores),
                "news_count": len(scores),
            }
            for day, scores in by_day.items()
        ]
        written = await upsert(
            session,
            DailySentiment,
            records,
            index_elements=["stock_id", "date"],
            update_cols=["sentiment_agg", "news_count"],
        )
    logger.info("build_daily_sentiment %s: %d ngày có tin", symbol, written)
    return written


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Chấm sentiment + gộp daily_sentiment 1 mã")
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    async def _run() -> None:
        await score_news(args.symbol)
        n = await build_daily_sentiment(args.symbol)
        print(f"✓ {args.symbol}: {n} ngày daily_sentiment")

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
