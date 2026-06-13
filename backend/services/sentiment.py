"""PhoBERT sentiment inference + tổng hợp daily_sentiment.

Backend CHỈ inference — model train ở ml/ rồi export sang ml/artifacts/phobert_sentiment/.

Mắt xích (thứ tự): score_news (chấm điểm tin) → build_daily_sentiment (gộp ngày/mã).
- score_text/score_texts (M8): điểm = p_pos − p_neg ∈ [-1,1] từ PhoBERT fine-tuned. Thiếu
  artifact/dep torch → degrade về 0.0 (pipeline vẫn chạy như trước khi có model).
- Thay stub→PhoBERT thì điểm tin CŨ (đang 0.0) phải re-score: `score_news(rescore=True)` /
  `rescore_all()` (xem CLI `--all`).

Bất biến:
  - Ngày không tin → sentiment_agg = 0 (feature builder điền; aggregate chỉ ghi ngày CÓ tin).
  - Chống leakage: gộp theo NGÀY của published_at (đã chuẩn hoá giờ VN ở crawler).
  - sentiment_agg ∈ [-1, 1]; chấm trên TITLE (content=None theo governance).

Chạy: cd backend && uv run --group inference python -m services.sentiment --all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Protocol

from sqlalchemy import select

from db.upsert import upsert
from models.database import DailySentiment, News, NewsStock, SessionLocal, Stock

logger = logging.getLogger(__name__)

# Artifact PhoBERT (save_pretrained: model + tokenizer) — gitignore, chỉ tải về máy chạy
# inference. Dep torch/transformers thuộc uv group `inference` (KHÔNG vào default install /
# Render); import LAZY trong _phobert_scorer.
MODEL_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "phobert_sentiment"

# Notebook lưu id2label {0:neg,1:neu,2:pos}; điểm = p_pos − p_neg ∈ [-1, 1] (đọc id2label từ
# config model để không phụ thuộc thứ tự cứng).
_BATCH = 32
_MAX_LEN = 128


class ScoreFn(Protocol):
    """Chấm điểm sentiment hàng loạt: list title → list điểm [-1,1] cùng độ dài."""

    def __call__(self, texts: list[str]) -> list[float]: ...


_default_scorer_cache: ScoreFn | None = None
_default_scorer_loaded = False


def _phobert_scorer() -> ScoreFn | None:
    """Dựng scorer PhoBERT (lazy import torch/transformers). None nếu thiếu artifact/dep
    → caller degrade về 0.0 (pipeline vẫn chạy trước khi tải model về)."""
    if not (MODEL_DIR / "config.json").exists():
        logger.warning("PhoBERT artifact chưa có (%s) → sentiment = 0.0 (degrade)", MODEL_DIR)
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        logger.warning("torch/transformers chưa cài (%s) → sentiment = 0.0 (degrade)", exc)
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    label2idx = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    pos_i, neg_i = label2idx.get("pos"), label2idx.get("neg")
    if pos_i is None or neg_i is None:
        logger.warning("model thiếu nhãn pos/neg (id2label=%s) → 0.0", model.config.id2label)
        return None

    def scorer(texts: list[str]) -> list[float]:
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), _BATCH):
                chunk = texts[i : i + _BATCH]
                enc = tokenizer(
                    chunk, truncation=True, padding=True, max_length=_MAX_LEN, return_tensors="pt"
                )
                probs = torch.softmax(model(**enc).logits, dim=-1)
                scores.extend((probs[:, pos_i] - probs[:, neg_i]).tolist())
        return scores

    logger.info("PhoBERT sentiment loaded từ %s (pos=%d neg=%d)", MODEL_DIR, pos_i, neg_i)
    return scorer


def _default_scorer() -> ScoreFn | None:
    """Scorer PhoBERT cache module-level (load 1 lần). None = degrade về 0.0."""
    global _default_scorer_cache, _default_scorer_loaded
    if not _default_scorer_loaded:
        _default_scorer_cache = _phobert_scorer()
        _default_scorer_loaded = True
    return _default_scorer_cache


def score_texts(texts: list[str], scorer: ScoreFn | None = None) -> list[float]:
    """Điểm sentiment [-1,1] cho nhiều title. `scorer` inject được để test không cần torch.

    scorer=None → PhoBERT cache; thiếu artifact/dep → trả 0.0 hết (degrade, pipeline vẫn chạy).
    """
    if not texts:
        return []
    if scorer is None:
        scorer = _default_scorer()
    if scorer is None:
        return [0.0] * len(texts)
    return scorer(texts)


def score_text(text: str, scorer: ScoreFn | None = None) -> float:
    """Điểm sentiment [-1,1] cho 1 title (tiện cho 1 câu / test). Xem score_texts."""
    return score_texts([text], scorer=scorer)[0]


async def score_news(
    symbol: str | None = None,
    rescore: bool = False,
    scorer: ScoreFn | None = None,
) -> int:
    """Chấm sentiment_score cho tin (theo title), chấm batch. Trả số tin đã chấm.

    `symbol=None` → mọi mã; có symbol → chỉ tin map tới mã đó.
    `rescore=False` → chỉ tin CHƯA có điểm (NULL); `rescore=True` → chấm LẠI tất cả
    (cần khi thay stub→PhoBERT: tin cũ đang mang điểm 0.0 của stub). `scorer` inject để test.
    """
    async with SessionLocal() as session:
        stmt = select(News.id, News.title)
        if not rescore:
            stmt = stmt.where(News.sentiment_score.is_(None))
        if symbol:
            stmt = (
                stmt.join(NewsStock, NewsStock.news_id == News.id)
                .join(Stock, Stock.id == NewsStock.stock_id)
                .where(Stock.symbol == symbol.upper())
            )
        rows = (await session.execute(stmt)).all()
        scores = score_texts([title or "" for _, title in rows], scorer=scorer)
        for (news_id, _), score in zip(rows, scores, strict=True):
            await session.execute(
                News.__table__.update().where(News.id == news_id).values(sentiment_score=score)
            )
        await session.commit()
    logger.info("score_news %s: chấm %d tin (rescore=%s)", symbol or "(tất cả)", len(rows), rescore)
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


async def rescore_all() -> tuple[int, int]:
    """Chấm LẠI toàn bộ news bằng scorer hiện hành + rebuild daily_sentiment mọi mã active.

    Dùng SAU khi thay stub→PhoBERT (tin cũ đang mang điểm 0.0 của stub). Trả (số tin, số mã).
    """
    n_news = await score_news(rescore=True)
    async with SessionLocal() as session:
        symbols = (
            (await session.execute(select(Stock.symbol).where(Stock.is_active.is_(True))))
            .scalars()
            .all()
        )
    for sym in symbols:
        await build_daily_sentiment(sym)
    logger.info("rescore_all: %d tin, %d mã rebuild daily_sentiment", n_news, len(symbols))
    return n_news, len(symbols)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Chấm sentiment + gộp daily_sentiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="chấm + gộp 1 mã")
    group.add_argument(
        "--all", action="store_true", help="re-score TẤT CẢ news + rebuild mọi mã (sau có PhoBERT)"
    )
    args = parser.parse_args()

    async def _run() -> None:
        if args.all:
            n_news, n_sym = await rescore_all()
            print(f"✓ rescore_all: {n_news} tin, {n_sym} mã daily_sentiment")
        else:
            await score_news(args.symbol)
            n = await build_daily_sentiment(args.symbol)
            print(f"✓ {args.symbol}: {n} ngày daily_sentiment")

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
