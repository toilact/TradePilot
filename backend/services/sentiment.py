"""PhoBERT sentiment inference + tổng hợp daily_sentiment.

Backend CHỈ inference — model train ở ml/ rồi export sang ml/artifacts/phobert_sentiment/.

Mắt xích (thứ tự): score_news (chấm điểm tin) → build_daily_sentiment (gộp ngày/mã).
- score_text/score_texts (M8): điểm = p_pos − p_neg ∈ [-1,1] từ PhoBERT fine-tuned. Thiếu
  artifact/dep torch → degrade về 0.0 (pipeline vẫn chạy như trước khi có model).
- Thay stub→PhoBERT thì điểm tin CŨ (đang 0.0) phải re-score: `score_news(rescore=True)` /
  `rescore_all()` (xem CLI `--all`).

Bất biến:
  - Ngày không tin → sentiment_agg = 0 (feature builder điền; aggregate chỉ ghi ngày CÓ tin).
  - Chống leakage + không mất tin nghỉ: gộp theo NGÀY GIAO DỊCH HIỆU LỰC (effective_trading_day)
    — tin >= 16:00 phiên T dồn sang phiên kế; tin cuối tuần/lễ dồn vào phiên giao dịch kế tiếp.
  - sentiment_agg ∈ [-1, 1]; chấm trên TITLE (content=None theo governance).
  - sentiment_extreme = signed max-abs điểm trong ngày (input lớp gating M8 Option C).

Chạy: cd backend && uv run --group inference python -m services.sentiment --all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from bisect import bisect_left
from datetime import date as date_cls
from datetime import datetime, timedelta
from datetime import time as time_cls
from pathlib import Path
from typing import Protocol

from sqlalchemy import select

from db.upsert import upsert
from models.database import DailySentiment, News, NewsStock, PriceHistory, SessionLocal, Stock

logger = logging.getLogger(__name__)

# Chống leakage: tin biết TRƯỚC 16:00 phiên T mới được dùng cho feature phiên T. Tin published
# >= 16:00 (sau giờ chốt) → sớm nhất ảnh hưởng phiên giao dịch KẾ TIẾP. Xem effective_trading_day.
MARKET_CLOSE = time_cls(16, 0)

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


def effective_trading_day(published_at: datetime, trading_days: list[date_cls]) -> date_cls | None:
    """Tin `published_at` (naive, giờ VN) → ngày GIAO DỊCH sớm nhất được phép dùng tin đó.

    Chống leakage: tin biết TRƯỚC 16:00 phiên T mới được dùng cho feature phiên T.
      - published_at < 16:00  → ứng viên = ngày lịch của tin (có thể dùng ngay phiên đó).
      - published_at >= 16:00 → ứng viên = ngày lịch + 1 (sau giờ chốt; T+1 mới biết).
    Rồi ánh xạ ứng viên → ngày giao dịch ĐẦU TIÊN >= ứng viên trong `trading_days` (đã sort)
    qua bisect_left — gom tin tối thứ Sáu / cuối tuần / lễ vào phiên kế tiếp (tránh left-join
    drop ở builder vì ngày nghỉ không có giá). None nếu không còn phiên nào >= ứng viên
    (tin sau phiên mới nhất → "chờ" tới khi phiên đó xuất hiện trong price_history).
    """
    candidate = published_at.date()
    if published_at.time() >= MARKET_CLOSE:
        candidate = candidate + timedelta(days=1)
    i = bisect_left(trading_days, candidate)
    if i >= len(trading_days):
        return None
    return trading_days[i]


def aggregate_daily(rows: list[tuple[datetime, float]], trading_days: list[date_cls]) -> list[dict]:
    """(published_at, score) → bản ghi daily_sentiment theo NGÀY GIAO DỊCH hiệu lực.

    Hàm THUẦN (không DB) để test. Mỗi ngày: sentiment_agg = trung bình; news_count = số tin;
    sentiment_extreme = điểm SIGNED MAX-ABS (tin |score| lớn nhất, giữ dấu) — input gating M8.
    Tin ánh xạ về None (sau phiên mới nhất) bị bỏ qua.
    """
    by_day: dict[date_cls, list[float]] = {}
    for published_at, score in rows:
        day = effective_trading_day(published_at, trading_days)
        if day is None:
            continue
        by_day.setdefault(day, []).append(score)
    return [
        {
            "date": day,
            "sentiment_agg": sum(scores) / len(scores),
            "news_count": len(scores),
            "sentiment_extreme": max(scores, key=abs),
        }
        for day, scores in by_day.items()
    ]


async def build_daily_sentiment(symbol: str) -> int:
    """Gộp news.sentiment_score → daily_sentiment theo (stock_id, NGÀY GIAO DỊCH). Trả số ngày ghi.

    Tin gom theo ngày giao dịch hiệu lực (effective_trading_day — cutoff 16:00 + dồn ngày nghỉ
    vào phiên kế tiếp), KHÔNG theo ngày lịch thô (chống leakage + không mất tin cuối tuần).
    Chỉ ghi ngày CÓ tin (ngày không tin → 0, builder điền). Idempotent theo (stock_id, date).
    """
    symbol = symbol.upper()
    async with SessionLocal() as session:
        stock_id = (
            await session.execute(select(Stock.id).where(Stock.symbol == symbol))
        ).scalar_one_or_none()
        if stock_id is None:
            logger.warning("build_daily_sentiment: chưa seed mã %s", symbol)
            return 0

        trading_days = (
            (
                await session.execute(
                    select(PriceHistory.date)
                    .where(PriceHistory.stock_id == stock_id)
                    .order_by(PriceHistory.date)
                )
            )
            .scalars()
            .all()
        )

        rows = (
            await session.execute(
                select(News.published_at, News.sentiment_score)
                .join(NewsStock, NewsStock.news_id == News.id)
                .where(NewsStock.stock_id == stock_id)
                .where(News.sentiment_score.is_not(None))
            )
        ).all()

        records = [
            {"stock_id": stock_id, **rec} for rec in aggregate_daily(list(rows), list(trading_days))
        ]
        written = await upsert(
            session,
            DailySentiment,
            records,
            index_elements=["stock_id", "date"],
            update_cols=["sentiment_agg", "news_count", "sentiment_extreme"],
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
