"""Export panel CHẨN ĐOÁN sentiment (M8 Pha 1) — như training_panel + cột sentiment_extreme.

Khác training_panel.csv: thêm `sentiment_extreme` (signed max-abs điểm/ngày) — cần cho Tầng 1b
gating-sim. Giữ nguyên tắc: builder (LGBM_V4_FEATURES) KHÔNG gánh sentiment_extreme (tránh
skew); script query thẳng daily_sentiment rồi merge — đúng cách inference sẽ làm.

Output: ml/data/diagnostic_panel.csv
Cột: date, symbol, close, *price-features, sentiment_agg, news_count, sentiment_extreme,
     sector, label

Chạy: cd backend && uv run python -m scripts.export_diagnostic_panel
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.stock_seed import VN30  # noqa: E402

OUTPUT = _REPO / "ml" / "data" / "diagnostic_panel.csv"


async def _extreme_by_date(stock_id: int) -> pd.DataFrame:
    """daily_sentiment.sentiment_extreme theo date cho 1 mã (None → 0.0 khi merge)."""
    from sqlalchemy import select

    from models.database import DailySentiment, SessionLocal

    async with SessionLocal() as s:
        rows = (
            await s.execute(
                select(DailySentiment.date, DailySentiment.sentiment_extreme).where(
                    DailySentiment.stock_id == stock_id
                )
            )
        ).all()
    return pd.DataFrame(rows, columns=["date", "sentiment_extreme"])


async def main() -> None:
    from sqlalchemy import select

    from features.builder import load_training_frame
    from models.database import SessionLocal, Stock

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for symbol in VN30:
        print(f"  {symbol}...", end=" ", flush=True)
        df = await load_training_frame(symbol)
        if df.empty:
            print("SKIP")
            continue
        async with SessionLocal() as s:
            stock_id = (
                await s.execute(select(Stock.id).where(Stock.symbol == symbol))
            ).scalar_one_or_none()
        ext = await _extreme_by_date(stock_id) if stock_id else pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        if not ext.empty:
            ext["date"] = pd.to_datetime(ext["date"])
            df = df.merge(ext, on="date", how="left")
        else:
            df["sentiment_extreme"] = 0.0
        df["sentiment_extreme"] = df["sentiment_extreme"].fillna(0.0)
        df.insert(1, "symbol", symbol)
        frames.append(df)
        print(f"{len(df)} rows")

    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    panel.to_csv(OUTPUT, index=False)
    news_rows = int((panel["news_count"] > 0).sum())
    print(f"\nDone: {len(panel)} rows ({len(frames)} mã) → {OUTPUT}")
    print(f"news-rows (news_count>0): {news_rows} ({100 * news_rows / len(panel):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
