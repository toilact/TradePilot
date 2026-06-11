"""Seed predictions STUB từ giá thật — để frontend chạy end-to-end TRƯỚC khi có TFT checkpoint.

⚠️ ĐÂY KHÔNG PHẢI DỰ ĐOÁN THẬT. Quy tắc stub đơn giản: so MA7 vs MA20 phiên gần nhất
(xu hướng ngắn vs trung hạn) → tang/giam/di_ngang. model_version="stub_v0" để phân biệt rõ
với checkpoint TFT thật sau này (tft_v1). Khi có TFT, chạy inference ghi đè (model_version khác).

Chạy: cd backend && uv run python -m scripts.seed_predictions
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.stock_seed import VN30  # noqa: E402 — sau sys.path setup

MODEL_VERSION = "stub_v0"  # KHÔNG phải TFT — phân biệt rõ với tft_v1


def next_trading_day(d: date) -> date:
    """Phiên giao dịch kế tiếp (T+1) — bỏ cuối tuần. KHÔNG bỏ lễ (xấp xỉ; accuracy job mới
    khớp với phiên THẬT có trong price_history, không phụ thuộc giá trị này — xem ActualResult)."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:  # 5=thứ Bảy, 6=Chủ nhật
        nxt += timedelta(days=1)
    return nxt


def stub_label(ma7: float, ma20: float) -> str:
    """Stub: MA7 vs MA20. >+0.5% → tang, <-0.5% → giam, còn lại di_ngang."""
    from services.labeling import LABEL_DOWN, LABEL_FLAT, LABEL_UP

    if ma20 <= 0:
        return LABEL_FLAT
    diff = (ma7 - ma20) / ma20
    if diff > 0.005:
        return LABEL_UP
    if diff < -0.005:
        return LABEL_DOWN
    return LABEL_FLAT


async def main() -> None:
    from sqlalchemy import select

    from db.upsert import upsert
    from features.builder import load_training_frame
    from models.database import Prediction, SessionLocal, Stock

    async with SessionLocal() as session:
        stock_ids = dict((await session.execute(select(Stock.symbol, Stock.id))).all())

        rows: list[dict] = []
        for symbol in VN30:
            stock_id = stock_ids.get(symbol)
            if stock_id is None:
                print(f"  {symbol}: SKIP (chưa có trong bảng stocks)")
                continue
            df = await load_training_frame(symbol)
            df = df.dropna(subset=["ma7", "ma20"])
            if df.empty:
                print(f"  {symbol}: SKIP (không đủ data MA)")
                continue
            last = df.iloc[-1]
            pred_date = pd.to_datetime(last["date"]).date()
            rows.append(
                {
                    "stock_id": stock_id,
                    "prediction_date": pred_date,
                    "target_date": next_trading_day(pred_date),  # T+1 phiên (bỏ cuối tuần)
                    "label": stub_label(float(last["ma7"]), float(last["ma20"])),
                    "confidence": 0.5,  # stub — không có xác suất thật
                    "model_version": MODEL_VERSION,
                }
            )
            print(f"  {symbol}: {rows[-1]['label']} @ {pred_date}")

        n = await upsert(
            session,
            Prediction,
            rows,
            index_elements=["stock_id", "prediction_date", "model_version"],
            update_cols=["target_date", "label", "confidence"],
        )
    print(f"\nDone: upsert {n} predictions (model_version={MODEL_VERSION})")


if __name__ == "__main__":
    asyncio.run(main())
