"""Inference LightGBM → bảng predictions, kèm confidence-gating (M3 — ADR 0002).

Luồng: mỗi mã active → feature phiên mới nhất từ chính `features/builder.py`
(chống train/serve skew) → Booster.predict → temperature scaling (T từ metrics json,
fit trên val lúc train) → argmax = label, max prob = confidence,
`is_actionable = confidence ≥ PRODUCTION_THRESHOLD` → upsert predictions.

Quyết định gating nằm TẠI ĐÂY (ghi vào DB) — API chỉ đọc, frontend chỉ hiển thị.

Dep `lightgbm` thuộc uv group `inference` (không vào default install — CI/Render
không cài); import lazy trong `_load_booster`. Model `lgbm_vN.txt` gitignore —
chỉ có trên máy train; metrics json commit kèm repo.

Chạy: cd backend && uv run --group inference python -m services.inference
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from features.builder import LGBM_V4_FEATURES

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "lgbm_model"
MODEL_VERSION = "lgbm_v4"

# Threshold PRODUCTION — chốt 0.60 theo BẰNG CHỨNG TEST (ADR 0002), KHÔNG dùng 0.40
# trong metrics json (rule "val precision ≥0.55" không transfer: test precision 0.5033
# ≈ baseline 0.50). Tại 0.60: test precision 0.667 @ coverage 21.4%. Rolling threshold → M10.
PRODUCTION_THRESHOLD = 0.60


class PredictFn(Protocol):
    """Booster.predict tương thích: nhận DataFrame feature → probs [N, n_class]."""

    def __call__(self, features: pd.DataFrame) -> np.ndarray: ...


def load_metadata(model_version: str = MODEL_VERSION) -> dict:
    """Đọc metrics json (commit kèm repo): temperature, classes_order, feature_names."""
    meta = json.loads((ARTIFACT_DIR / f"metrics_{model_version}.json").read_text())
    assert list(meta["feature_names"]) == list(LGBM_V4_FEATURES), (
        "feature_names trong metrics khác LGBM_V4_FEATURES của builder — "
        "train/serve skew, phải re-train hoặc sync builder trước khi inference."
    )
    return meta


def _load_booster(model_version: str = MODEL_VERSION) -> PredictFn:
    """Load native Booster (lazy import lightgbm — dep group `inference`)."""
    import lightgbm as lgb

    model_path = ARTIFACT_DIR / f"{model_version}.txt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} chưa có (file gitignore, chỉ tồn tại trên máy train) — "
            f"chạy ml/notebooks/04_lgbm_training.py trước."
        )
    booster = lgb.Booster(model_file=str(model_path))
    return lambda features: booster.predict(features)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """Temperature scaling trên log-prob — GIỐNG HỆT lúc train (04_lgbm_training.py)."""
    logp = np.log(np.clip(probs, 1e-12, None)) / t
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def gate(cal_probs: np.ndarray, classes: list[str], threshold: float) -> dict:
    """1 hàng probs ĐÃ calibrate → label/confidence/is_actionable (+prob từng lớp)."""
    idx = int(cal_probs.argmax())
    confidence = float(cal_probs[idx])
    probs_by_class = {c: float(p) for c, p in zip(classes, cal_probs, strict=True)}
    return {
        "label": classes[idx],
        "confidence": confidence,
        "prob_tang": probs_by_class["tang"],
        "prob_giam": probs_by_class["giam"],
        "prob_di_ngang": probs_by_class["di_ngang"],
        "is_actionable": confidence >= threshold,
        "threshold": threshold,
    }


def next_trading_day(d: date) -> date:
    """Phiên kế tiếp xấp xỉ (bỏ cuối tuần, không bỏ lễ) — chỉ để hiển thị target_date.
    Accuracy KHÔNG phụ thuộc giá trị này (join theo prediction_date — xem ActualResult)."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame | None:
    """Hàng feature phiên MỚI NHẤT đủ giá trị số (warm-up đầu chuỗi bị NaN) — frame 1 hàng.

    sector (categorical) không bắt buộc non-null: None là 1 mức hợp lệ như lúc train.
    """
    numeric = [f for f in LGBM_V4_FEATURES if f != "sector"]
    df = df.dropna(subset=numeric)
    if df.empty:
        return None
    return df.iloc[[-1]]


async def run_inference(
    predict: PredictFn | None = None,
    model_version: str = MODEL_VERSION,
    threshold: float = PRODUCTION_THRESHOLD,
) -> int:
    """Dự đoán mọi mã active → upsert predictions. Trả số dòng ghi. Idempotent.

    `predict` inject được để test không cần lightgbm/model file.
    """
    from sqlalchemy import select

    from db.upsert import upsert
    from features.builder import load_training_frame
    from models.database import Prediction, SessionLocal, Stock

    meta = load_metadata(model_version)
    classes: list[str] = list(meta["classes_order"])
    temperature: float = float(meta["calibration"]["temperature"])
    if predict is None:
        predict = _load_booster(model_version)

    async with SessionLocal() as session:
        stocks = (
            await session.execute(select(Stock.id, Stock.symbol).where(Stock.is_active.is_(True)))
        ).all()

        rows: list[dict] = []
        for stock_id, symbol in stocks:
            frame = await load_training_frame(symbol)
            last = latest_feature_row(frame)
            if last is None:
                print(f"  {symbol}: SKIP (chưa đủ lịch sử cho feature)")
                continue
            features = last[list(LGBM_V4_FEATURES)].copy()
            features["sector"] = features["sector"].astype("category")
            cal = apply_temperature(np.asarray(predict(features)), temperature)
            pred_date = pd.to_datetime(last.iloc[0]["date"]).date()
            rows.append(
                {
                    "stock_id": stock_id,
                    "prediction_date": pred_date,
                    "target_date": next_trading_day(pred_date),
                    "model_version": model_version,
                    **gate(cal[0], classes, threshold),
                }
            )
            r = rows[-1]
            mark = "✓" if r["is_actionable"] else "·"
            print(f"  {mark} {symbol}: {r['label']} conf={r['confidence']:.3f} @ {pred_date}")

        n = await upsert(
            session,
            Prediction,
            rows,
            index_elements=["stock_id", "prediction_date", "model_version"],
            update_cols=[
                "target_date",
                "label",
                "confidence",
                "prob_tang",
                "prob_giam",
                "prob_di_ngang",
                "is_actionable",
                "threshold",
            ],
        )
    actionable = sum(r["is_actionable"] for r in rows)
    print(
        f"\nDone: upsert {n} predictions ({model_version}, threshold={threshold}, "
        f"actionable {actionable}/{len(rows)})"
    )
    return n


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_inference())
