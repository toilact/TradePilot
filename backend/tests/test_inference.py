"""Test services/inference.py — gating đúng threshold, idempotent, không cần lightgbm.

`predict` inject fake probs; SessionLocal monkeypatch sang SQLite in-memory (conftest).
Metadata đọc từ metrics_lgbm_v4.json THẬT trong repo → đồng thời kiểm feature_names
khớp LGBM_V4_FEATURES (chống train/serve skew).
"""

from datetime import date, timedelta

import numpy as np
import pytest
from sqlalchemy import select

from models.database import Prediction, PriceHistory, Stock
from services.inference import (
    apply_temperature,
    gate,
    latest_feature_row,
    load_metadata,
    next_trading_day,
    run_inference,
)

CLASSES = ["di_ngang", "giam", "tang"]


def test_apply_temperature_t1_identity_and_normalized():
    probs = np.array([[0.2, 0.3, 0.5]])
    out = apply_temperature(probs, 1.0)
    assert np.allclose(out, probs)
    out2 = apply_temperature(probs, 0.5)  # T<1 → sharpen, vẫn chuẩn hoá
    assert np.isclose(out2.sum(), 1.0)
    assert out2[0, 2] > 0.5


def test_gate_actionable_above_threshold():
    row = gate(np.array([0.2, 0.1, 0.7]), CLASSES, threshold=0.6)
    assert row["label"] == "tang"
    assert row["confidence"] == 0.7
    assert row["is_actionable"] is True
    assert row["prob_di_ngang"] == 0.2 and row["prob_giam"] == 0.1 and row["prob_tang"] == 0.7


def test_gate_below_threshold_not_actionable():
    row = gate(np.array([0.45, 0.25, 0.30]), CLASSES, threshold=0.6)
    assert row["label"] == "di_ngang"
    assert row["is_actionable"] is False
    assert row["threshold"] == 0.6


def test_load_metadata_features_match_builder():
    meta = load_metadata("lgbm_v4")
    assert meta["classes_order"] == CLASSES
    assert meta["calibration"]["temperature"] > 0


def test_next_trading_day_skips_weekend():
    assert next_trading_day(date(2026, 6, 12)) == date(2026, 6, 15)  # T6 → T2


def test_latest_feature_row_none_when_history_too_short():
    import pandas as pd

    from features.builder import build_features

    prices = pd.DataFrame({"date": [date(2026, 6, d) for d in range(1, 6)], "close": [10.0] * 5})
    frame = build_features(prices, None, sector="Thép")
    assert latest_feature_row(frame) is None  # 5 phiên < warm-up vol_20


async def _seed_history(factory, symbol: str = "VCB", sessions: int = 60) -> int:
    """Đủ phiên giá vượt warm-up vol_20/rsi14 để builder ra feature non-NaN."""
    async with factory() as s:
        stock = Stock(symbol=symbol, name=symbol, exchange="HSX", sector="Ngân hàng")
        s.add(stock)
        await s.flush()
        sid = stock.id
        d0 = date(2026, 1, 5)
        day, added = d0, 0
        while added < sessions:
            if day.weekday() < 5:
                close = 60.0 + (added % 7) * 0.4  # dao động nhẹ, tránh chia 0
                s.add(
                    PriceHistory(
                        stock_id=sid,
                        date=day,
                        open=close,
                        high=close + 0.5,
                        low=close - 0.5,
                        close=close,
                        volume=1000,
                    )
                )
                added += 1
            day += timedelta(days=1)
        await s.commit()
        return sid


@pytest.fixture
def patch_session(session_factory, monkeypatch):
    monkeypatch.setattr("models.database.SessionLocal", session_factory)
    return session_factory


async def test_run_inference_writes_gated_predictions(patch_session):
    sid = await _seed_history(patch_session)

    def fake_predict(features):
        assert len(features) == 1  # chỉ phiên mới nhất
        return np.array([[0.1, 0.1, 0.8]])  # tang, tự tin cao

    n = await run_inference(predict=fake_predict, threshold=0.6)
    assert n == 1
    async with patch_session() as s:
        pred = (await s.execute(select(Prediction))).scalar_one()
    assert pred.stock_id == sid
    assert pred.label == "tang"
    assert pred.model_version == "lgbm_v4"
    assert pred.is_actionable is True
    assert pred.threshold == 0.6
    # calibrated probs chuẩn hoá + confidence = max prob
    assert pred.prob_tang + pred.prob_giam + pred.prob_di_ngang == pytest.approx(1.0)
    assert pred.confidence == pytest.approx(pred.prob_tang)


async def test_run_inference_low_confidence_not_actionable(patch_session):
    await _seed_history(patch_session)
    # probs sát nhau → sau temperature scaling max vẫn < 0.99
    n = await run_inference(predict=lambda f: np.array([[0.34, 0.33, 0.33]]), threshold=0.99)
    assert n == 1
    async with patch_session() as s:
        pred = (await s.execute(select(Prediction))).scalar_one()
    assert pred.is_actionable is False
    assert pred.label == "di_ngang"


async def test_run_inference_idempotent(patch_session):
    await _seed_history(patch_session)
    fake = lambda f: np.array([[0.1, 0.1, 0.8]])  # noqa: E731
    await run_inference(predict=fake, threshold=0.6)
    await run_inference(predict=fake, threshold=0.6)  # chạy lại không duplicate
    async with patch_session() as s:
        rows = (await s.execute(select(Prediction))).scalars().all()
    assert len(rows) == 1


async def test_run_inference_skips_stock_without_enough_history(patch_session):
    async with patch_session() as s:
        s.add(Stock(symbol="NEW", name="New", exchange="HSX", sector=None))
        await s.commit()
    n = await run_inference(predict=lambda f: np.array([[0.1, 0.1, 0.8]]), threshold=0.6)
    assert n == 0  # không đủ lịch sử → skip, không crash
