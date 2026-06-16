"""Test drift_monitor (M10) — compute_drift là hàm THUẦN, test bằng data giả.

Bất biến kiểm tra:
- 3 tín hiệu drift kích hoạt ĐỘC LẬP (precision < 0.50 / 1 lớp > 70% pred-dist / coverage < 10%).
- Guard mẫu tối thiểu: data mỏng → KHÔNG alert (dù precision tệ), chỉ ghi notes.
- Cửa sổ rolling chỉ tính 30 phiên gần nhất; chỉ tính version production (bỏ stub).
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

from services.drift_monitor import (
    LABELS,
    _simulate_report,
    build_drift_message,
    compute_drift,
)

V = "lgbm_v4"


def _other(label: str) -> str:
    return "tang" if label != "tang" else "giam"


def _mk_preds(n_sessions, stocks=30, version=V, label_fn=None, actionable_fn=None):
    """Sinh preds giả: mặc định nhãn xoay vòng 3 lớp (không collapse), tất cả actionable."""
    base = date(2026, 1, 1)
    preds = []
    for s in range(n_sessions):
        d = base + timedelta(days=s)
        for sid in range(1, stocks + 1):
            label = label_fn(s, sid) if label_fn else LABELS[(s + sid) % 3]
            actionable = actionable_fn(s, sid) if actionable_fn else True
            preds.append(
                SimpleNamespace(
                    stock_id=sid,
                    prediction_date=d,
                    label=label,
                    model_version=version,
                    is_actionable=actionable,
                )
            )
    return preds


def _all_correct(preds):
    return {(p.stock_id, p.prediction_date): p.label for p in preds}


def test_healthy_no_drift():
    preds = _mk_preds(30)
    report = compute_drift(preds, _all_correct(preds), V)
    assert not report.drift_detected
    assert report.reasons == []
    assert report.n_sessions == 30 and report.n_preds == 900
    assert report.precision_on_actionable == 1.0
    assert report.coverage == 1.0


def test_precision_drift():
    preds = _mk_preds(30)
    actual_map = {(p.stock_id, p.prediction_date): _other(p.label) for p in preds}  # 100% sai
    report = compute_drift(preds, actual_map, V)
    assert report.drift_detected
    assert any("precision" in r for r in report.reasons)
    assert report.precision_on_actionable == 0.0


def test_pred_dist_collapse_drift():
    preds = _mk_preds(30, label_fn=lambda s, sid: "di_ngang")  # đoán 100% di_ngang
    actual_map = {
        (p.stock_id, p.prediction_date): "di_ngang" for p in preds
    }  # precision 1.0 (cô lập)
    report = compute_drift(preds, actual_map, V)
    assert report.drift_detected
    assert any("pred-dist" in r for r in report.reasons)
    assert not any("precision" in r for r in report.reasons)  # precision vẫn tốt → chỉ pred-dist


def test_low_coverage_drift():
    preds = _mk_preds(30, actionable_fn=lambda s, sid: sid <= 2)  # 2/30 ≈ 6.7% < 10%
    report = compute_drift(preds, _all_correct(preds), V)
    assert report.drift_detected
    assert any("coverage" in r for r in report.reasons)
    assert not any("precision" in r for r in report.reasons)


def test_thin_data_no_false_alert():
    """~1 phiên: dù precision = 0, mẫu chưa đủ → KHÔNG alert (chống báo giả lúc mới live)."""
    preds = _mk_preds(1, actionable_fn=lambda s, sid: sid <= 3)
    actual_map = {
        (p.stock_id, p.prediction_date): _other(p.label) for p in preds if p.stock_id <= 3
    }
    report = compute_drift(preds, actual_map, V)
    assert not report.drift_detected
    assert any("insufficient_dist_data" in n for n in report.notes)
    assert any("insufficient_precision_data" in n for n in report.notes)


def test_rolling_window_keeps_last_30():
    preds = _mk_preds(40)
    report = compute_drift(preds, _all_correct(preds), V)
    assert report.n_sessions == 30  # chỉ 30 phiên cuối
    assert report.n_preds == 900


def test_ignores_other_version():
    preds = _mk_preds(30)
    stub = [
        SimpleNamespace(
            stock_id=99,
            prediction_date=p.prediction_date,
            label="di_ngang",
            model_version="stub_v0",
            is_actionable=True,
        )
        for p in preds
    ]
    report = compute_drift(preds + stub, _all_correct(preds), V)
    assert report.version == V
    assert report.n_preds == 900  # stub_v0 không được tính


def test_simulate_report_triggers_alert_message():
    report = _simulate_report()
    assert report.drift_detected
    msg = build_drift_message(report)
    assert "DRIFT" in msg and "CẦN RETRAIN" in msg
    assert "pred-dist" in msg
