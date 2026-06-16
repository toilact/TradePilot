"""Drift monitor — giám sát model production trên rolling 30 phiên (M10 model governance).

Biến hệ thống "tự hành" thành "tự giám sát": cuối pipeline daily (bước 7), so các dự đoán
gần đây với kết quả thực tế và phát Telegram "cần retrain" khi phát hiện drift/tiền-collapse.

Ngưỡng (chốt 2026-06-16 — cân bằng 3 tín hiệu):
  - precision trên tập dám đoán (rolling 30) < 0.50  → kém baseline luôn-đi-ngang.
  - 1 nhãn chiếm > 70% pred-dist                       → tiền-collapse (TFT từng collapse 100%).
  - coverage < 10%                                      → model im lặng (gating chặn gần hết).

Guard mẫu tối thiểu (chống BÁO GIẢ lúc data còn mỏng ~3 phiên): chỉ chấm precision khi đủ
MIN_ACTIONABLE_SCORED tín hiệu đã có actual; chỉ chấm pred-dist/coverage khi đủ MIN_PREDS_FOR_DIST.
Dưới ngưỡng mẫu → ghi `notes`, KHÔNG bắn alert.

`compute_drift` là hàm THUẦN (test bằng data giả).
Chạy thử alert: `uv run python -m services.drift_monitor --simulate`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import ActualResult, Prediction

logger = structlog.get_logger(__name__)

LABELS = ("di_ngang", "giam", "tang")

# Cửa sổ rolling = 30 phiên giao dịch (30 prediction_date distinct của version production).
ROLLING_SESSIONS = 30
# Guard mẫu tối thiểu — dưới ngưỡng thì bỏ qua check tương ứng (không alert).
MIN_ACTIONABLE_SCORED = 30  # tín hiệu actionable ĐÃ có actual mới đủ tin để chấm precision
MIN_PREDS_FOR_DIST = 60  # tổng dự đoán đủ để pred-dist/coverage có ý nghĩa
# Ngưỡng cảnh báo.
PRECISION_FLOOR = 0.50
PRED_DIST_CEILING = 0.70
COVERAGE_FLOOR = 0.10


@dataclass
class DriftReport:
    version: str
    n_sessions: int
    n_preds: int
    coverage: float | None
    pred_dist: dict[str, float]
    precision_on_actionable: float | None
    n_actionable_scored: int
    actual_dist: dict[str, float]
    drift_detected: bool
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def compute_drift(preds, actual_map: dict, current_version: str) -> DriftReport:
    """Tính drift trên 30 phiên gần nhất của `current_version`. Hàm THUẦN (không DB).

    `preds`: iterable dòng có .stock_id/.prediction_date/.label/.model_version/.is_actionable.
    `actual_map`: {(stock_id, prediction_date): actual_label} — khớp theo NGÀY T (prediction_date),
    đúng contract `read_api.get_accuracy` (actual của dự đoán ngày T là nhãn thực tế tại T).
    """
    cur = [p for p in preds if p.model_version == current_version]
    sessions = sorted({p.prediction_date for p in cur})
    window_dates = set(sessions[-ROLLING_SESSIONS:])
    window = [p for p in cur if p.prediction_date in window_dates]

    n_preds = len(window)
    n_sessions = len(window_dates)

    pred_counts = Counter(p.label for p in window)
    pred_dist = {
        lbl: round(pred_counts.get(lbl, 0) / n_preds, 4) if n_preds else 0.0 for lbl in LABELS
    }

    n_actionable = sum(1 for p in window if p.is_actionable)
    coverage = round(n_actionable / n_preds, 4) if n_preds else None

    scored = [
        (p, actual_map[(p.stock_id, p.prediction_date)])
        for p in window
        if p.is_actionable and (p.stock_id, p.prediction_date) in actual_map
    ]
    n_actionable_scored = len(scored)
    precision = (
        round(sum(p.label == a for p, a in scored) / n_actionable_scored, 4)
        if n_actionable_scored
        else None
    )

    actuals_in_window = [
        actual_map[(p.stock_id, p.prediction_date)]
        for p in window
        if (p.stock_id, p.prediction_date) in actual_map
    ]
    actual_counts = Counter(actuals_in_window)
    actual_dist = {
        lbl: round(actual_counts.get(lbl, 0) / len(actuals_in_window), 4)
        if actuals_in_window
        else 0.0
        for lbl in LABELS
    }

    reasons: list[str] = []
    notes: list[str] = []

    # reasons KHÔNG dùng ký tự `<`/`>` — chúng đi vào Telegram (notifier parse_mode=HTML),
    # `<` sẽ bị hiểu là tag → 400 Bad Request. Dùng chữ "ngưỡng"/"dưới" thay cho dấu so sánh.
    # --- pred-dist + coverage (gated by tổng số dự đoán) ---
    if n_preds >= MIN_PREDS_FOR_DIST:
        top_label, top_frac = max(pred_dist.items(), key=lambda kv: kv[1])
        if top_frac > PRED_DIST_CEILING:
            reasons.append(
                f"pred-dist tiền-collapse: {top_label} {top_frac:.0%} vượt {PRED_DIST_CEILING:.0%}"
            )
        if coverage is not None and coverage < COVERAGE_FLOOR:
            reasons.append(f"coverage thấp: {coverage:.1%} dưới {COVERAGE_FLOOR:.0%}")
    else:
        notes.append(f"insufficient_dist_data: {n_preds}/{MIN_PREDS_FOR_DIST} dự đoán")

    # --- precision trên tập dám đoán (gated by số actionable đã chấm) ---
    if n_actionable_scored >= MIN_ACTIONABLE_SCORED:
        if precision is not None and precision < PRECISION_FLOOR:
            reasons.append(
                f"precision dám-đoán: {precision:.1%} dưới {PRECISION_FLOOR:.0%} (≈ baseline)"
            )
    else:
        notes.append(
            f"insufficient_precision_data: {n_actionable_scored}/{MIN_ACTIONABLE_SCORED} đã chấm"
        )

    return DriftReport(
        version=current_version,
        n_sessions=n_sessions,
        n_preds=n_preds,
        coverage=coverage,
        pred_dist=pred_dist,
        precision_on_actionable=precision,
        n_actionable_scored=n_actionable_scored,
        actual_dist=actual_dist,
        drift_detected=bool(reasons),
        reasons=reasons,
        notes=notes,
    )


def build_drift_message(report: DriftReport) -> str:
    """Tin Telegram tổng kết drift. Hàm THUẦN để test."""
    head = (
        "⚠️ TradePilot DRIFT — CẦN RETRAIN"
        if report.drift_detected
        else "✅ TradePilot drift check: ổn"
    )
    lines = [head, f"model {report.version} · {report.n_sessions} phiên / {report.n_preds} dự đoán"]
    if report.coverage is not None:
        lines.append(f"coverage: {report.coverage:.1%}")
    prec = report.precision_on_actionable
    if prec is not None:
        lines.append(f"precision dám-đoán: {prec:.1%} (n={report.n_actionable_scored})")
    lines.append("pred-dist: " + ", ".join(f"{k} {v:.0%}" for k, v in report.pred_dist.items()))
    if report.reasons:
        lines.append("Tín hiệu drift:")
        lines.extend(f"• {r}" for r in report.reasons)
    if report.notes:
        lines.extend(f"({n})" for n in report.notes)
    return "\n".join(lines)


async def check_drift(session: AsyncSession) -> DriftReport:
    """Load predictions (version production) + actuals từ DB → compute_drift."""
    from services.inference import MODEL_VERSION

    preds = (
        await session.execute(
            select(
                Prediction.stock_id,
                Prediction.prediction_date,
                Prediction.label,
                Prediction.model_version,
                Prediction.is_actionable,
            )
        )
    ).all()
    actuals = (
        await session.execute(select(ActualResult.stock_id, ActualResult.date, ActualResult.label))
    ).all()
    actual_map = {(sid, d): lbl for sid, d, lbl in actuals}
    return compute_drift(preds, actual_map, MODEL_VERSION)


async def run_drift_check(notify=None) -> DriftReport:
    """Bước pipeline: chấm drift, CHỈ gửi Telegram khi drift_detected. KHÔNG raise."""
    from models.database import SessionLocal

    if notify is None:
        from services.notifier import send_telegram

        notify = send_telegram

    async with SessionLocal() as session:
        report = await check_drift(session)

    logger.info(
        "drift_check",
        version=report.version,
        drift=report.drift_detected,
        reasons=report.reasons,
        notes=report.notes,
        n_preds=report.n_preds,
        n_scored=report.n_actionable_scored,
    )
    if report.drift_detected:
        try:
            await notify(build_drift_message(report))
        except Exception:  # notifier vốn không raise — phòng notify inject lỗi
            logger.exception("drift_notify_failed")
    return report


def _simulate_report() -> DriftReport:
    """Data giả COLLAPSE để demo alert (DoD): 30 phiên × 30 mã, đoán 100% di_ngang, actual lệch."""
    from datetime import date, timedelta
    from types import SimpleNamespace

    base = date(2026, 1, 1)
    preds = []
    actual_map: dict = {}
    for s in range(ROLLING_SESSIONS):
        d = base + timedelta(days=s)
        for stock_id in range(1, 31):
            preds.append(
                SimpleNamespace(
                    stock_id=stock_id,
                    prediction_date=d,
                    label="di_ngang",
                    model_version="lgbm_v4",
                    is_actionable=True,
                )
            )
            actual_map[(stock_id, d)] = "tang" if stock_id % 2 else "giam"
    return compute_drift(preds, actual_map, "lgbm_v4")


def _main() -> None:
    from logging_config import configure_logging
    from services.notifier import send_telegram

    configure_logging()
    parser = argparse.ArgumentParser(description="Drift monitor TradePilot")
    parser.add_argument(
        "--simulate", action="store_true", help="Bắn alert drift bằng data giả collapse"
    )
    parser.add_argument(
        "--live", action="store_true", help="Chấm drift thật từ DB (không alert nếu ổn)"
    )
    args = parser.parse_args()

    if args.simulate:
        report = _simulate_report()
        msg = build_drift_message(report)
        print(msg)
        ok = asyncio.run(send_telegram(msg))
        print("✓ Đã gửi Telegram" if ok else "✗ Gửi thất bại — xem log (token/chat_id/mạng)")
    elif args.live:
        report = asyncio.run(run_drift_check())
        print(build_drift_message(report))
    else:
        parser.error("dùng --simulate (demo alert) hoặc --live (chấm DB thật)")


if __name__ == "__main__":
    _main()
