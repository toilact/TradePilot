"""Bất biến cốt lõi: gán nhãn 3 lớp theo ngưỡng ±1%.

Dùng chung cho train (ml/), backtest, và tính actual_results (backend).
Tăng: pct > +1% | Giảm: pct < -1% | Đi ngang: trong [-1%, +1%].
"""

from __future__ import annotations

THRESHOLD = 0.01  # ±1%

LABEL_UP = "tang"
LABEL_DOWN = "giam"
LABEL_FLAT = "di_ngang"


def pct_change(close_t: float, close_t1: float) -> float:
    """% thay đổi của close T+1 so với close T (dạng tỉ lệ, vd 0.012 = +1.2%)."""
    if close_t <= 0:
        raise ValueError("close_t phải > 0")
    return (close_t1 - close_t) / close_t


def label_from_pct(pct: float) -> str:
    """Map % thay đổi → nhãn theo ngưỡng ±1%."""
    if pct > THRESHOLD:
        return LABEL_UP
    if pct < -THRESHOLD:
        return LABEL_DOWN
    return LABEL_FLAT


def label_from_close(close_t: float, close_t1: float) -> str:
    return label_from_pct(pct_change(close_t, close_t1))
