"""Test bất biến nhãn ±1% — chỗ bug âm thầm phá accuracy nếu sai."""

import pytest

from services.labeling import (
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    label_from_close,
    label_from_pct,
)


@pytest.mark.parametrize(
    "pct,expected",
    [
        (0.02, LABEL_UP),     # +2% → Tăng
        (0.0101, LABEL_UP),   # ngay trên ngưỡng
        (0.01, LABEL_FLAT),   # đúng +1% → Đi ngang (biên không tính)
        (0.0, LABEL_FLAT),
        (-0.01, LABEL_FLAT),  # đúng -1% → Đi ngang
        (-0.0101, LABEL_DOWN),
        (-0.05, LABEL_DOWN),
    ],
)
def test_label_from_pct(pct, expected):
    assert label_from_pct(pct) == expected


def test_label_from_close():
    assert label_from_close(100.0, 102.0) == LABEL_UP
    assert label_from_close(100.0, 100.5) == LABEL_FLAT
    assert label_from_close(100.0, 97.0) == LABEL_DOWN


def test_invalid_close_raises():
    with pytest.raises(ValueError):
        label_from_close(0.0, 100.0)
