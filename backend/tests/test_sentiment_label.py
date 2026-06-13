"""Test sentiment_label: parse phản hồi LLM + map nhãn→điểm. KHÔNG gọi API thật."""

import pytest

from services.sentiment_label import build_prompt, label_to_score, parse_labels


def test_parse_labels_plain_json():
    assert parse_labels('["pos","neg","neu"]', 3) == ["pos", "neg", "neu"]


def test_parse_labels_with_code_fence():
    raw = '```json\n["pos", "neu"]\n```'
    assert parse_labels(raw, 2) == ["pos", "neu"]


def test_parse_labels_ignores_preamble_before_array():
    # LLM lỡ thêm giải thích/ví dụ trước mảng JSON → vẫn rút đúng mảng đầu tiên.
    raw = 'Đây là kết quả phân loại:\n["pos","neg","neu"]'
    assert parse_labels(raw, 3) == ["pos", "neg", "neu"]


def test_parse_labels_unknown_label_falls_back_to_neu():
    # nhãn lạ → 'neu' (an toàn, không bịa chiều)
    assert parse_labels('["pos","weird","neg"]', 3) == ["pos", "neu", "neg"]


def test_parse_labels_wrong_count_raises():
    # lệch số phần tử → raise để tránh lệch alignment title↔nhãn
    with pytest.raises(ValueError):
        parse_labels('["pos"]', 3)


def test_parse_labels_not_a_list_raises():
    with pytest.raises(ValueError):
        parse_labels('"pos"', 1)


def test_label_to_score():
    assert label_to_score("pos") == 1.0
    assert label_to_score("neg") == -1.0
    assert label_to_score("neu") == 0.0
    assert label_to_score("xxx") == 0.0  # nhãn lạ → trung tính


def test_build_prompt_contains_titles():
    p = build_prompt(["VCB lãi kỷ lục", "HPG giảm sàn"])
    assert "VCB lãi kỷ lục" in p and "HPG giảm sàn" in p
    assert "pos" in p and "neg" in p and "neu" in p  # có hướng dẫn nhãn
