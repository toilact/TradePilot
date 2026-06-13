"""Test decode published_at từ ID bài CafeF (hàm thuần, không network)."""

from datetime import datetime

from scripts.ingest_titles import parse_cafef_id_datetime


def test_decode_valid_cafef_id():
    url = "https://cafef.vn/ssi-dong-cua-chi-nhanh-188260612153557943.chn"
    assert parse_cafef_id_datetime(url) == datetime(2026, 6, 12, 15, 35, 57)


def test_decode_another_prefix():
    url = "https://cafef.vn/vinfast-co-the-ghi-nhan-1-khoan-lai-khong-lo-18826061308334668.chn"
    assert parse_cafef_id_datetime(url) == datetime(2026, 6, 13, 8, 33, 46)


def test_decode_no_id_returns_none():
    assert parse_cafef_id_datetime("https://cafef.vn/thi-truong-chung-khoan.chn") is None


def test_decode_non_chn_url_returns_none():
    assert parse_cafef_id_datetime("https://example.com/article-123.html") is None
