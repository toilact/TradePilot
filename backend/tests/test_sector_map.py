"""Test sector mapping (M8 Pha 1a): tin vĩ mô/ngành → toàn mã cùng ngành + union match_stocks."""

from services.crawler import match_stocks
from services.sector_map import SECTOR_KEYWORDS, match_sector_stock_ids

# Fixture: ngành → stock_id (mô phỏng stocks.sector). Banking nhiều mã (đòn bẩy chính).
_SECTOR_IDS = {
    "Ngân hàng": [1, 2, 3],  # VCB, BID, CTG
    "Tài nguyên Cơ bản": [10],  # HPG
    "Dầu khí": [20],  # PLX
}


def test_macro_bank_news_maps_all_banks():
    # Tin vĩ mô ngân hàng KHÔNG nêu mã → vẫn map cả 3 mã ngân hàng (coverage).
    ids = match_sector_stock_ids("Ngân hàng Nhà nước hạ lãi suất huy động", _SECTOR_IDS)
    assert set(ids) == {1, 2, 3}


def test_steel_keyword_maps_sector():
    assert match_sector_stock_ids("Giá thép xây dựng tăng mạnh quý 2", _SECTOR_IDS) == [10]


def test_no_keyword_returns_empty():
    assert match_sector_stock_ids("Thị trường khởi sắc phiên cuối tuần", _SECTOR_IDS) == []


def test_generic_chung_khoan_does_not_false_map():
    # "chứng khoán" trần (gần như mọi tin thị trường có) KHÔNG được map (tránh false-positive).
    assert match_sector_stock_ids("Thị trường chứng khoán hồi phục", _SECTOR_IDS) == []


def test_match_stocks_unions_symbol_and_sector():
    # match_stocks: symbol PLX (token) + keyword 'giá dầu' → union PLX-id (không nhân đôi).
    lookup = {"PLX": 20, "VCB": 1}
    ids = match_stocks("PLX hưởng lợi khi giá dầu thô tăng", lookup, _SECTOR_IDS)
    assert ids == [20]  # PLX 1 lần (symbol + sector trùng → dedupe)


def test_match_stocks_without_sector_lookup_unchanged():
    # sector_lookup=None → hành vi cũ (chỉ symbol/tên).
    lookup = {"VCB": 1}
    assert match_stocks("VCB báo lãi, ngành ngân hàng khởi sắc", lookup) == [1]


def test_sector_keys_are_unique():
    assert len(SECTOR_KEYWORDS) == len(set(SECTOR_KEYWORDS))
