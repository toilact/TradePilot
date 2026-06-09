"""Fetch OHLCV qua vnstock → lưu price_history.

TODO Phase 1.1 (mắt xích đầu tiên của walking skeleton):
  - dùng vnstock lấy lịch sử giá 5 năm cho 1 mã (bắt đầu VCB)
  - upsert vào bảng price_history theo (stock_id, date)
"""

from __future__ import annotations


async def fetch_price_history(symbol: str, years: int = 5) -> int:
    """Lấy & lưu OHLCV cho `symbol`. Trả về số dòng đã ghi. (chưa implement)"""
    raise NotImplementedError("Phase 1.1: tích hợp vnstock + upsert price_history")
