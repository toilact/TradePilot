"""Nạp tin (title + url) đã cào sẵn (vd firecrawl) → news + news_stocks (governance).

Đầu vào: JSON list [{"title": ..., "url": "https://cafef.vn/...chn"}] — cào từ trang chuyên mục
CafeF. Script này KHÔNG tự cào (firecrawl là tool của agent) — nhận JSON rồi nạp lịch sự:
  - published_at decode TỪ ID bài CafeF (ID mã hoá yyMMddHHmmss) — chính xác tới giờ, đủ cho
    bất biến chống leakage (tin ≤ 16:00 phiên T). Bỏ tin không decode được ngày hợp lệ.
  - match title → mã đã seed (match_stocks, N-N); bỏ tin không khớp mã nào (không tạo news mồ côi).
  - persist title + url + published_at; content=None, sentiment_score=None (không tái xuất bản).
  - idempotent theo url (bỏ qua tin đã có).

Chạy: cd backend && uv run python -m scripts.ingest_titles ../ml/data/firecrawl_titles.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def parse_cafef_id_datetime(url: str) -> datetime | None:
    """Decode ngày-đăng từ ID bài CafeF (yyMMddHHmmss nhúng trong số cuối url). None nếu không có.

    Vd '.../ssi-dong-cua-chi-nhanh-188260612153557943.chn' → 2026-06-12 15:35:57
    (prefix '188' + '260612153557' + seq). Quét cửa sổ 12 chữ số đầu tiên hợp lệ.
    """
    m = re.search(r"(\d{15,25})\.chn", url)  # ID CafeF ~17-18 chữ số; chặn trên tránh quét lạ
    if not m:
        return None
    digits = m.group(1)
    for start in range(len(digits) - 11):
        chunk = digits[start : start + 12]
        try:
            dt = datetime.strptime(chunk, "%y%m%d%H%M%S")
        except ValueError:
            continue
        if 2020 <= dt.year <= 2035:  # cửa sổ năm hợp lý → tin cậy
            return dt
    return None


async def main() -> None:
    from services.crawler import _persist_multi, _stock_lookup, match_stocks

    parser = argparse.ArgumentParser(description="Nạp title+url đã cào (JSON) → news")
    parser.add_argument("json_path", help="file JSON [{title,url}] từ firecrawl")
    args = parser.parse_args()

    items = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    if isinstance(items, dict):  # chấp nhận {"articles": [...]}
        items = items.get("articles", [])

    lookup = await _stock_lookup()
    if not lookup:
        sys.exit("Chưa seed mã nào — không match được. Seed stocks trước.")

    rows: list[dict] = []
    skipped_date = skipped_match = 0
    seen: set[str] = set()
    for it in items:
        title, url = (it.get("title") or "").strip(), (it.get("url") or "").strip()
        if not title or not url or url in seen or "cafef.vn" not in url:
            continue
        seen.add(url)
        published_at = parse_cafef_id_datetime(url)
        if published_at is None:
            skipped_date += 1
            continue
        stock_ids = match_stocks(title, lookup)
        if not stock_ids:
            skipped_match += 1
            continue
        rows.append(
            {
                "title": title,
                "url": url,
                "published_at": published_at,
                "source": "cafef",
                "stock_ids": stock_ids,
            }
        )

    new_count = await _persist_multi(rows)
    print(
        f"Nạp {len(items)} tin: {len(rows)} khớp mã & có ngày, ghi {new_count} MỚI "
        f"(bỏ {skipped_date} không decode được ngày, {skipped_match} không khớp mã)."
    )


if __name__ == "__main__":
    asyncio.run(main())
