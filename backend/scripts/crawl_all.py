"""Gom crawl tin cho 30 mã VN30 + feed RSS — để có thêm title cho fine-tune sentiment.

Tái dùng crawler sẵn có (lịch sự: robots.txt + delay + UA, idempotent theo url).
Chạy: cd backend && uv run python -m scripts.crawl_all

Có thể chạy nhiều lần (idempotent — chỉ thêm bài mới). Mất vài phút vì có delay lịch sự.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


from services.stock_seed import VN30  # noqa: E402 — sau sys.path setup


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from services.crawler import crawl_news, crawl_rss

    total_new = 0
    for symbol in VN30:
        try:
            total_new += await crawl_news(symbol)
        except Exception as exc:  # 1 mã lỗi không chặn cả batch
            logging.warning("crawl_news %s lỗi: %s", symbol, exc)

    try:
        total_new += await crawl_rss()
    except Exception as exc:
        logging.warning("crawl_rss lỗi: %s", exc)

    print(f"\nDone: {total_new} bài mới (30 mã VN30 + RSS).")


if __name__ == "__main__":
    asyncio.run(main())
