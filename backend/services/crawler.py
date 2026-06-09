"""Crawl tin tức CafeF + FireAnt → news + news_stocks.

Bất biến:
  - crawl LỊCH SỰ: tôn trọng robots.txt, rate-limit + delay, User-Agent rõ ràng
  - chỉ lưu link + metadata (+ sentiment sau), KHÔNG tái xuất bản nguyên văn
  - 1 bài → nhiều mã qua news_stocks

TODO Phase 1.1: parser CafeF + FireAnt (có HTML fixture để test).
Chạy thủ công: python -m services.crawler --symbol VCB
"""

from __future__ import annotations

import argparse
import asyncio

REQUEST_DELAY_SECONDS = 1.0  # tránh bị block IP
USER_AGENT = "TradePilotBot/0.1 (personal research project)"


async def crawl_news(symbol: str) -> int:
    """Crawl tin mới cho `symbol`. Trả về số bài đã lưu. (chưa implement)"""
    raise NotImplementedError("Phase 1.1: parser CafeF/FireAnt + map news_stocks")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Crawl tin tức 1 mã CK")
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    asyncio.run(crawl_news(args.symbol))


if __name__ == "__main__":
    _main()
