"""Crawl tin tức CafeF → news + news_stocks (FireAnt: adapter chờ token API).

Bất biến:
  - crawl LỊCH SỰ: tôn trọng robots.txt, rate-limit + delay, User-Agent rõ ràng,
    skip URL đã có (news.url UNIQUE) để tránh request lặp.
  - chỉ lưu link + metadata (title, published_at); content=None, sentiment_score=None.
    KHÔNG tái xuất bản nguyên văn bài báo ở bước crawl.
  - 1 bài → nhiều mã qua news_stocks (N-N). Phase 1.1 map về mã đang crawl.
  - published_at parse KÈM GIỜ (Asia/Ho_Chi_Minh) — nền tảng chống data leakage:
    feature ngày T chỉ dùng tin published_at ≤ 16:00 phiên T.

Chạy thủ công: python -m services.crawler --symbol VCB

FireAnt: trang công khai render client-side (Next.js), tin nằm sau API cần token →
chưa crawl được lịch sự trong Phase 1.1. Để adapter rõ ràng, không bịa selector.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import urllib.robotparser
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select

from db.upsert import upsert
from models.database import News, NewsStock, SessionLocal
from services.stock_seed import seed_stock

logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 1.0  # tránh bị block IP
USER_AGENT = "TradePilotBot/0.1 (personal research project)"

CAFEF_BASE = "https://cafef.vn"
SOURCE_CAFEF = "cafef"


def parse_cafef(html: str, base_url: str = CAFEF_BASE) -> list[dict]:
    """HTML trang tin CafeF theo mã → list dict {title, url, published_at, source}.

    Hàm THUẦN (không network) để test bằng fixture. Bỏ qua item thiếu trường bắt buộc.
    Cấu trúc thật: a.box-category-link-title (title + href .chn) +
    p.time_cate > span.time ("dd/mm/yyyy HH:MM").
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    for link in soup.select("a.box-category-link-title"):
        href = link.get("href")
        title = link.get("title") or link.get_text(strip=True)
        if not href or not title:
            continue
        # span.time đứng ngay sau trong cùng item (h3 → p.time_cate kế tiếp)
        item = link.find_parent(["div", "li"]) or link.parent
        time_el = item.select_one("p.time_cate span.time") if item else None
        published_at = _parse_cafef_time(time_el.get_text(strip=True)) if time_el else None
        if published_at is None:
            continue
        out.append(
            {
                "title": title.strip(),
                "url": urljoin(base_url, href),
                "published_at": published_at,
                "source": SOURCE_CAFEF,
            }
        )
    return out


def _parse_cafef_time(raw: str) -> datetime | None:
    """'19/05/2026 17:36' → datetime NAIVE theo giờ VN. None nếu sai định dạng.

    News.published_at là DateTime không-tz; mọi tin chuẩn hoá về cùng quy ước giờ VN
    để so sánh '≤ 16:00 phiên T' (chống leakage) nhất quán. CafeF vốn đã là giờ VN.
    """
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    logger.warning("CafeF: không parse được thời gian %r", raw)
    return None


def _robots_ok(url: str) -> bool:
    """robots.txt của host có cho USER_AGENT lấy `url` không. Lỗi mạng → cho phép (fail-open)."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as exc:  # noqa: BLE001 — không chặn pipeline vì lỗi đọc robots
        logger.warning("Không đọc được robots.txt %s: %s", robots_url, exc)
        return True
    return rp.can_fetch(USER_AGENT, url)


def _cafef_news_url(symbol: str) -> str:
    return f"{CAFEF_BASE}/{symbol.lower()}.html"


async def _fetch(url: str) -> str | None:
    """GET 1 trang với UA rõ ràng + delay lịch sự. None nếu robots cấm hoặc lỗi."""
    if not _robots_ok(url):
        logger.warning("robots.txt cấm crawl %s — bỏ qua", url)
        return None
    await asyncio.sleep(REQUEST_DELAY_SECONDS)  # rate-limit lịch sự
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=20.0, follow_redirects=True
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def _persist(rows: list[dict], stock_id: int) -> int:
    """Upsert news (idempotent theo url) + map news_stocks. Skip url đã có.

    Trả về số bài MỚI ghi. content/sentiment_score để None — không tái xuất bản nguyên văn.
    """
    if not rows:
        return 0
    urls = [r["url"] for r in rows]
    async with SessionLocal() as session:
        existing = set(
            (await session.execute(select(News.url).where(News.url.in_(urls)))).scalars().all()
        )
        new_rows = [r for r in rows if r["url"] not in existing]

        # Upsert toàn bộ (idempotent) để cập nhật title/published_at nếu đổi; content giữ None.
        await upsert(
            session,
            News,
            [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "source": r["source"],
                    "published_at": r["published_at"],
                    "content": None,
                    "sentiment_score": None,
                }
                for r in rows
            ],
            index_elements=["url"],
            update_cols=["title", "published_at", "source"],
        )

        id_by_url = dict(
            (await session.execute(select(News.url, News.id).where(News.url.in_(urls)))).all()
        )
        await upsert(
            session,
            NewsStock,
            [{"news_id": id_by_url[u], "stock_id": stock_id} for u in urls],
            index_elements=["news_id", "stock_id"],
            update_cols=["stock_id"],
        )
    return len(new_rows)


async def crawl_news(symbol: str) -> int:
    """Crawl tin CafeF cho `symbol` → lưu news + news_stocks. Trả số bài MỚI."""
    symbol = symbol.upper()
    stock_id = await seed_stock(symbol)

    html = await _fetch(_cafef_news_url(symbol))
    if html is None:
        logger.warning("crawler %s: không lấy được trang CafeF", symbol)
        return 0

    rows = parse_cafef(html)
    new_count = await _persist(rows, stock_id)
    logger.info("crawler %s: %d bài (mới %d) từ CafeF", symbol, len(rows), new_count)
    return new_count


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Crawl tin tức 1 mã CK (CafeF)")
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    n = asyncio.run(crawl_news(args.symbol))
    print(f"✓ {args.symbol}: ghi {n} bài mới")


if __name__ == "__main__":
    _main()
