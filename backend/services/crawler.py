"""Crawl tin tức CafeF → news + news_stocks. Hai nguồn:

  1. Trang theo mã (cafef.vn/<symbol>.html): tin đã gắn sẵn mã, ~20 bài gần nhất.
  2. Feed RSS chuyên mục: tin toàn thị trường (50/feed) → map về mã qua match_stocks (N-N).

Bất biến:
  - crawl LỊCH SỰ: tôn trọng robots.txt, rate-limit + delay, User-Agent rõ ràng,
    skip URL đã có (news.url UNIQUE) để tránh request lặp.
  - chỉ lưu link + metadata (title, published_at); content=None, sentiment_score=None.
    KHÔNG tái xuất bản nguyên văn bài báo ở bước crawl.
  - 1 bài → nhiều mã qua news_stocks (N-N).
  - published_at parse KÈM GIỜ (chuẩn hoá giờ VN, naive) — nền tảng chống data leakage:
    feature ngày T chỉ dùng tin published_at ≤ 16:00 phiên T.

Chạy thủ công:
  python -m services.crawler --symbol VCB   # trang theo mã
  python -m services.crawler --rss          # feed RSS, map về các mã đã seed

FireAnt: trang công khai render client-side (Next.js), tin nằm sau API cần token →
chưa crawl được lịch sự. Để adapter rõ ràng, không bịa selector.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import urllib.robotparser
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx
import structlog
from bs4 import BeautifulSoup
from sqlalchemy import select

from db.upsert import upsert
from models.database import News, NewsStock, SessionLocal, Stock
from services.stock_seed import seed_stock

logger = structlog.get_logger(__name__)

REQUEST_DELAY_SECONDS = 1.0  # tránh bị block IP
USER_AGENT = "TradePilotBot/0.1 (personal research project)"

CAFEF_BASE = "https://cafef.vn"
SOURCE_CAFEF = "cafef"
VN_OFFSET = timezone(timedelta(hours=7))  # giờ VN; published_at lưu naive theo quy ước này

# Feed RSS chuyên mục chứng khoán CafeF (tin toàn thị trường → map về mã qua match_stocks).
CAFEF_RSS_FEEDS = (
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://cafef.vn/doanh-nghiep.rss",
)


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
    logger.warning("cafef_time_parse_failed", raw=raw)
    return None


def parse_cafef_rss(xml: str) -> list[dict]:
    """XML feed RSS CafeF → list dict {title, url, published_at, source}.

    Hàm THUẦN (không network) để test bằng fixture. Bỏ qua item thiếu trường bắt buộc.
    Chỉ lấy title/link/pubDate — KHÔNG đọc <description> (giữ bất biến không tái xuất bản).
    """
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        logger.warning("rss_xml_parse_failed", error=str(exc))
        return []
    out: list[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        published_at = _parse_rss_time(item.findtext("pubDate"))
        if not title or not url or published_at is None:
            continue
        out.append(
            {"title": title, "url": url, "published_at": published_at, "source": SOURCE_CAFEF}
        )
    return out


def _parse_rss_time(raw: str | None) -> datetime | None:
    """pubDate RFC-822 ('Tue, 09 Jun 26 17:16:00 +0700') → datetime NAIVE giờ VN.

    Chuẩn hoá mọi tin về cùng quy ước giờ VN (như _parse_cafef_time) để so sánh
    '≤ 16:00 phiên T' (chống leakage) nhất quán. Có offset → đổi sang VN rồi bỏ tzinfo.
    """
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError) as exc:
        logger.warning("rss_pubdate_parse_failed", raw=raw, error=str(exc))
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(VN_OFFSET).replace(tzinfo=None)
    return dt


def match_stocks(title: str, lookup: dict[str, int]) -> list[int]:
    """Tìm các stock_id có symbol/tên xuất hiện trong `title`. 1 bài → nhiều mã (N-N).

    `lookup`: map key (symbol viết hoa HOẶC tên công ty lowercase) → stock_id.
    - symbol khớp dạng token nguyên (\\bVCB\\b) để tránh dính chuỗi con.
    - tên công ty khớp substring case-insensitive.
    Trả list stock_id duy nhất (giữ thứ tự xuất hiện ổn định).
    """
    title_l = title.lower()
    found: dict[int, None] = {}  # dùng dict giữ thứ tự + loại trùng
    for key, sid in lookup.items():
        if key.isupper():  # symbol: khớp token nguyên
            if re.search(rf"\b{re.escape(key)}\b", title):
                found[sid] = None
        elif key and key in title_l:  # tên công ty: substring lowercase
            found[sid] = None
    return list(found)


async def _stock_lookup() -> dict[str, int]:
    """Bảng tra {SYMBOL, tên-lowercase} → stock_id từ bảng stocks (mã đã seed)."""
    lookup: dict[str, int] = {}
    async with SessionLocal() as session:
        rows = (await session.execute(select(Stock.id, Stock.symbol, Stock.name))).all()
    for sid, symbol, name in rows:
        lookup[symbol.upper()] = sid
        if name and name.upper() != symbol.upper():
            lookup[name.lower()] = sid
    return lookup


def _robots_ok(url: str) -> bool:
    """robots.txt của host có cho USER_AGENT lấy `url` không. Lỗi mạng → cho phép (fail-open)."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as exc:  # noqa: BLE001 — không chặn pipeline vì lỗi đọc robots
        logger.warning("robots_read_failed", url=robots_url, error=str(exc))
        return True
    return rp.can_fetch(USER_AGENT, url)


def _cafef_news_url(symbol: str) -> str:
    return f"{CAFEF_BASE}/{symbol.lower()}.html"


async def _fetch(url: str) -> str | None:
    """GET 1 trang với UA rõ ràng + delay lịch sự. None nếu robots cấm hoặc lỗi."""
    if not _robots_ok(url):
        logger.warning("robots_disallowed", url=url)
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


async def _persist_multi(rows: list[dict]) -> int:
    """Như _persist nhưng mỗi row mang sẵn `stock_ids: list[int]` (RSS map nhiều mã).

    Upsert news theo url (content=None), rồi map mọi cặp (news_id, stock_id). Idempotent.
    Trả số bài MỚI (url chưa từng có).
    """
    rows = [r for r in rows if r.get("stock_ids")]  # bỏ bài không khớp mã nào
    if not rows:
        return 0
    urls = [r["url"] for r in rows]
    async with SessionLocal() as session:
        existing = set(
            (await session.execute(select(News.url).where(News.url.in_(urls)))).scalars().all()
        )
        new_count = sum(1 for u in urls if u not in existing)

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
        links = [
            {"news_id": id_by_url[r["url"]], "stock_id": sid}
            for r in rows
            for sid in r["stock_ids"]
        ]
        await upsert(
            session,
            NewsStock,
            links,
            index_elements=["news_id", "stock_id"],
            update_cols=["stock_id"],
        )
    return new_count


async def crawl_rss(feeds: tuple[str, ...] = CAFEF_RSS_FEEDS) -> int:
    """Đọc các feed RSS CafeF → map bài về mã (theo stocks đã seed) → lưu. Trả số bài MỚI.

    Bài không khớp mã nào bị bỏ qua (không tạo news mồ côi). Mã phải đã có trong `stocks`.
    """
    lookup = await _stock_lookup()
    if not lookup:
        logger.warning("crawl_rss_no_stocks", hint="seed mã trước (vd crawl_news VCB)")
        return 0

    total_new = 0
    for feed in feeds:
        xml = await _fetch(feed)
        if xml is None:
            continue
        rows = parse_cafef_rss(xml)
        for r in rows:
            r["stock_ids"] = match_stocks(r["title"], lookup)
        matched = [r for r in rows if r["stock_ids"]]
        new_count = await _persist_multi(matched)
        total_new += new_count
        logger.info("rss_crawled", feed=feed, total=len(rows), matched=len(matched), new=new_count)
    return total_new


async def crawl_news(symbol: str) -> int:
    """Crawl tin CafeF cho `symbol` → lưu news + news_stocks. Trả số bài MỚI."""
    symbol = symbol.upper()
    stock_id = await seed_stock(symbol)

    html = await _fetch(_cafef_news_url(symbol))
    if html is None:
        logger.warning("cafef_fetch_failed", symbol=symbol)
        return 0

    rows = parse_cafef(html)
    new_count = await _persist(rows, stock_id)
    logger.info("news_crawled", symbol=symbol, total=len(rows), new=new_count)
    return new_count


def _main() -> None:
    from logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Crawl tin tức CafeF (trang theo mã hoặc RSS)")
    parser.add_argument("--symbol", help="Crawl trang tin theo mã, vd VCB")
    parser.add_argument("--rss", action="store_true", help="Crawl feed RSS chuyên mục → map về mã")
    args = parser.parse_args()
    if not args.symbol and not args.rss:
        parser.error("cần --symbol SYMBOL hoặc --rss")

    if args.symbol:
        n = asyncio.run(crawl_news(args.symbol))
        print(f"✓ {args.symbol}: ghi {n} bài mới")
    if args.rss:
        n = asyncio.run(crawl_rss())
        print(f"✓ RSS: ghi {n} bài mới")


if __name__ == "__main__":
    _main()
