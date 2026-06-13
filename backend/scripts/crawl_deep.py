"""Cào SÂU tin CafeF theo mã (phân trang `trang-N.html`) → news + news_stocks.

Mở rộng phần **D** của M8: `crawl_news` (crawler.py) chỉ lấy trang 1 (~20 bài mới nhất/mã).
Trang mã CafeF phân trang `cafef.vn/<symbol>/trang-N.html` với CÙNG cấu trúc HTML → tái dùng
`parse_cafef`. Script này lặp trang 1..N mọi mã đã seed để lấy lịch sử sâu hơn — tăng data train
sentiment (M8) + % phiên có `news_count>0`.

Lịch sự (kế thừa crawler): `_fetch` đã lo robots.txt + User-Agent rõ + delay 1s/req; `_persist`
idempotent theo url (bỏ url đã có) → chạy lại an toàn. Chỉ lưu title+url+published_at, content=None
(governance: không tái xuất bản). Per-symbol nên dùng đúng stock_id (chính xác hơn match title).

Dừng sớm mỗi mã khi 1 trang không còn bài (hết lịch sử) để khỏi request thừa.

Chạy: cd backend && uv run python -m scripts.crawl_deep --pages 8
      uv run python -m scripts.crawl_deep --pages 5 --symbols VCB,FPT
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _page_url(symbol: str, page: int) -> str:
    """Trang 1 = `<symbol>.html`; trang ≥2 = `<symbol>/trang-N.html` (cấu trúc CafeF)."""
    sym = symbol.lower()
    if page <= 1:
        return f"https://cafef.vn/{sym}.html"
    return f"https://cafef.vn/{sym}/trang-{page}.html"


async def crawl_symbol_deep(symbol: str, max_pages: int) -> tuple[int, int]:
    """Cào tới `max_pages` trang của 1 mã → (số bài parse được, số bài MỚI ghi DB)."""
    from services.crawler import _fetch, _persist, parse_cafef
    from services.stock_seed import seed_stock

    symbol = symbol.upper()
    stock_id = await seed_stock(symbol)
    total_seen = total_new = 0
    for page in range(1, max_pages + 1):
        try:
            html = await _fetch(_page_url(symbol, page))
        except Exception as exc:  # noqa: BLE001 — 1 trang timeout/lỗi mạng KHÔNG hạ cả run
            print(f"    {symbol} trang {page}: lỗi mạng ({type(exc).__name__}) → bỏ qua")
            continue
        if html is None:
            break
        rows = parse_cafef(html)
        if not rows:  # hết lịch sử (hoặc trang vượt số trang có thật)
            break
        total_seen += len(rows)
        total_new += await _persist(rows, stock_id)
    return total_seen, total_new


async def main() -> None:
    from sqlalchemy import select

    from models.database import SessionLocal, Stock

    parser = argparse.ArgumentParser(description="Cào sâu tin CafeF theo mã (phân trang)")
    parser.add_argument("--pages", type=int, default=8, help="số trang tối đa mỗi mã (default 8)")
    parser.add_argument(
        "--symbols", default=None, help="CSV mã cụ thể (vd VCB,FPT); mặc định mọi mã đã seed"
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        async with SessionLocal() as session:
            symbols = [
                s for (s,) in (await session.execute(select(Stock.symbol).order_by(Stock.symbol)))
            ]

    print(f"Cào sâu {len(symbols)} mã, tối đa {args.pages} trang/mã...")
    grand_seen = grand_new = 0
    for sym in symbols:
        seen, new = await crawl_symbol_deep(sym, args.pages)
        grand_seen += seen
        grand_new += new
        print(f"  {sym:5} parse {seen:4d} bài → {new:4d} MỚI")
    print(f"\n✓ Tổng: parse {grand_seen} bài, ghi {grand_new} MỚI vào news.")
    print("Tiếp: export title chưa nhãn → label (rubric góc-giá-T+1) → sentiment_labeled.csv")


if __name__ == "__main__":
    asyncio.run(main())
