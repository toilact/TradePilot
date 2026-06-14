"""Ánh xạ LẠI tin hiện có sang toàn mã cùng ngành (M8 Pha 1a) — KHÔNG cào gì thêm.

Tăng coverage tin trên data ĐANG CÓ: chạy match_sector_stock_ids trên mọi news.title → thêm
cặp (news_id, stock_id) cho tin vĩ mô/ngành (idempotent, chỉ THÊM link, không xoá). Rồi rebuild
daily_sentiment mọi mã (tin đã có sentiment_score sẵn → không re-score). Đo coverage trước/sau.

Mục đích: làm diagnostic Pha 1 giàu coverage TRƯỚC khi quyết định có đáng cào (Pha 2) không.

Chạy: cd backend && uv run python -m scripts.remap_sectors
      uv run python -m scripts.remap_sectors --dry-run   # chỉ đếm, không ghi
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


async def remap(dry_run: bool = False) -> tuple[int, int, int]:
    """Trả (số tin có khớp ngành, số cặp news_stock MỚI, số cặp đã tồn tại)."""
    from sqlalchemy import select

    from db.upsert import upsert
    from models.database import News, NewsStock, SessionLocal
    from services.sector_map import load_sector_lookup, match_sector_stock_ids

    sector_lookup = await load_sector_lookup()
    async with SessionLocal() as session:
        news = (await session.execute(select(News.id, News.title))).all()
        existing = {
            (nid, sid)
            for nid, sid in (
                await session.execute(select(NewsStock.news_id, NewsStock.stock_id))
            ).all()
        }

        matched_news = 0
        new_pairs: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for news_id, title in news:
            sids = match_sector_stock_ids(title or "", sector_lookup)
            if sids:
                matched_news += 1
            for sid in sids:
                pair = (news_id, sid)
                if pair in existing or pair in seen:
                    continue
                seen.add(pair)
                new_pairs.append({"news_id": news_id, "stock_id": sid})

        n_existing = sum(
            1
            for nid, title in news
            for sid in match_sector_stock_ids(title or "", sector_lookup)
            if (nid, sid) in existing
        )
        if not dry_run and new_pairs:
            await upsert(
                session,
                NewsStock,
                new_pairs,
                index_elements=["news_id", "stock_id"],
                update_cols=["stock_id"],
            )
    return matched_news, len(new_pairs), n_existing


async def rebuild_daily() -> int:
    """Rebuild daily_sentiment mọi mã active (tin đã có điểm sẵn → không re-score). Trả số mã."""
    from sqlalchemy import select

    from models.database import SessionLocal, Stock
    from services.sentiment import build_daily_sentiment

    async with SessionLocal() as session:
        symbols = (
            (await session.execute(select(Stock.symbol).where(Stock.is_active.is_(True))))
            .scalars()
            .all()
        )
    for sym in symbols:
        await build_daily_sentiment(sym)
    return len(symbols)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Remap tin hiện có sang mã cùng ngành (M8 Pha 1a)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ đếm, không ghi DB")
    args = parser.parse_args()

    async def _run() -> None:
        matched, new_pairs, existed = await remap(dry_run=args.dry_run)
        print(f"tin khớp ngành: {matched} | cặp news_stock mới: {new_pairs} | đã có: {existed}")
        if args.dry_run:
            print("(dry-run — không ghi, không rebuild)")
            return
        n_sym = await rebuild_daily()
        print(f"✓ rebuild daily_sentiment {n_sym} mã")

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
