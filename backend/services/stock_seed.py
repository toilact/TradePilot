"""Seed bảng `stocks` — lấy metadata mã từ vnstock (không hardcode).

Phase 1.1 chỉ seed 1 mã (VCB) trước khi fetch giá. Top 100 để Phase 2.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from db.upsert import upsert
from models.database import SessionLocal, Stock

logger = logging.getLogger(__name__)


def _fetch_metadata(symbol: str, source: str = "vci") -> dict:
    """Lấy tên/sàn/ngành từ vnstock. Có fallback tối thiểu nếu thiếu.

    Bọc trong hàm sync để gọi qua asyncio.to_thread (vnstock là blocking I/O).
    """
    name, exchange, sector = symbol, "HOSE", None
    try:
        from vnstock import Listing

        df = Listing(source=source).symbols_by_exchange()
        # df có cột symbol + exchange (tên cột có thể khác theo version) → tra cứu mềm dẻo.
        cols = {c.lower(): c for c in df.columns}
        sym_col = cols.get("symbol") or cols.get("ticker")
        exch_col = cols.get("exchange") or cols.get("comgroupcode") or cols.get("board")
        name_col = (
            cols.get("organ_name") or cols.get("organ_short_name") or cols.get("company_name")
        )
        if sym_col:
            row = df[df[sym_col].astype(str).str.upper() == symbol.upper()]
            if not row.empty:
                if exch_col:
                    exchange = str(row.iloc[0][exch_col]) or exchange
                if name_col:
                    name = str(row.iloc[0][name_col]) or name
    except Exception as exc:  # noqa: BLE001 — metadata là phụ, không chặn pipeline
        logger.warning("Không lấy được metadata vnstock cho %s: %s", symbol, exc)

    return {"symbol": symbol.upper(), "name": name, "exchange": exchange, "sector": sector}


async def seed_stock(symbol: str, source: str = "vci") -> int:
    """Upsert mã vào `stocks` (theo symbol unique). Trả về stock_id."""
    meta = await asyncio.to_thread(_fetch_metadata, symbol, source)
    async with SessionLocal() as session:
        await upsert(
            session,
            Stock,
            [meta],
            index_elements=["symbol"],
            update_cols=["name", "exchange", "sector"],
        )
        result = await session.execute(select(Stock.id).where(Stock.symbol == meta["symbol"]))
        stock_id = result.scalar_one()
    logger.info("Seed stock %s → id=%s (%s)", meta["symbol"], stock_id, meta["exchange"])
    return stock_id
