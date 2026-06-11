"""Seed bảng `stocks` — lấy metadata mã từ vnstock (không hardcode).

Phase 1.1 chỉ seed 1 mã (VCB) trước khi fetch giá. Top 100 để Phase 2.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from sqlalchemy import select

from db.upsert import upsert
from models.database import SessionLocal, Stock

logger = logging.getLogger(__name__)

# VN30 — nguồn duy nhất cho mọi script (export/seed/crawl import từ đây, không copy).
VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]  # fmt: skip

# Sector lấy ở ICB level 2 ("Ngân hàng", "Bất động sản", "Thực phẩm và đồ uống"...):
# level 1 quá rộng (VIC/VHM/VRE đều "Tài chính"), level 4 quá hẹp (M2, grill 2026-06-11).
_ICB_SECTOR_LEVEL = 2


@lru_cache(maxsize=2)
def _exchange_df(source: str):
    """Listing theo sàn — cache module-level: seed 30 mã chỉ gọi API 1 lần."""
    from vnstock import Listing

    return Listing(source=source).symbols_by_exchange()


@lru_cache(maxsize=2)
def _industries_df(source: str):
    """Listing theo ngành ICB (mỗi mã 4 hàng level 1-4) — cache như trên."""
    from vnstock import Listing

    return Listing(source=source).symbols_by_industries()


def _cols(df) -> dict[str, str]:
    """Map tên cột lowercase → tên thật (vnstock đổi tên cột theo version → tra cứu mềm dẻo)."""
    return {c.lower(): c for c in df.columns}


def _fetch_metadata(symbol: str, source: str = "vci") -> dict:
    """Lấy tên/sàn/ngành từ vnstock. Có fallback tối thiểu nếu thiếu.

    Bọc trong hàm sync để gọi qua asyncio.to_thread (vnstock là blocking I/O).
    """
    name, exchange, sector = symbol, "HOSE", None
    try:
        df = _exchange_df(source)
        cols = _cols(df)
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

    try:
        ind = _industries_df(source)
        cols = _cols(ind)
        sym_col = cols.get("symbol") or cols.get("ticker")
        icb_name_col = cols.get("icb_name") or cols.get("industry") or cols.get("sector")
        icb_level_col = cols.get("icb_level")
        if sym_col and icb_name_col:
            rows = ind[ind[sym_col].astype(str).str.upper() == symbol.upper()]
            if icb_level_col is not None and not rows.empty:
                lvl = rows[rows[icb_level_col] == _ICB_SECTOR_LEVEL]
                if lvl.empty:
                    # KHÔNG fallback im lặng sang level khác — sector sai granularity còn tệ hơn
                    # thiếu (model học nhầm). Log để người seed biết vnstock đổi format.
                    logger.warning(
                        "vnstock không có ICB level %s cho %s — bỏ qua sector (có level: %s)",
                        _ICB_SECTOR_LEVEL,
                        symbol,
                        sorted(rows[icb_level_col].unique().tolist()),
                    )
                    rows = rows.iloc[0:0]
                else:
                    rows = lvl
            if not rows.empty:
                sector = str(rows.iloc[0][icb_name_col]).strip() or None
    except Exception as exc:  # noqa: BLE001 — sector là phụ, không chặn pipeline
        logger.warning("Không lấy được sector vnstock cho %s: %s", symbol, exc)

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
