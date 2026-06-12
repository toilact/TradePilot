"""Endpoint tin tức theo mã — M6 (trang chi tiết hết mock 100%)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from cache import KEY_PREFIX, TTL_NEWS, cached_json
from models.database import get_session
from services import read_api

router = APIRouter(prefix="/api", tags=["news"])


@router.get("/news")
async def get_news(
    symbol: str = Query(..., description="Mã CK, vd VCB (bắt buộc)."),
    limit: int = Query(10, ge=1, le=50, description="Số bài tối đa."),
    session: AsyncSession = Depends(get_session),
):
    sym = symbol.upper()

    async def produce():
        # 404 raise TRONG producer → không bị cache (mã sai không chiếm key 15ph)
        news = await read_api.get_news(session, sym, limit)
        if news is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy mã {symbol}")
        return news

    data, status = await cached_json(f"{KEY_PREFIX}news:{sym}:{limit}", TTL_NEWS, produce)
    return JSONResponse(content=data, headers={"X-Cache": status})
