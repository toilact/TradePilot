"""Endpoint dự đoán — đọc DB thật (Phase 1.4) + Redis cache TTL 1h (M6)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from cache import KEY_PREFIX, TTL_PREDICTIONS, cached_json
from models.database import get_session
from services import read_api

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/predictions")
async def get_predictions(
    symbol: str | None = Query(None, description="Mã CK, vd VCB. Bỏ trống → trả tất cả."),
    session: AsyncSession = Depends(get_session),
):
    if symbol is None:

        async def produce_all():
            return await read_api.list_predictions(session)

        data, status = await cached_json(
            f"{KEY_PREFIX}predictions:all", TTL_PREDICTIONS, produce_all
        )
        return JSONResponse(content=data, headers={"X-Cache": status})

    sym = symbol.upper()

    async def produce_one():
        # 404 raise TRONG producer → mã sai không bị cache
        pred = await read_api.get_prediction(session, sym)
        if pred is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy mã {symbol}")
        return pred

    data, status = await cached_json(f"{KEY_PREFIX}predictions:{sym}", TTL_PREDICTIONS, produce_one)
    return JSONResponse(content=data, headers={"X-Cache": status})
