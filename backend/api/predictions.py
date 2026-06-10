"""Endpoint dự đoán — đọc DB thật (Phase 1.4). Chưa Redis cache (tối ưu sau)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_session
from services import read_api

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/predictions")
async def get_predictions(
    symbol: str | None = Query(None, description="Mã CK, vd VCB. Bỏ trống → trả tất cả."),
    session: AsyncSession = Depends(get_session),
):
    if symbol is None:
        return await read_api.list_predictions(session)
    pred = await read_api.get_prediction(session, symbol)
    if pred is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy mã {symbol}")
    return pred
