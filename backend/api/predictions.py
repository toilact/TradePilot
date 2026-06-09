"""Endpoint dự đoán. TODO Phase 1.4: đọc prediction mới nhất từ DB (qua cache Redis)."""

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/predictions")
async def get_prediction(symbol: str = Query(..., description="Mã CK, vd VCB")):
    # TODO: query bảng predictions theo symbol, ưu tiên đọc từ Redis cache.
    return {"symbol": symbol, "label": None, "confidence": None, "detail": "not_implemented"}
