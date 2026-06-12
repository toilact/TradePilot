"""Endpoint dữ liệu mã CK — giá + sentiment timeline, accuracy + Redis cache TTL 1h (M6)."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cache import KEY_PREFIX, TTL_ACCURACY, TTL_HISTORY, cached_response
from models.database import get_session
from services import read_api

router = APIRouter(prefix="/api", tags=["stocks"])


@router.get("/stocks/{symbol}/history")
async def get_history(symbol: str, session: AsyncSession = Depends(get_session)):
    sym = symbol.upper()

    async def produce():
        return await read_api.get_history(session, sym)

    return await cached_response(f"{KEY_PREFIX}history:{sym}", TTL_HISTORY, produce)


@router.get("/accuracy")
async def get_accuracy(session: AsyncSession = Depends(get_session)):
    async def produce():
        return await read_api.get_accuracy(session)

    return await cached_response(f"{KEY_PREFIX}accuracy", TTL_ACCURACY, produce)
