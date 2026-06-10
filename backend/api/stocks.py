"""Endpoint dữ liệu mã CK — giá + sentiment timeline, thống kê accuracy (Phase 1.4)."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_session
from services import read_api

router = APIRouter(prefix="/api", tags=["stocks"])


@router.get("/stocks/{symbol}/history")
async def get_history(symbol: str, session: AsyncSession = Depends(get_session)):
    return await read_api.get_history(session, symbol)


@router.get("/accuracy")
async def get_accuracy(session: AsyncSession = Depends(get_session)):
    return await read_api.get_accuracy(session)
