"""Endpoint dữ liệu mã CK. TODO Phase 1.4: giá + sentiment timeline, thống kê accuracy."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["stocks"])


@router.get("/stocks/{symbol}/history")
async def get_history(symbol: str):
    # TODO: join price_history + daily_sentiment theo (stock, date).
    return {"symbol": symbol, "history": [], "detail": "not_implemented"}


@router.get("/accuracy")
async def get_accuracy():
    # TODO: so predictions vs actual_results, theo model_version + 30/90 ngày.
    return {"accuracy": None, "detail": "not_implemented"}
