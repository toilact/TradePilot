"""Auth + watchlist. TODO Phase 3: xác thực JWT từ NextAuth (Email + Google)."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["auth"])


@router.get("/watchlist")
async def get_watchlist():
    # TODO: yêu cầu JWT hợp lệ; trả danh sách mã user đang theo dõi.
    return {"watchlist": [], "detail": "not_implemented"}
