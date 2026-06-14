"""Auth (JWT từ Auth.js v5) + watchlist CRUD — M9.

Hợp đồng JWT: Auth.js v5 ký session bằng **JWS HS256** (override encode/decode, tắt JWE),
key = bytes thô của AUTH_SECRET. Backend verify cùng AUTH_SECRET → lấy `email` claim →
upsert `users` → gắn watchlist theo user. Frontend gọi qua BFF proxy (cookie httpOnly →
Authorization: Bearer), backend chỉ thấy Bearer token.
"""

from __future__ import annotations

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.upsert import upsert
from models.database import Stock, User, Watchlist, get_session

router = APIRouter(prefix="/api", tags=["auth"])

# vnstock đặt sàn "HSX"; frontend dùng "HOSE" — map về tên frontend (giống read_api).
_EXCHANGE_MAP = {"HSX": "HOSE"}
_UNAUTH = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Không xác thực")


def _bearer_token(authorization: str | None) -> str:
    """Tách `<token>` từ header `Authorization: Bearer <token>`; sai dạng → 401."""
    if not authorization:
        raise _UNAUTH
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _UNAUTH
    return token.strip()


async def get_current_user(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Dependency: verify JWT HS256 (cùng AUTH_SECRET) → upsert user theo email → trả User.

    Mọi lỗi (thiếu/sai token, hết hạn, thiếu email, chưa cấu hình secret) → 401.
    """
    token = _bearer_token(authorization)
    if not settings.auth_secret:
        raise _UNAUTH  # chưa cấu hình auth → coi như không xác thực được
    try:
        payload = jwt.decode(token, settings.auth_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise _UNAUTH from None
    email = payload.get("email")
    if not email:
        raise _UNAUTH
    # Upsert theo email (idempotent): login lại refresh `name`, không tạo trùng user.
    await upsert(
        session,
        User,
        [{"email": email, "name": payload.get("name"), "provider": "google"}],
        index_elements=["email"],
        update_cols=["name"],
    )
    # populate_existing: nếu user đã ở identity-map (cùng session), ép nạp lại giá trị vừa upsert
    # (vd name refresh) thay vì trả bản cache cũ.
    return (
        await session.execute(
            select(User).where(User.email == email).execution_options(populate_existing=True)
        )
    ).scalar_one()


class WatchlistAdd(BaseModel):
    symbol: str


@router.get("/watchlist")
async def get_watchlist(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Danh sách mã user theo dõi (mới thêm trước). Không token → 401. KHÔNG cache (per-user)."""
    rows = (
        await session.execute(
            select(Stock.symbol, Stock.name, Stock.exchange)
            .join(Watchlist, Watchlist.stock_id == Stock.id)
            .where(Watchlist.user_id == user.id)
            .order_by(Watchlist.created_at.desc(), Watchlist.id.desc())
        )
    ).all()
    return [
        {"symbol": r.symbol, "name": r.name, "exchange": _EXCHANGE_MAP.get(r.exchange, r.exchange)}
        for r in rows
    ]


@router.post("/watchlist")
async def add_watchlist(
    body: WatchlistAdd,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Thêm mã vào watchlist. Mã không tồn tại → 404. Idempotent (thêm lại không nhân đôi)."""
    symbol = body.symbol.strip().upper()
    stock_id = (
        await session.execute(select(Stock.id).where(Stock.symbol == symbol))
    ).scalar_one_or_none()
    if stock_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Mã {symbol} không tồn tại"
        )
    # ON CONFLICT(user_id, stock_id) DO UPDATE SET stock_id=excluded — no-op, idempotent
    # (hợp lệ cả PostgreSQL lẫn SQLite; DO NOTHING với set rỗng thì helper không hỗ trợ).
    await upsert(
        session,
        Watchlist,
        [{"user_id": user.id, "stock_id": stock_id}],
        index_elements=["user_id", "stock_id"],
        update_cols=["stock_id"],
    )
    return {"ok": True, "symbol": symbol}


@router.delete("/watchlist")
async def remove_watchlist(
    symbol: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Bỏ mã khỏi watchlist. Idempotent: mã không có (hoặc không trong list) vẫn trả 200."""
    symbol = symbol.strip().upper()
    stock_id = (
        await session.execute(select(Stock.id).where(Stock.symbol == symbol))
    ).scalar_one_or_none()
    if stock_id is not None:
        await session.execute(
            delete(Watchlist).where(Watchlist.user_id == user.id, Watchlist.stock_id == stock_id)
        )
        await session.commit()
    return {"ok": True, "symbol": symbol}
