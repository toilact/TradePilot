"""Test M9 — verify JWT HS256 (Auth.js ↔ FastAPI) + watchlist CRUD.

"Integration test verify JWT SỚM NHẤT" (ticklist M9): mint token bằng PyJWT với cùng
AUTH_SECRET = chính hợp đồng mà Auth.js (jose HS256) phải khớp. Gọi route function trực
tiếp (style test_gating/test_deploy) — không cần TestClient.
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from sqlalchemy import select

from api.auth import (
    WatchlistAdd,
    add_watchlist,
    get_current_user,
    get_watchlist,
    remove_watchlist,
)
from config import settings
from models.database import Stock, User

SECRET = "test-secret-m9"


@pytest.fixture(autouse=True)
def _auth_secret(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", SECRET)


def _mint(email="user@example.com", name="User", secret=SECRET, exp_delta=timedelta(hours=1)):
    now = datetime.now(UTC)
    payload = {"email": email, "name": name, "iat": now, "exp": now + exp_delta}
    return jwt.encode(payload, secret, algorithm="HS256")


def _bearer(**kw) -> str:
    return f"Bearer {_mint(**kw)}"


async def _seed_stock(s, symbol="VCB") -> int:
    st = Stock(symbol=symbol, name=symbol, exchange="HSX", sector="Ngân hàng")
    s.add(st)
    await s.flush()
    return st.id


# --- 401: token thiếu / sai / hết hạn / thiếu email / chưa cấu hình secret ---


async def test_missing_header_401(session_factory):
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization=None, session=s)
    assert ei.value.status_code == 401


async def test_empty_bearer_401(session_factory):
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization="Bearer   ", session=s)
    assert ei.value.status_code == 401


async def test_wrong_scheme_401(session_factory):
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization=_mint(), session=s)  # thiếu "Bearer "
    assert ei.value.status_code == 401


async def test_wrong_secret_401(session_factory):
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization=_bearer(secret="khac-secret"), session=s)
    assert ei.value.status_code == 401


async def test_expired_token_401(session_factory):
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization=_bearer(exp_delta=timedelta(hours=-1)), session=s)
    assert ei.value.status_code == 401


async def test_token_without_email_401(session_factory):
    now = datetime.now(UTC)
    token = jwt.encode({"name": "X", "exp": now + timedelta(hours=1)}, SECRET, algorithm="HS256")
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization=f"Bearer {token}", session=s)
    assert ei.value.status_code == 401


async def test_unconfigured_secret_401(session_factory, monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", "")
    async with session_factory() as s:
        with pytest.raises(HTTPException) as ei:
            await get_current_user(authorization=_bearer(), session=s)
    assert ei.value.status_code == 401


# --- upsert users theo email (idempotent) ---


async def test_valid_token_upserts_user_once(session_factory):
    async with session_factory() as s:
        u1 = await get_current_user(authorization=_bearer(email="k@x.com", name="K1"), session=s)
        u2 = await get_current_user(authorization=_bearer(email="k@x.com", name="K2"), session=s)
        rows = (await s.execute(select(User).where(User.email == "k@x.com"))).scalars().all()
    assert u1.id == u2.id  # cùng email → cùng user row
    assert len(rows) == 1  # không tạo trùng
    assert u2.name == "K2"  # name refresh khi login lại
    assert u2.provider == "google"


# --- watchlist CRUD: 200 + idempotent + 404 + cô lập theo user ---


async def test_watchlist_add_get_delete_idempotent(session_factory):
    async with session_factory() as s:
        await _seed_stock(s, "VCB")
        user = await get_current_user(authorization=_bearer(), session=s)
        assert await get_watchlist(user=user, session=s) == []  # ban đầu rỗng

        r = await add_watchlist(WatchlistAdd(symbol="vcb"), user=user, session=s)  # normalize hoa
        assert r == {"ok": True, "symbol": "VCB"}
        wl = await get_watchlist(user=user, session=s)
        assert [x["symbol"] for x in wl] == ["VCB"]
        assert wl[0]["exchange"] == "HOSE"  # HSX → HOSE

        await add_watchlist(WatchlistAdd(symbol="VCB"), user=user, session=s)  # thêm lại
        assert len(await get_watchlist(user=user, session=s)) == 1  # KHÔNG nhân đôi

        await remove_watchlist(symbol="VCB", user=user, session=s)
        assert await get_watchlist(user=user, session=s) == []
        r2 = await remove_watchlist(symbol="VCB", user=user, session=s)  # xoá lại
        assert r2["ok"] is True  # idempotent


async def test_add_unknown_symbol_404(session_factory):
    async with session_factory() as s:
        user = await get_current_user(authorization=_bearer(), session=s)
        with pytest.raises(HTTPException) as ei:
            await add_watchlist(WatchlistAdd(symbol="ZZZ"), user=user, session=s)
    assert ei.value.status_code == 404


async def test_watchlist_isolated_per_user(session_factory):
    async with session_factory() as s:
        await _seed_stock(s, "VCB")
        await _seed_stock(s, "FPT")
        ua = await get_current_user(authorization=_bearer(email="a@x.com"), session=s)
        ub = await get_current_user(authorization=_bearer(email="b@x.com"), session=s)
        await add_watchlist(WatchlistAdd(symbol="VCB"), user=ua, session=s)
        await add_watchlist(WatchlistAdd(symbol="FPT"), user=ub, session=s)
        wl_a = await get_watchlist(user=ua, session=s)
        wl_b = await get_watchlist(user=ub, session=s)
    assert [x["symbol"] for x in wl_a] == ["VCB"]
    assert [x["symbol"] for x in wl_b] == ["FPT"]
