"""Redis cache (Upstash) cho read API — M6.

Nguyên tắc DEGRADE GRACEFULLY (bất biến của module này):
- `REDIS_URL` rỗng → không tạo client, mọi helper trả thẳng kết quả từ producer/DB.
- Redis lỗi runtime (mạng, timeout, auth) → log warning rồi coi như cache miss,
  KHÔNG BAO GIỜ raise ra route. Cache chết thì web chậm đi, không được chết theo.

Key namespace: mọi key có prefix `tp:` — cuối daily pipeline bust bằng SCAN `tp:*` + DEL
(xem `bust_cache` + bước `cache_bust` trong services/scheduler.py).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from config import settings

logger = structlog.get_logger(__name__)

KEY_PREFIX = "tp:"

# TTL theo endpoint (giây) — data đổi 1 lần/ngày sau pipeline; pipeline bust chủ động
# nên TTL chỉ là lưới an toàn (vd quên bust / sửa data tay).
TTL_PREDICTIONS = 3600
TTL_ACCURACY = 3600
TTL_HISTORY = 3600
TTL_NEWS = 900

# Timeout ngắn để degrade NHANH khi Redis chết — user không chờ 30s mặc định.
_SOCKET_TIMEOUT_SECONDS = 2.0

_client = None
_client_initialized = False


def get_client():
    """Redis client singleton (None nếu không cấu hình REDIS_URL).

    Lazy + cache cả kết quả None để không parse URL lại mỗi request.
    Test inject client giả qua `set_client()`.
    """
    global _client, _client_initialized
    if _client_initialized:
        return _client
    if settings.redis_url:
        import redis.asyncio as redis

        try:
            _client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=_SOCKET_TIMEOUT_SECONDS,
                socket_timeout=_SOCKET_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # URL sai định dạng (vd dán nhầm REST token thay vì rediss://...) —
            # degrade như không có Redis, KHÔNG để route 500.
            logger.warning("redis_url_invalid", error=str(exc))
            _client = None
    _client_initialized = True
    return _client


def set_client(client) -> None:
    """Inject client cho test: fakeredis, hoặc None = mô phỏng không có REDIS_URL."""
    global _client, _client_initialized
    _client = client
    _client_initialized = True


def reset_client() -> None:
    """Quên client đã init — get_client lần sau đọc lại settings (teardown test)."""
    global _client, _client_initialized
    _client = None
    _client_initialized = False


async def cached_json(
    key: str, ttl: int, producer: Callable[[], Awaitable[Any]]
) -> tuple[Any, str]:
    """Trả (data, "HIT"|"MISS") — đọc cache trước, miss thì gọi producer rồi set.

    `key` PHẢI bắt đầu bằng KEY_PREFIX (để bust theo prefix không sót).
    Kết quả producer qua `jsonable_encoder` (date/datetime/Decimal → JSON-native) vì
    cả json.dumps lẫn JSONResponse đều KHÔNG tự encode như return-dict FastAPI mặc định.
    Mọi lỗi Redis → coi như miss, không raise.
    """
    assert key.startswith(KEY_PREFIX), f"cache key phải có prefix {KEY_PREFIX!r}: {key}"
    client = get_client()
    if client is not None:
        try:
            raw = await client.get(key)
            if raw is not None:
                return json.loads(raw), "HIT"
        except Exception as exc:
            logger.warning("cache_get_failed", key=key, error=str(exc))
    data = jsonable_encoder(await producer())
    if client is not None:
        try:
            await client.set(key, json.dumps(data), ex=ttl)
        except Exception as exc:
            logger.warning("cache_set_failed", key=key, error=str(exc))
    return data, "MISS"


async def cached_response(
    key: str, ttl: int, producer: Callable[[], Awaitable[Any]]
) -> JSONResponse:
    """JSONResponse + header `X-Cache: HIT|MISS` — wrapper dùng chung cho mọi route đọc."""
    data, status = await cached_json(key, ttl, producer)
    return JSONResponse(content=data, headers={"X-Cache": status})


async def bust_cache() -> int | None:
    """Xoá mọi key `tp:*` (cuối daily pipeline). Trả số key đã xoá, None nếu không có Redis.

    SCAN thay vì KEYS (không block Redis), gom 1 lệnh DEL (không round-trip từng key);
    KHÔNG FLUSHDB — instance có thể dùng chung cho việc khác sau này (rate-limit, session).
    """
    client = get_client()
    if client is None:
        return None
    keys = [key async for key in client.scan_iter(match=f"{KEY_PREFIX}*")]
    if not keys:
        return 0
    return await client.delete(*keys)
