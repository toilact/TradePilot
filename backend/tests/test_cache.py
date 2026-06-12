"""Test cache Redis (M6): HIT/MISS, degrade gracefully, bust theo prefix, bước pipeline.

Dùng fakeredis (async) — không cần Upstash thật. Bất biến quan trọng nhất:
Redis chết/không cấu hình → mọi helper vẫn trả data từ producer, KHÔNG raise.
"""

import pytest
from fakeredis.aioredis import FakeRedis

import cache
from cache import KEY_PREFIX, bust_cache, cached_json


@pytest.fixture
def fake_redis():
    client = FakeRedis(decode_responses=True)
    cache.set_client(client)
    yield client
    cache.reset_client()


@pytest.fixture
def no_redis():
    """Mô phỏng REDIS_URL rỗng: client = None."""
    cache.set_client(None)
    yield
    cache.reset_client()


@pytest.fixture
def broken_redis():
    """Redis hỏng: mọi thao tác raise — cached_json/bust phải nuốt và degrade."""

    class _BrokenRedis:
        async def get(self, key):
            raise ConnectionError("redis down")

        async def set(self, key, value, ex=None):
            raise ConnectionError("redis down")

        async def scan_iter(self, match=None):
            # async generator đúng semantics redis.asyncio — lỗi nổ lúc iterate
            raise ConnectionError("redis down")
            yield  # pragma: no cover — đánh dấu đây là generator

        async def delete(self, *keys):
            raise ConnectionError("redis down")

    cache.set_client(_BrokenRedis())
    yield
    cache.reset_client()


async def test_miss_then_hit(fake_redis):
    calls = 0

    async def produce():
        nonlocal calls
        calls += 1
        return {"a": 1}

    data1, status1 = await cached_json(f"{KEY_PREFIX}t", 60, produce)
    data2, status2 = await cached_json(f"{KEY_PREFIX}t", 60, produce)
    assert (status1, status2) == ("MISS", "HIT")
    assert data1 == data2 == {"a": 1}
    assert calls == 1  # HIT không gọi lại producer/DB


async def test_no_redis_degrades_to_producer(no_redis):
    async def produce():
        return [1, 2]

    data, status = await cached_json(f"{KEY_PREFIX}t", 60, produce)
    assert (data, status) == ([1, 2], "MISS")


async def test_broken_redis_degrades_not_raise(broken_redis):
    async def produce():
        return {"ok": True}

    data, status = await cached_json(f"{KEY_PREFIX}t", 60, produce)
    assert (data, status) == ({"ok": True}, "MISS")


async def test_date_in_producer_result_serializable(fake_redis):
    """Producer trả date/datetime → jsonable_encoder lo (JSONResponse không tự encode)."""
    from datetime import date

    async def produce():
        return {"d": date(2026, 6, 12)}

    data1, _ = await cached_json(f"{KEY_PREFIX}t", 60, produce)
    data2, status2 = await cached_json(f"{KEY_PREFIX}t", 60, produce)
    assert data1 == data2 == {"d": "2026-06-12"}
    assert status2 == "HIT"


async def test_producer_exception_not_cached(fake_redis):
    """404 raise trong producer → propagate, KHÔNG ghi cache (mã sai không chiếm key)."""

    async def boom():
        raise ValueError("404")

    with pytest.raises(ValueError):
        await cached_json(f"{KEY_PREFIX}t", 60, boom)
    assert await fake_redis.get(f"{KEY_PREFIX}t") is None


async def test_key_must_have_prefix(fake_redis):
    async def produce():
        return 1

    with pytest.raises(AssertionError):
        await cached_json("khong-prefix", 60, produce)


async def test_bust_only_tp_keys(fake_redis):
    await fake_redis.set(f"{KEY_PREFIX}a", "1")
    await fake_redis.set(f"{KEY_PREFIX}b", "2")
    await fake_redis.set("khac:c", "3")
    assert await bust_cache() == 2
    assert await fake_redis.get("khac:c") == "3"  # không FLUSHDB — key ngoài namespace còn


async def test_bust_no_redis_returns_none(no_redis):
    assert await bust_cache() is None


async def test_bust_empty_returns_zero(fake_redis):
    assert await bust_cache() == 0


async def test_invalid_redis_url_degrades(monkeypatch):
    """Dán nhầm REST token thay vì rediss://... → get_client trả None, không raise."""
    from config import settings

    cache.reset_client()
    monkeypatch.setattr(settings, "redis_url", "khong-phai-url")
    try:
        assert cache.get_client() is None
    finally:
        cache.reset_client()


# --- bước pipeline cache_bust (scheduler M6) ---


async def test_step_cache_bust_skip_without_redis(no_redis):
    from services.scheduler import _step_cache_bust

    result = await _step_cache_bust()
    assert result.ok and not result.warning
    assert "skip" in result.detail


async def test_step_cache_bust_deletes(fake_redis):
    from services.scheduler import _step_cache_bust

    await fake_redis.set(f"{KEY_PREFIX}x", "1")
    result = await _step_cache_bust()
    assert result.ok and "1" in result.detail


async def test_step_cache_bust_redis_error_is_warning_not_fail(broken_redis):
    from services.scheduler import _step_cache_bust

    result = await _step_cache_bust()
    assert result.ok and result.warning  # ⚠️ chứ không ❌ — chờ TTL tự hết


def test_pipeline_has_cache_bust_last():
    """Chốt contract M6: pipeline 6 bước, cache_bust CUỐI (bust sau khi data mới đã ghi)."""
    from services.scheduler import PIPELINE_STEPS

    names = [s.__name__.removeprefix("_step_") for s in PIPELINE_STEPS]
    assert names == ["prices", "actual_results", "news", "sentiment", "inference", "cache_bust"]


# --- X-Cache header ở tầng route ---


async def test_route_sets_x_cache_header(fake_redis, session_factory):
    import json

    from api.stocks import get_accuracy

    async with session_factory() as session:
        resp1 = await get_accuracy(session=session)
        resp2 = await get_accuracy(session=session)
    assert resp1.headers["X-Cache"] == "MISS"
    assert resp2.headers["X-Cache"] == "HIT"
    assert json.loads(resp1.body) == json.loads(resp2.body)
