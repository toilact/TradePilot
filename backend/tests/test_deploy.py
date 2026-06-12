"""Test M7 — /healthz (version + db ping) và cấu hình CORS từ env.

Gọi route function trực tiếp (pattern test_cache) — không cần TestClient/Supabase.
"""

import json

import pytest

from config import settings
from main import app, healthz


async def test_healthz_ok(session_factory):
    async with session_factory() as session:
        resp = await healthz(session=session)
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body == {"status": "ok", "version": app.version, "db": "ok"}


async def test_healthz_db_error_returns_503():
    class _BrokenSession:
        async def execute(self, stmt):
            raise ConnectionError("db down")

    resp = await healthz(session=_BrokenSession())
    assert resp.status_code == 503
    body = json.loads(resp.body)
    assert (body["status"], body["db"]) == ("degraded", "error")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:3000", ["http://localhost:3000"]),
        (
            "https://tradepilot.vercel.app, http://localhost:3000",
            ["https://tradepilot.vercel.app", "http://localhost:3000"],
        ),
        ("", []),  # rỗng → không origin nào (regex vẫn có thể mở preview)
        ("a,,b ,", ["a", "b"]),  # bỏ phần tử rỗng + strip space
    ],
)
def test_allowed_origins_list_parsing(monkeypatch, raw, expected):
    monkeypatch.setattr(settings, "allowed_origins", raw)
    assert settings.allowed_origins_list == expected
