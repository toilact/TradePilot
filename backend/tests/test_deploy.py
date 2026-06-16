"""Test M7 — /healthz (version + db ping) và cấu hình CORS từ env.

Gọi route function trực tiếp (pattern test_cache) — không cần TestClient/Supabase.
"""

import datetime as dt
import json

import pytest

from config import settings
from main import FRESHNESS_MAX_TRADING_DAYS, TZ_VN, app, healthz
from main import _trading_days_between as tdb


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


class _FakeResult:
    def __init__(self, value):
        self._v = value

    def scalar(self):
        return self._v


class _FreshnessSession:
    """Fake session: execute lần 1 = ping (bỏ qua), lần 2 = max(prediction_date)."""

    def __init__(self, latest):
        self.latest = latest
        self.calls = 0

    async def execute(self, stmt):
        self.calls += 1
        return _FakeResult(None if self.calls == 1 else self.latest)


def test_trading_days_between():
    # 06-12 Thứ Sáu (xác nhận trong CONTEXT) → 06-15 Mon / 06-16 Tue / 06-17 Wed
    assert tdb(dt.date(2026, 6, 15), dt.date(2026, 6, 17)) == 2  # Mon→Wed
    assert tdb(dt.date(2026, 6, 12), dt.date(2026, 6, 15)) == 1  # Fri→Mon (bỏ T7/CN)
    assert tdb(dt.date(2026, 6, 16), dt.date(2026, 6, 16)) == 0  # cùng ngày


async def test_healthz_default_no_freshness_field():
    """Mặc định (check=None): KHÔNG có field freshness — giữ nguyên contract Render health check."""
    resp = await healthz(session=_FreshnessSession(None))
    assert resp.status_code == 200
    assert "freshness" not in json.loads(resp.body)


async def test_healthz_freshness_fresh():
    from datetime import datetime

    today = datetime.now(TZ_VN).date()
    resp = await healthz(check="freshness", session=_FreshnessSession(today))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["status"] == "ok"
    assert body["freshness"]["fresh"] is True
    assert body["freshness"]["tradingDaysStale"] == 0


async def test_healthz_freshness_stale_returns_503():
    from datetime import datetime, timedelta

    stale = datetime.now(TZ_VN).date() - timedelta(days=30)
    resp = await healthz(check="freshness", session=_FreshnessSession(stale))
    assert resp.status_code == 503
    body = json.loads(resp.body)
    assert body["status"] == "stale"
    assert body["freshness"]["fresh"] is False
    assert body["freshness"]["tradingDaysStale"] > FRESHNESS_MAX_TRADING_DAYS


async def test_healthz_freshness_no_predictions_returns_503():
    resp = await healthz(check="freshness", session=_FreshnessSession(None))
    assert resp.status_code == 503
    assert json.loads(resp.body)["freshness"]["latestPrediction"] is None


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
