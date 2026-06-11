"""Test notifier Telegram (mock httpx — không gọi mạng) + logging JSON (M4).

Bất biến kiểm tra:
- send_telegram KHÔNG BAO GIỜ raise (lỗi mạng → False) — notifier không được giết pipeline.
- Thiếu cấu hình → False + không tạo request nào.
- configure_logging: log stdlib lẫn structlog đều xuất JSON 1 dòng.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from config import settings
from services import notifier
from services.notifier import send_telegram


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


class _FakeClient:
    """Thay httpx.AsyncClient — ghi lại call, không chạm mạng."""

    calls: list[tuple[str, dict]] = []
    status_code = 200
    raise_exc: Exception | None = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, json: dict | None = None):
        if _FakeClient.raise_exc is not None:
            raise _FakeClient.raise_exc
        _FakeClient.calls.append((url, json or {}))
        return _FakeResponse(_FakeClient.status_code)


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.status_code = 200
    _FakeClient.raise_exc = None
    monkeypatch.setattr(notifier.httpx, "AsyncClient", _FakeClient)
    return _FakeClient


@pytest.fixture
def telegram_configured(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "123:ABC")
    monkeypatch.setattr(settings, "telegram_chat_id", "42")


async def test_send_ok(fake_httpx, telegram_configured):
    ok = await send_telegram("xin chào")
    assert ok is True
    assert len(fake_httpx.calls) == 1
    url, payload = fake_httpx.calls[0]
    assert url == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert payload["chat_id"] == "42"
    assert payload["text"] == "xin chào"


async def test_not_configured_returns_false(fake_httpx, monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    monkeypatch.setattr(settings, "telegram_chat_id", "")
    ok = await send_telegram("hi")
    assert ok is False
    assert fake_httpx.calls == []  # không tạo request nào


async def test_http_error_returns_false_khong_raise(fake_httpx, telegram_configured):
    fake_httpx.status_code = 500
    assert await send_telegram("hi") is False


async def test_network_error_returns_false_khong_raise(fake_httpx, telegram_configured):
    fake_httpx.raise_exc = httpx.ConnectError("mạng rớt")
    assert await send_telegram("hi") is False


def test_logging_json_ca_stdlib_lan_structlog(capsys):
    """Sau configure_logging, log stdlib (services cũ) lẫn structlog đều ra JSON parse được."""
    import structlog

    from logging_config import configure_logging

    configure_logging("INFO")
    logging.getLogger("test_stdlib").info("hello %s", "world")
    structlog.get_logger("test_structlog").info("sự_kiện", symbol="VCB", rows=3)

    lines = [ln for ln in capsys.readouterr().err.strip().splitlines() if ln]
    assert len(lines) >= 2
    stdlib_rec = json.loads(lines[-2])
    struct_rec = json.loads(lines[-1])
    assert stdlib_rec["event"] == "hello world"
    assert stdlib_rec["level"] == "info"
    assert struct_rec["event"] == "sự_kiện"  # ensure_ascii=False — tiếng Việt giữ nguyên
    assert struct_rec["symbol"] == "VCB"
    assert struct_rec["rows"] == 3
