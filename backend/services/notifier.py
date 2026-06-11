"""Notifier Telegram — alert khi daily job fail / crawl 0 tin (M4 observability).

Nguyên tắc:
- KHÔNG BAO GIỜ raise: notifier là kênh báo lỗi, nó mà giết pipeline thì phản tác dụng.
  Lỗi mạng/cấu hình → log warning + trả False, caller tự quyết.
- Thiếu token/chat_id (.env) → bỏ qua êm (dev/CI không cần Telegram).
- Bot API `sendMessage` qua httpx (đã có trong deps), parse_mode HTML.

Test gửi thật: cd backend && uv run python -m services.notifier --test
"""

from __future__ import annotations

import argparse
import asyncio

import httpx
import structlog

from config import settings

logger = structlog.get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org"
_TIMEOUT = 10.0


async def send_telegram(text: str) -> bool:
    """Gửi `text` tới chat đã cấu hình. True nếu gửi thành công, False mọi trường hợp khác."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        logger.warning("telegram_not_configured", hint="set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
        return False

    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        # KHÔNG log url/exc nguyên văn — có thể chứa token trong path
        logger.warning("telegram_send_failed", error=type(exc).__name__, detail=_safe(str(exc)))
        return False
    logger.info("telegram_sent", chars=len(text))
    return True


def _safe(msg: str) -> str:
    """Che token bot nếu lọt vào message lỗi (httpx in URL trong repr)."""
    token = settings.telegram_bot_token
    return msg.replace(token, "***") if token else msg


def _main() -> None:
    from logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Notifier Telegram TradePilot")
    parser.add_argument("--test", action="store_true", help="Gửi tin nhắn test thật")
    args = parser.parse_args()
    if not args.test:
        parser.error("dùng --test để gửi tin nhắn thử")

    ok = asyncio.run(send_telegram("✅ TradePilot notifier hoạt động — tin nhắn test (M4)."))
    print("✓ Đã gửi Telegram" if ok else "✗ Gửi thất bại — xem log (token/chat_id/mạng)")


if __name__ == "__main__":
    _main()
