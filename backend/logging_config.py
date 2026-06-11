"""Structured logging JSON cho toàn backend (M4 — observability nền).

structlog + ProcessorFormatter: MỌI log (structlog lẫn stdlib `logging` — gồm uvicorn,
sqlalchemy, các services dùng `logging.getLogger`) đi qua 1 formatter duy nhất, xuất
JSON 1 dòng/event ra stdout. Không file handler, không OpenTelemetry — đúng phạm vi M4.

Dùng:
    from logging_config import configure_logging
    configure_logging()                      # gọi 1 lần ở entrypoint (main.py / script)
    logger = structlog.get_logger(__name__)  # trong module
    logger.info("price_fetched", symbol="VCB", rows=120)
"""

from __future__ import annotations

import logging

import structlog

from config import settings

# Processor chung: áp cho cả event structlog lẫn record stdlib (foreign_pre_chain).
_SHARED_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(level: str | None = None) -> None:
    """Cấu hình root logger xuất JSON. Idempotent — gọi nhiều lần không nhân đôi handler."""
    log_level = (level or settings.log_level or "INFO").upper()

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # ensure_ascii=False: log tiếng Việt đọc được, không bị \uXXXX
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]  # thay sạch (idempotent), không addHandler chồng
    root.setLevel(log_level)

    # uvicorn tự gắn handler riêng (format text) → gỡ, cho propagate lên root JSON.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # access log tắt hẳn — middleware log_requests (main.py) đã log request JSON, tránh đúp.
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False


def init_sentry() -> bool:
    """Init Sentry nếu có SENTRY_DSN (không có → no-op, app chạy bình thường).

    traces_sample_rate=0: chỉ bắt lỗi, không APM — đúng phạm vi M4, tiết kiệm quota free.
    LoggingIntegration mặc định: logger.exception/error (kể cả qua structlog→stdlib)
    tự thành Sentry event. Dùng chung cho main.py (API) lẫn scripts (pipeline).
    """
    if not settings.sentry_dsn:
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.app_env,
        traces_sample_rate=0.0,
        send_default_pii=False,
    )
    return True
