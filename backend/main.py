"""TradePilot backend — FastAPI entrypoint.

Chạy dev: uv run uvicorn main:app --reload
Kiểm tra: http://localhost:8000/health  và  http://localhost:8000/docs
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from config import settings
from logging_config import configure_logging, init_sentry

configure_logging()
logger = structlog.get_logger(__name__)

if init_sentry():
    logger.info("sentry_initialized", environment=settings.app_env)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402 — sau Sentry init (docs Sentry)

from api import auth, predictions, stocks  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Trigger chính của pipeline là launchd 16:05 (scripts/run_daily_pipeline.py).
    # APScheduler in-app chỉ bật qua ENABLE_SCHEDULER (deploy sau — Render không có launchd).
    scheduler = None
    if settings.enable_scheduler:
        from services.scheduler import create_scheduler

        scheduler = create_scheduler()
        scheduler.start()
        logger.info("scheduler_started", cron="16:00 mon-fri Asia/Ho_Chi_Minh")
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(title="TradePilot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log JSON mỗi request: path, status, duration_ms. Exception vẫn log rồi re-raise."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        raise
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


app.include_router(predictions.router)
app.include_router(stocks.router)
app.include_router(auth.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tradepilot-backend"}
