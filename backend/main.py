"""TradePilot backend — FastAPI entrypoint.

Chạy dev: uv run uvicorn main:app --reload
Kiểm tra: http://localhost:8000/health  và  http://localhost:8000/docs
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from logging_config import configure_logging, init_sentry

configure_logging()
logger = structlog.get_logger(__name__)

if init_sentry():
    logger.info("sentry_initialized", environment=settings.app_env)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402 — sau Sentry init (docs Sentry)

from api import auth, news, predictions, stocks  # noqa: E402
from models.database import get_session  # noqa: E402


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
    allow_origins=settings.allowed_origins_list,  # M7: env ALLOWED_ORIGINS (Vercel prod + dev)
    allow_origin_regex=settings.allowed_origin_regex or None,  # preview trade-pilot-kato-*
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cache"],  # cho client cross-origin đọc được trạng thái cache (M6)
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
app.include_router(news.router)
app.include_router(auth.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tradepilot-backend"}


@app.get("/healthz")
async def healthz(session: AsyncSession = Depends(get_session)):  # noqa: B008 — pattern FastAPI
    """Health check production (M7): version + DB ping.

    Render health check + cron-job.org keep-alive gọi route này.
    DB chết → 503 để hệ thống ping bên ngoài báo động được.
    """
    try:
        await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        logger.exception("healthz_db_ping_failed")
        db_status = "error"
    body = {
        "status": "ok" if db_status == "ok" else "degraded",
        "version": app.version,
        "db": db_status,
    }
    return JSONResponse(body, status_code=200 if db_status == "ok" else 503)
