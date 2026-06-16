"""TradePilot backend — FastAPI entrypoint.

Chạy dev: uv run uvicorn main:app --reload
Kiểm tra: http://localhost:8000/health  và  http://localhost:8000/docs
"""

import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from logging_config import configure_logging, init_sentry

configure_logging()
logger = structlog.get_logger(__name__)

if init_sentry():
    logger.info("sentry_initialized", environment=settings.app_env)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402 — sau Sentry init (docs Sentry)

from api import auth, news, predictions, stocks  # noqa: E402
from models.database import Prediction, get_session  # noqa: E402

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
# Watchdog freshness (M10): cho phép trễ tối đa 2 ngày giao dịch (lưới cho 1 phiên lỡ + cuối tuần).
FRESHNESS_MAX_TRADING_DAYS = 2


def _trading_days_between(latest: date, today: date) -> int:
    """Số ngày giao dịch (T2–T6) từ `latest` (loại trừ) tới `today` (gồm cả).

    Bỏ cuối tuần; KHÔNG trừ ngày lễ (chấp nhận false-positive hiếm — đủ cho solo, watchdog
    chỉ cần bắt được "Mac quên chạy" vài ngày liền)."""
    days = 0
    d = latest + timedelta(days=1)
    while d <= today:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)
    return days


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
async def healthz(
    check: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008 — pattern FastAPI
):
    """Health check production (M7) + watchdog freshness (M10).

    Mặc định (Render health check): version + DB ping; DB chết → 503.
    `?check=freshness` (cron-job.org gọi): thêm kiểm tra prediction mới nhất — quá
    FRESHNESS_MAX_TRADING_DAYS ngày giao dịch → 503 (Mac quên chạy pipeline → báo động).
    GIỮ default KHÔNG đổi để prediction cũ không khiến Render kill service.
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
    if db_status != "ok":
        return JSONResponse(body, status_code=503)

    if check == "freshness":
        from services.inference import MODEL_VERSION

        latest = (
            await session.execute(
                select(func.max(Prediction.prediction_date)).where(
                    Prediction.model_version == MODEL_VERSION
                )
            )
        ).scalar()
        today = datetime.now(TZ_VN).date()
        stale_days = _trading_days_between(latest, today) if latest is not None else None
        fresh = latest is not None and stale_days <= FRESHNESS_MAX_TRADING_DAYS
        body["freshness"] = {
            "latestPrediction": latest.isoformat() if latest else None,
            "tradingDaysStale": stale_days,
            "fresh": fresh,
        }
        if not fresh:
            body["status"] = "stale"
            return JSONResponse(body, status_code=503)

    return JSONResponse(body, status_code=200)
