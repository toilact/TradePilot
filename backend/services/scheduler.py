"""Pipeline 1 lần/ngày ~16:00 (sau đóng cửa) cho VN30 — M5.

Thứ tự (mỗi bước try/except RIÊNG — 1 bước fail không giết pipeline):
  1. prices          — OHLCV vnstock (full refetch: giá adjusted cập nhật hồi tố)
  2. actual_results  — chấm prediction hôm qua (cần close HÔM NAY → phải sau bước 1;
                       vẫn trước inference, đúng tinh thần "chấm trước khi ghi mới")
  3. news            — crawl RSS + trang CafeF từng mã (lịch sự, delay 1s)
  4. sentiment       — score_news (stub trả 0 — M8 thay PhoBERT) + daily_sentiment
  5. inference       — lgbm_v4 + gating (cần `uv run --group inference`)
  6. cache_bust      — xoá Redis `tp:*` để web phục vụ data mới ngay (M6; không Redis → skip)
Cuối cùng: Telegram tổng kết (kèm ⚠️ khi crawl 0 tin / ❌ khi bước fail).

Bất biến chống leakage: inference ngày T chỉ dùng tin published_at ≤ 16:00 phiên T.
Ngày nghỉ lễ: vẫn chạy hết — các bước idempotent (không giá mới → inference ghi lại
đúng row cũ, không duplicate); tổng kết sẽ hiện "0 dòng giá mới".

Trigger thật: launchd 16:05 mon–fri (scripts/run_daily_pipeline.py). APScheduler trong
app chỉ bật khi ENABLE_SCHEDULER=true (default false — dành cho deploy sau).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
logger = structlog.get_logger(__name__)

# vnstock guest = 20 request/phút; fetch ~1.5s + nghỉ 2s ≈ 3.5s/mã → ~17 req/phút, dưới trần.
# (Đăng ký API key free lên 60 req/phút nếu muốn nhanh hơn — không bắt buộc.)
PRICE_PACING_SECONDS = 2.0


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str  # mô tả ngắn (số liệu) hoặc thông điệp lỗi
    warning: bool = False  # ok nhưng đáng chú ý (vd crawl 0 tin)


async def _step_prices() -> StepResult:
    from services.price_fetcher import fetch_price_history
    from services.stock_seed import VN30

    total, failed = 0, []
    for symbol in VN30:
        # vnstock rate-limit thoát bằng SystemExit (không phải Exception) — phải bắt
        # per-symbol để 1 mã lỗi không mất cả 30 mã.
        try:
            total += await fetch_price_history(symbol)
        except (Exception, SystemExit):
            logger.exception("price_fetch_failed", symbol=symbol)
            failed.append(symbol)
        await asyncio.sleep(PRICE_PACING_SECONDS)
    if len(failed) == len(VN30):
        raise RuntimeError(f"giá fail toàn bộ {len(VN30)} mã — vnstock sập/đổi API?")
    if failed:
        ok_count = len(VN30) - len(failed)
        detail = f"{ok_count}/{len(VN30)} mã, upsert {total} dòng; FAIL: {', '.join(failed)}"
        return StepResult("prices", True, detail, warning=True)
    return StepResult("prices", True, f"{len(VN30)} mã, upsert {total} dòng")


async def _step_actual_results() -> StepResult:
    from models.database import SessionLocal
    from services.actual_results import fill_actual_results

    async with SessionLocal() as session:
        n = await fill_actual_results(session)
    return StepResult("actual_results", True, f"chấm {n} nhãn thực tế")


async def _step_news() -> StepResult:
    from services.crawler import crawl_news, crawl_rss
    from services.stock_seed import VN30

    new_total = await crawl_rss()
    for symbol in VN30:
        new_total += await crawl_news(symbol)
    if new_total == 0:
        # bất biến alert: crawl 0 tin đáng ngờ (selector vỡ / bị chặn) — cảnh báo, không fail
        return StepResult("news", True, "0 bài mới — kiểm tra crawler/nguồn", warning=True)
    return StepResult("news", True, f"{new_total} bài mới")


async def _step_sentiment() -> StepResult:
    from services.sentiment import build_daily_sentiment, score_news
    from services.stock_seed import VN30

    scored = await score_news()
    days = 0
    for symbol in VN30:
        days += await build_daily_sentiment(symbol)
    return StepResult("sentiment", True, f"chấm {scored} tin (stub=0), gộp {days} ngày-mã")


async def _step_inference() -> StepResult:
    from sqlalchemy import func, select

    from models.database import Prediction, SessionLocal
    from services.inference import MODEL_VERSION, run_inference

    n = await run_inference()
    async with SessionLocal() as session:
        latest = (
            await session.execute(
                select(func.max(Prediction.prediction_date)).where(
                    Prediction.model_version == MODEL_VERSION
                )
            )
        ).scalar()
        actionable = (
            await session.execute(
                select(func.count())
                .select_from(Prediction)
                .where(
                    Prediction.model_version == MODEL_VERSION,
                    Prediction.prediction_date == latest,
                    Prediction.is_actionable.is_(True),
                )
            )
        ).scalar()
    return StepResult("inference", True, f"{n} predictions @ {latest}, actionable {actionable}")


async def _step_cache_bust() -> StepResult:
    """Xoá cache `tp:*` để web phục vụ predictions/news MỚI ngay (M6 — TTL chỉ là lưới an toàn).

    Tự bắt lỗi Redis → warning (không ❌): cache bust fail thì data cũ sống thêm tối đa
    1 TTL, không đáng coi là pipeline fail.
    """
    from cache import bust_cache

    try:
        deleted = await bust_cache()
    except Exception as exc:
        logger.warning("cache_bust_failed", error=str(exc))
        return StepResult("cache_bust", True, f"Redis lỗi — chờ TTL tự hết: {exc}", warning=True)
    if deleted is None:
        return StepResult("cache_bust", True, "skip — không có REDIS_URL")
    return StepResult("cache_bust", True, f"xoá {deleted} key")


PIPELINE_STEPS = (
    _step_prices,
    _step_actual_results,
    _step_news,
    _step_sentiment,
    _step_inference,
    _step_cache_bust,
)


def build_summary(results: list[StepResult]) -> str:
    """Tin nhắn Telegram tổng kết. Hàm THUẦN để test."""
    failed = [r for r in results if not r.ok]
    warned = [r for r in results if r.ok and r.warning]
    if failed:
        head = f"❌ TradePilot pipeline: {len(failed)} bước FAIL"
    elif warned:
        head = "⚠️ TradePilot pipeline: xong, có cảnh báo"
    else:
        head = "✅ TradePilot pipeline: xong"
    now = datetime.now(TZ_VN).strftime("%Y-%m-%d %H:%M")
    lines = [f"{head} — {now}"]
    for r in results:
        icon = "❌" if not r.ok else ("⚠️" if r.warning else "✅")
        lines.append(f"{icon} {r.name}: {r.detail}")
    return "\n".join(lines)


async def run_daily_pipeline(steps=PIPELINE_STEPS, notify=None) -> list[StepResult]:
    """Chạy tuần tự các bước, mỗi bước try/except riêng, cuối cùng Telegram tổng kết.

    `steps`/`notify` inject được để test (không network/model). Lỗi từng bước đã
    logger.exception → lên Sentry qua LoggingIntegration (khi có DSN).
    """
    if notify is None:
        from services.notifier import send_telegram

        notify = send_telegram

    results: list[StepResult] = []
    for step in steps:
        name = step.__name__.removeprefix("_step_")
        logger.info("pipeline_step_start", step=name)
        try:
            result = await step()
        # SystemExit kèm theo: vnstock rate-limit gọi sys.exit() — không được giết pipeline
        # (bài học lần chạy DoD đầu: chết giữa bước 1, không Telegram). KeyboardInterrupt vẫn thoát.
        except (Exception, SystemExit) as exc:
            logger.exception("pipeline_step_failed", step=name)
            result = StepResult(name, False, f"{type(exc).__name__}: {exc}")
        else:
            logger.info("pipeline_step_ok", step=name, detail=result.detail, warning=result.warning)
        results.append(result)

    summary = build_summary(results)
    try:
        await notify(summary)
    except Exception:  # send_telegram vốn không raise — phòng notify inject lỗi
        logger.exception("pipeline_notify_failed")
    logger.info("pipeline_done", ok=all(r.ok for r in results))
    return results


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=TZ_VN)
    # day_of_week mon-fri: bỏ cuối tuần (không có phiên giao dịch).
    scheduler.add_job(run_daily_pipeline, CronTrigger(day_of_week="mon-fri", hour=16, minute=0))
    return scheduler
