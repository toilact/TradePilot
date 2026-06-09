"""Job chạy 1 lần/ngày lúc ~16:00 (sau đóng cửa) cho top 100 mã.

Thứ tự: price_fetcher → crawler → sentiment(score) → build_daily_sentiment
        → TFT inference → ghi predictions → invalidate Redis cache
        → fill actual_results khi có close T+1.

Bất biến chống leakage: inference ngày T chỉ dùng tin published_at ≤ 16:00 phiên T.
TODO Phase 1.4: nối các service + alert khi job fail hoặc crawl 0 tin.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")


async def run_daily_pipeline() -> None:
    """Chạy toàn bộ pipeline cho 1 phiên. (chưa implement)"""
    raise NotImplementedError("Phase 1.4: nối pipeline + logging + alert")


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=TZ_VN)
    # day_of_week mon-fri: bỏ cuối tuần (không có phiên giao dịch).
    scheduler.add_job(run_daily_pipeline, CronTrigger(day_of_week="mon-fri", hour=16, minute=0))
    return scheduler
