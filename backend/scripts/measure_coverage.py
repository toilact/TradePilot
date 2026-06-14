"""Đo COVERAGE tin tức: % phiên-mã có news (news_count>0) trên tổng phiên giao dịch, theo năm.

Pha 0 của M8-cải-tiến: binding constraint của "sentiment có ích cho dự đoán giá" là COVERAGE
(bao nhiêu phiên-mã thật sự có tin để chấm), không phải chất lượng PhoBERT. Số này quyết định
vùng "recent-dense" cho diagnostic (Pha 1) và có đáng cào thêm (Pha 2) hay không.

Mẫu số = số hàng (stock_id, trading_day) trong price_history (mọi phiên giao dịch thật).
Tử số   = số hàng đó có daily_sentiment.news_count > 0 (đã ánh xạ theo effective_trading_day).

CHỈ ĐỌC — không sửa gì. Chạy: cd backend && uv run python -m scripts.measure_coverage
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


async def _gather() -> tuple[dict[int, int], dict[int, int]]:
    """Trả (sessions_by_year, news_sessions_by_year) — đếm trên toàn panel 30 mã."""
    from sqlalchemy import select

    from models.database import DailySentiment, PriceHistory, SessionLocal

    sessions: dict[int, int] = defaultdict(int)
    news_sessions: dict[int, int] = defaultdict(int)
    async with SessionLocal() as s:
        for (d,) in (await s.execute(select(PriceHistory.date))).all():
            sessions[d.year] += 1
        # daily_sentiment chỉ ghi ngày CÓ tin → mỗi hàng news_count>0 là 1 phiên-mã có tin.
        for d, nc in (
            await s.execute(select(DailySentiment.date, DailySentiment.news_count))
        ).all():
            if nc and nc > 0:
                news_sessions[d.year] += 1
    return sessions, news_sessions


def _print_table(sessions: dict[int, int], news_sessions: dict[int, int]) -> None:
    print(f"{'năm':>6} {'phiên-mã':>10} {'có tin':>8} {'coverage':>10}")
    print("-" * 38)
    tot_s = tot_n = 0
    for year in sorted(sessions):
        s, n = sessions[year], news_sessions.get(year, 0)
        tot_s += s
        tot_n += n
        cov = 100.0 * n / s if s else 0.0
        print(f"{year:>6} {s:>10} {n:>8} {cov:>9.1f}%")
    print("-" * 38)
    cov = 100.0 * tot_n / tot_s if tot_s else 0.0
    print(f"{'TỔNG':>6} {tot_s:>10} {tot_n:>8} {cov:>9.1f}%")


def _main() -> None:
    sessions, news_sessions = asyncio.run(_gather())
    if not sessions:
        print("price_history rỗng — chưa có data giá.")
        return
    _print_table(sessions, news_sessions)


if __name__ == "__main__":
    _main()
