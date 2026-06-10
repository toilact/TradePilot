"""Fill actual_results — nhãn THỰC TẾ cho mỗi prediction, để trang accuracy chấm điểm.

Mắt xích "fill actual_results khi có close T+1" trong pipeline (xem scheduler.py).

Nguyên tắc:
- Nhãn thực tế của prediction ngày T = label_from_close(close[T], close[phiên kế tiếp]).
- Phiên kế tiếp = bản ghi price_history có date > T NHỎ NHẤT (không phải +1 ngày lịch) →
  tự động đúng cả cuối tuần/nghỉ lễ vì price_history chỉ chứa phiên THẬT.
- ActualResult.date = prediction_date (ngày T) — theo contract ở models/database.py.
- Idempotent (upsert theo stock_id+date). Chỉ ghi khi đã có close phiên kế tiếp; chưa tới
  T+1 → bỏ qua (lần chạy sau sẽ fill).
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.upsert import upsert
from models.database import ActualResult, Prediction, PriceHistory
from services.labeling import label_from_close


async def fill_actual_results(session: AsyncSession) -> int:
    """Tính + ghi nhãn thực tế cho các prediction đã có close phiên kế tiếp. Trả số actual ghi.

    Gom query (không N+1): distinct (stock_id, prediction_date) 1 lần + toàn bộ price 1 lần,
    rồi tìm phiên kế tiếp in-memory bằng bisect trên list ngày sắp tăng dần.
    """
    targets = (
        await session.execute(select(Prediction.stock_id, Prediction.prediction_date).distinct())
    ).all()
    if not targets:
        return 0

    stock_ids = {sid for sid, _ in targets}
    price_rows = (
        await session.execute(
            select(PriceHistory.stock_id, PriceHistory.date, PriceHistory.close)
            .where(PriceHistory.stock_id.in_(stock_ids))
            .order_by(PriceHistory.stock_id, PriceHistory.date)
        )
    ).all()

    # per-stock: list date (tăng dần) + map date→close để tra cứu O(1) / bisect.
    dates_by_stock: dict[int, list[date]] = {}
    close_by_stock: dict[int, dict[date, float]] = {}
    for sid, d, c in price_rows:
        dates_by_stock.setdefault(sid, []).append(d)
        close_by_stock.setdefault(sid, {})[d] = float(c)

    rows: list[dict] = []
    for sid, t in targets:
        dates = dates_by_stock.get(sid)
        if not dates:
            continue
        close_t = close_by_stock[sid].get(t)
        if close_t is None:  # không có giá ngày T (lạ — bỏ qua)
            continue
        # phiên kế tiếp = ngày > t nhỏ nhất
        i = bisect_right(dates, t)
        if i >= len(dates):  # chưa có phiên kế tiếp → chưa chấm được
            continue
        close_next = close_by_stock[sid][dates[i]]
        rows.append({"stock_id": sid, "date": t, "label": label_from_close(close_t, close_next)})

    return await upsert(
        session,
        ActualResult,
        rows,
        index_elements=["stock_id", "date"],
        update_cols=["label"],
    )
