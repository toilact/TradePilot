"""Chạy tay: fill actual_results (nhãn thực tế) cho các prediction đã có close phiên kế tiếp.

Chạy: cd backend && uv run python -m scripts.fill_actual_results

Kỳ vọng khi predictions mới chỉ cho ngày gần nhất: 0 hoặc rất ít (close T+1 chưa có) —
đúng bản chất walk-forward. Lần sau khi đã có close phiên kế tiếp sẽ chấm được.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


async def main() -> None:
    from sqlalchemy import func, select

    from models.database import Prediction, SessionLocal
    from services.actual_results import fill_actual_results

    async with SessionLocal() as session:
        distinct_sub = (
            select(Prediction.stock_id, Prediction.prediction_date).distinct().subquery()
        )
        total_pred = (
            await session.execute(select(func.count()).select_from(distinct_sub))
        ).scalar_one()
        n = await fill_actual_results(session)

    print(f"Đã ghi/cập nhật {n} actual_results.")
    print(f"(Tổng {total_pred} cặp (mã, ngày) có prediction; "
          "phần chưa chấm = chưa có phiên kế tiếp.)")


if __name__ == "__main__":
    asyncio.run(main())
