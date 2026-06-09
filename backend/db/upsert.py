"""Helper upsert đa dialect (PostgreSQL + SQLite).

Cả hai dialect đều hỗ trợ `INSERT ... ON CONFLICT (...) DO UPDATE`.
Tách riêng để (a) tái dùng cho crawler/sentiment, (b) test được trên SQLite in-memory
dù production chạy PostgreSQL. Batch toàn bộ rows trong 1 statement (idempotent).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession


async def upsert(
    session: AsyncSession,
    model: Any,
    rows: Sequence[dict],
    index_elements: Sequence[str],
    update_cols: Sequence[str],
) -> int:
    """Upsert `rows` vào bảng của `model`.

    - `index_elements`: cột tạo nên unique constraint để phát hiện conflict.
    - `update_cols`: cột sẽ ghi đè khi đã tồn tại (vd OHLCV khi giá adjusted đổi).
    Trả về số dòng đầu vào (đã batch).
    """
    if not rows:
        return 0

    dialect = session.bind.dialect.name
    insert_fn = pg_insert if dialect == "postgresql" else sqlite_insert
    stmt = insert_fn(model).values(list(rows))
    stmt = stmt.on_conflict_do_update(
        index_elements=list(index_elements),
        set_={col: getattr(stmt.excluded, col) for col in update_cols},
    )
    await session.execute(stmt)
    await session.commit()
    return len(rows)
