"""Fixtures test — DB SQLite in-memory dùng chung 1 connection (StaticPool)."""

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from models.database import Base


@pytest_asyncio.fixture
async def session_factory():
    """async_sessionmaker trỏ tới SQLite in-memory, đã tạo sẵn 9 bảng."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()
