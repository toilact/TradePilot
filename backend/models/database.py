"""Kết nối Supabase (PostgreSQL) + định nghĩa bảng bằng SQLAlchemy 2.0 async.

Schema bám theo PLAN.md. Bất biến: 1 bài báo → nhiều mã qua news_stocks;
daily_sentiment là input cho TFT (0 nếu ngày không có tin).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import settings

engine = create_async_engine(settings.database_url or "sqlite+aiosqlite:///:memory:", echo=False)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# Postgres: BIGINT identity (autoincrement). SQLite: INTEGER PK để alias rowid (autoincrement).
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Stock(Base):
    __tablename__ = "stocks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    exchange: Mapped[str] = mapped_column(String(10))  # HOSE / HNX / UPCOM
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class PriceHistory(Base):
    __tablename__ = "price_history"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_price_stock_date"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)


class News(Base):
    __tablename__ = "news"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text, unique=True)
    source: Mapped[str] = mapped_column(String(50))  # cafef / fireant
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class NewsStock(Base):
    """Quan hệ nhiều-nhiều: 1 bài báo có thể nhắc nhiều mã."""

    __tablename__ = "news_stocks"
    __table_args__ = (UniqueConstraint("news_id", "stock_id", name="uq_news_stock"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    news_id: Mapped[int] = mapped_column(ForeignKey("news.id"), index=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)


class DailySentiment(Base):
    """Sentiment tổng hợp theo ngày/mã — input cho TFT. 0 nếu ngày không có tin."""

    __tablename__ = "daily_sentiment"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_sentiment_stock_date"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    sentiment_agg: Mapped[float] = mapped_column(Float, default=0.0)
    news_count: Mapped[int] = mapped_column(Integer, default=0)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "stock_id", "prediction_date", "model_version", name="uq_pred_stock_date_ver"
        ),
    )
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    prediction_date: Mapped[date] = mapped_column(Date, index=True)  # ngày T
    target_date: Mapped[date] = mapped_column(Date)  # T+1
    label: Mapped[str] = mapped_column(String(10))  # tang / giam / di_ngang
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(20))


class ActualResult(Base):
    """Nhãn thực tế (fill sau khi biết close T+1) — để tính accuracy."""

    __tablename__ = "actual_results"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_actual_stock_date"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    label: Mapped[str] = mapped_column(String(10))


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider: Mapped[str] = mapped_column(String(20))  # email / google
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Watchlist(Base):
    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("user_id", "stock_id", name="uq_watchlist_user_stock"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — mở session DB cho mỗi request."""
    async with SessionLocal() as session:
        yield session
