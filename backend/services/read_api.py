"""Đọc dữ liệu cho API (Phase 1.4). Backend là nguồn sự thật — trả JSON khớp frontend/lib/types.ts.

Tách khỏi route để test được. Query thẳng Supabase (chưa Redis — tối ưu sau).
Shape trả về dùng camelCase khớp frontend (changePct, modelVersion...).
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    ActualResult,
    DailySentiment,
    Prediction,
    PriceHistory,
    Stock,
)

# vnstock/TCBS đặt tên sàn "HSX"; frontend dùng "HOSE" (tên chính thức). Map về tên frontend.
_EXCHANGE_MAP = {"HSX": "HOSE"}


def _exchange(raw: str) -> str:
    return _EXCHANGE_MAP.get(raw, raw)


async def _latest_sentiment(session: AsyncSession, stock_id: int) -> float:
    """sentiment_agg phiên gần nhất của mã (0 nếu chưa có daily_sentiment)."""
    row = (
        await session.execute(
            select(DailySentiment.sentiment_agg)
            .where(DailySentiment.stock_id == stock_id)
            .order_by(DailySentiment.date.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return float(row) if row is not None else 0.0


async def _close_and_change(session: AsyncSession, stock_id: int) -> tuple[float, float]:
    """(close gần nhất, % thay đổi vs phiên trước). changePct=0 nếu chỉ có 1 phiên."""
    rows = (
        await session.execute(
            select(PriceHistory.close)
            .where(PriceHistory.stock_id == stock_id)
            .order_by(PriceHistory.date.desc())
            .limit(2)
        )
    ).scalars().all()
    if not rows:
        return 0.0, 0.0
    close = float(rows[0])
    if len(rows) < 2 or rows[1] == 0:
        return close, 0.0
    prev = float(rows[1])
    return close, round((close - prev) / prev * 100, 2)


async def _build_prediction(session: AsyncSession, stock: Stock) -> dict:
    """Ghép 1 mã → dict Prediction khớp frontend. label/confidence từ predictions mới nhất."""
    pred = (
        await session.execute(
            select(Prediction)
            .where(Prediction.stock_id == stock.id)
            .order_by(Prediction.prediction_date.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    close, change_pct = await _close_and_change(session, stock.id)
    sentiment = await _latest_sentiment(session, stock.id)
    return {
        "symbol": stock.symbol,
        "name": stock.name,
        "exchange": _exchange(stock.exchange),
        "sector": stock.sector or "",
        "close": close,
        "changePct": change_pct,
        "label": pred.label if pred else None,
        "confidence": float(pred.confidence) if pred else None,
        "sentiment": sentiment,
        "modelVersion": pred.model_version if pred else None,
    }


async def list_predictions(session: AsyncSession) -> list[dict]:
    """Tất cả mã active có ít nhất 1 prediction → list (sắp theo symbol)."""
    stocks = (
        await session.execute(
            select(Stock).where(Stock.is_active.is_(True)).order_by(Stock.symbol)
        )
    ).scalars().all()
    out = []
    for stock in stocks:
        item = await _build_prediction(session, stock)
        if item["label"] is not None:  # chỉ trả mã đã có dự đoán
            out.append(item)
    return out


async def get_prediction(session: AsyncSession, symbol: str) -> dict | None:
    """1 mã theo symbol (None nếu không tồn tại)."""
    stock = (
        await session.execute(select(Stock).where(Stock.symbol == symbol.upper()))
    ).scalar_one_or_none()
    if stock is None:
        return None
    return await _build_prediction(session, stock)


async def get_history(session: AsyncSession, symbol: str) -> list[dict]:
    """Timeline [{date, close, sentiment}] cho 1 mã (date tăng dần). sentiment=0 ngày không tin."""
    stock = (
        await session.execute(select(Stock).where(Stock.symbol == symbol.upper()))
    ).scalar_one_or_none()
    if stock is None:
        return []

    prices = (
        await session.execute(
            select(PriceHistory.date, PriceHistory.close)
            .where(PriceHistory.stock_id == stock.id)
            .order_by(PriceHistory.date)
        )
    ).all()
    sent_rows = (
        await session.execute(
            select(DailySentiment.date, DailySentiment.sentiment_agg).where(
                DailySentiment.stock_id == stock.id
            )
        )
    ).all()
    sent_by_date: dict[date, float] = {d: float(s) for d, s in sent_rows}

    return [
        {
            "date": d.isoformat(),
            "close": float(c),
            "sentiment": sent_by_date.get(d, 0.0),
        }
        for d, c in prices
    ]


async def get_accuracy(session: AsyncSession) -> dict:
    """So predictions vs actual_results → AccuracySummary khớp frontend.

    Stub-aware: nếu chưa có actual_results (chưa biết close T+1 thực tế), trả 0 + series rỗng.
    """
    from collections import defaultdict

    preds = (
        await session.execute(
            select(
                Prediction.stock_id,
                Prediction.target_date,
                Prediction.label,
                Prediction.model_version,
            )
        )
    ).all()
    actuals = (
        await session.execute(
            select(ActualResult.stock_id, ActualResult.date, ActualResult.label)
        )
    ).all()
    actual_map = {(sid, d): lbl for sid, d, lbl in actuals}

    matched: list[tuple[date, bool, str]] = []  # (date, correct, predicted_label)
    model_version = None
    for sid, tdate, plabel, mver in preds:
        model_version = mver
        actual = actual_map.get((sid, tdate))
        if actual is None:
            continue
        matched.append((tdate, actual == plabel, plabel))

    if not matched:
        return {
            "overall": 0.0,
            "last30": 0.0,
            "byLabel": {"tang": 0.0, "giam": 0.0, "di_ngang": 0.0},
            "series": [],
            "modelVersion": model_version or "n/a",
            "detail": "chưa có actual_results để chấm — cần biết close T+1 thực tế",
        }

    overall = sum(c for _, c, _ in matched) / len(matched)

    # by label
    by_label: dict[str, list[bool]] = defaultdict(list)
    for _, correct, plabel in matched:
        by_label[plabel].append(correct)
    by_label_acc = {
        lbl: round(sum(v) / len(v), 4) if v else 0.0
        for lbl, v in (("tang", by_label["tang"]), ("giam", by_label["giam"]),
                       ("di_ngang", by_label["di_ngang"]))
    }

    # rolling accuracy theo ngày (series)
    per_day: dict[date, list[bool]] = defaultdict(list)
    for d, correct, _ in matched:
        per_day[d].append(correct)
    series = [
        {"date": d.isoformat(), "accuracy": round(sum(v) / len(v), 4)}
        for d, v in sorted(per_day.items())
    ]
    last30 = (
        round(sum(c for _, c, _ in matched[-30:]) / len(matched[-30:]), 4)
        if matched
        else 0.0
    )

    return {
        "overall": round(overall, 4),
        "last30": last30,
        "byLabel": by_label_acc,
        "series": series,
        "modelVersion": model_version or "n/a",
    }
