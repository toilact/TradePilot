"""Đọc dữ liệu cho API (Phase 1.4). Backend là nguồn sự thật — trả JSON khớp frontend/lib/types.ts.

Tách khỏi route để test được. Query thẳng Supabase (chưa Redis — tối ưu sau).
Shape trả về dùng camelCase khớp frontend (changePct, modelVersion...).
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import (
    ActualResult,
    DailySentiment,
    News,
    NewsStock,
    Prediction,
    PriceHistory,
    Stock,
)

# vnstock/TCBS đặt tên sàn "HSX"; frontend dùng "HOSE" (tên chính thức). Map về tên frontend.
_EXCHANGE_MAP = {"HSX": "HOSE"}

# Confidence-gating (M3 — ADR 0002): trạng thái hiển thị khi model không đủ tự tin.
# Backend quyết định qua is_actionable (ghi lúc inference) — frontend KHÔNG tự suy từ threshold.
DISPLAY_NO_SIGNAL = "khong_du_tin_hieu"
_STUB_VERSION = "stub_v0"


def _pred_order():
    """Thứ tự chọn prediction hiển thị: mới nhất trước; trùng ngày → ưu tiên model thật
    hơn stub_v0; còn trùng nữa → bản ghi mới nhất (id lớn)."""
    return (
        Prediction.prediction_date.desc(),
        (Prediction.model_version != _STUB_VERSION).desc(),
        Prediction.id.desc(),
    )


def _gating_fields(pred) -> dict:
    """Field dự đoán + gating từ 1 prediction (ORM object hoặc Row cùng tên cột).

    `display` = label khi actionable, ngược lại DISPLAY_NO_SIGNAL — UI render theo field
    này. Vẫn trả label/probs đầy đủ để trang chi tiết minh bạch 3 prob bar.
    """
    actionable = bool(pred.is_actionable)
    return {
        "label": pred.label,
        "confidence": float(pred.confidence),
        "probTang": float(pred.prob_tang) if pred.prob_tang is not None else None,
        "probGiam": float(pred.prob_giam) if pred.prob_giam is not None else None,
        "probDiNgang": float(pred.prob_di_ngang) if pred.prob_di_ngang is not None else None,
        "isActionable": actionable,
        "threshold": float(pred.threshold) if pred.threshold is not None else None,
        "display": pred.label if actionable else DISPLAY_NO_SIGNAL,
        "modelVersion": pred.model_version,
    }


# Mã tồn tại nhưng CHƯA có prediction (mới thêm, chưa qua inference) — display vẫn phải là
# trạng thái hợp lệ cho UI (badge render theo display, không chấp nhận None).
_EMPTY_GATING = {
    "label": None,
    "confidence": None,
    "probTang": None,
    "probGiam": None,
    "probDiNgang": None,
    "isActionable": False,
    "threshold": None,
    "display": DISPLAY_NO_SIGNAL,
    "modelVersion": None,
}


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


def _close(closes_desc: list[float]) -> float:
    """close gần nhất từ list giá sắp GIẢM dần theo ngày (rỗng → 0)."""
    return closes_desc[0] if closes_desc else 0.0


def _change_pct(closes_desc: list[float]) -> float:
    """% thay đổi phiên gần nhất vs phiên trước (0 nếu <2 phiên hoặc prev=0)."""
    if len(closes_desc) < 2 or closes_desc[1] == 0:
        return 0.0
    close, prev = closes_desc[0], closes_desc[1]
    return round((close - prev) / prev * 100, 2)


async def _close_and_change(session: AsyncSession, stock_id: int) -> tuple[float, float]:
    """(close gần nhất, % thay đổi vs phiên trước). changePct=0 nếu chỉ có 1 phiên."""
    closes = [
        float(c)
        for c in (
            await session.execute(
                select(PriceHistory.close)
                .where(PriceHistory.stock_id == stock_id)
                .order_by(PriceHistory.date.desc())
                .limit(2)
            )
        )
        .scalars()
        .all()
    ]
    return _close(closes), _change_pct(closes)


async def _build_prediction(session: AsyncSession, stock: Stock) -> dict:
    """Ghép 1 mã → dict Prediction khớp frontend. label/confidence từ predictions mới nhất."""
    pred = (
        await session.execute(
            select(Prediction)
            .where(Prediction.stock_id == stock.id)
            .order_by(*_pred_order())
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
        "sentiment": sentiment,
        **(_gating_fields(pred) if pred else _EMPTY_GATING),
    }


async def list_predictions(session: AsyncSession) -> list[dict]:
    """Tất cả mã active có ít nhất 1 prediction → list (sắp theo symbol).

    Gom thành 3 query (KHÔNG N+1): latest prediction / 2 giá gần nhất / latest sentiment
    cho mọi mã trong 1 lượt, dùng window function row_number().
    """
    stocks = (
        (
            await session.execute(
                select(Stock).where(Stock.is_active.is_(True)).order_by(Stock.symbol)
            )
        )
        .scalars()
        .all()
    )
    if not stocks:
        return []
    stock_ids = [s.id for s in stocks]

    # --- 1 query: prediction hiển thị mỗi mã (mới nhất, ưu tiên non-stub) ---
    pred_rn = (
        func.row_number().over(partition_by=Prediction.stock_id, order_by=_pred_order()).label("rn")
    )
    pred_sub = (
        select(
            Prediction.stock_id,
            Prediction.label,
            Prediction.confidence,
            Prediction.model_version,
            Prediction.prob_tang,
            Prediction.prob_giam,
            Prediction.prob_di_ngang,
            Prediction.is_actionable,
            Prediction.threshold,
            pred_rn,
        )
        .where(Prediction.stock_id.in_(stock_ids))
        .subquery()
    )
    pred_rows = (await session.execute(select(pred_sub).where(pred_sub.c.rn == 1))).all()
    pred_by_stock = {r.stock_id: r for r in pred_rows}

    # --- 1 query: 2 giá gần nhất mỗi mã (để tính close + changePct) ---
    price_rn = (
        func.row_number()
        .over(partition_by=PriceHistory.stock_id, order_by=PriceHistory.date.desc())
        .label("rn")
    )
    price_sub = (
        select(PriceHistory.stock_id, PriceHistory.close, price_rn)
        .where(PriceHistory.stock_id.in_(stock_ids))
        .subquery()
    )
    price_rows = (
        await session.execute(
            select(price_sub)
            .where(price_sub.c.rn <= 2)
            .order_by(price_sub.c.stock_id, price_sub.c.rn)
        )
    ).all()
    closes_by_stock: dict[int, list[float]] = {}
    for r in price_rows:
        closes_by_stock.setdefault(r.stock_id, []).append(float(r.close))

    # --- 1 query: sentiment mới nhất mỗi mã ---
    sent_rn = (
        func.row_number()
        .over(partition_by=DailySentiment.stock_id, order_by=DailySentiment.date.desc())
        .label("rn")
    )
    sent_sub = (
        select(DailySentiment.stock_id, DailySentiment.sentiment_agg, sent_rn)
        .where(DailySentiment.stock_id.in_(stock_ids))
        .subquery()
    )
    sent_rows = (await session.execute(select(sent_sub).where(sent_sub.c.rn == 1))).all()
    sent_by_stock = {r.stock_id: float(r.sentiment_agg) for r in sent_rows}

    out = []
    for stock in stocks:
        pred = pred_by_stock.get(stock.id)
        if pred is None:  # chỉ trả mã đã có dự đoán
            continue
        out.append(
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "exchange": _exchange(stock.exchange),
                "sector": stock.sector or "",
                "close": _close(closes_by_stock.get(stock.id, [])),
                "changePct": _change_pct(closes_by_stock.get(stock.id, [])),
                "sentiment": sent_by_stock.get(stock.id, 0.0),
                **_gating_fields(pred),
            }
        )
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


async def get_news(session: AsyncSession, symbol: str, limit: int = 10) -> list[dict] | None:
    """Tin mới nhất của 1 mã (join news–news_stocks, published_at desc) khớp NewsItem frontend.

    None nếu mã không tồn tại (route trả 404 — phân biệt với mã có nhưng chưa có tin → []).
    `sentiment_score` NULL (stub chưa chấm — M8) → 0.0, nhất quán quy ước "không tin → 0".
    """
    stock = (
        await session.execute(select(Stock).where(Stock.symbol == symbol.upper()))
    ).scalar_one_or_none()
    if stock is None:
        return None
    rows = (
        await session.execute(
            select(News)
            .join(NewsStock, NewsStock.news_id == News.id)
            .where(NewsStock.stock_id == stock.id)
            .order_by(News.published_at.desc())
            .limit(limit)
        )
    ).scalars()
    return [
        {
            "title": n.title,
            "source": n.source,
            "publishedAt": n.published_at.isoformat(),
            "sentiment": float(n.sentiment_score) if n.sentiment_score is not None else 0.0,
            "url": n.url,
        }
        for n in rows
    ]


def _rolling_by_version(preds, actual_map: dict, window: int = 30) -> list[dict]:
    """Accuracy TRƯỢT `window` phiên cho từng model version (M10 — chart drift theo version).

    Mỗi điểm = accuracy gộp (sample-weighted) trên `window` phiên gần nhất tính tới phiên đó.
    Version production xếp trước stub_v0 để frontend tô gold cho đường chính.
    """
    from collections import defaultdict

    per_session: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in preds:
        actual = actual_map.get((r.stock_id, r.prediction_date))
        if actual is None:
            continue
        per_session[r.model_version][r.prediction_date].append(actual == r.label)

    out = []
    for ver, sess_map in per_session.items():
        counts = [(d, sum(sess_map[d]), len(sess_map[d])) for d in sorted(sess_map)]
        points = []
        for i in range(len(counts)):
            lo = max(0, i - window + 1)
            corr = sum(c for _, c, _ in counts[lo : i + 1])
            tot = sum(t for _, _, t in counts[lo : i + 1])
            points.append({"date": counts[i][0].isoformat(), "accuracy": round(corr / tot, 4)})
        out.append({"version": ver, "points": points})
    out.sort(key=lambda v: (v["version"] == _STUB_VERSION, v["version"]))
    return out


async def get_accuracy(session: AsyncSession) -> dict:
    """So predictions vs actual_results → AccuracySummary khớp frontend.

    CONTRACT khớp: theo (stock_id, prediction_date) — tức `ActualResult.date` mang NGÀY T
    (prediction_date) của dự đoán, là nhãn THỰC TẾ của dự đoán ngày T. KHÔNG khớp theo
    `target_date` (T+1) vì target_date tính sẵn có thể lệch khi nghỉ lễ; prediction_date
    luôn là phiên giao dịch thật đã biết → join ổn định. target_date chỉ để hiển thị.

    Stub-aware: nếu chưa có actual_results, trả 0 + series rỗng.
    """
    from collections import defaultdict

    preds = (
        await session.execute(
            select(
                Prediction.stock_id,
                Prediction.prediction_date,
                Prediction.label,
                Prediction.model_version,
                Prediction.is_actionable,
            )
        )
    ).all()
    actuals = (
        await session.execute(select(ActualResult.stock_id, ActualResult.date, ActualResult.label))
    ).all()
    actual_map = {(sid, d): lbl for sid, d, lbl in actuals}

    # Version hiện hành = version của prediction mới nhất (trùng ngày → ưu tiên non-stub).
    # Coverage/Precision gating (M3) chấm theo ĐÚNG version này — trộn version sẽ vô nghĩa.
    current_version = None
    if preds:
        current_version = max(
            preds, key=lambda r: (r.prediction_date, r.model_version != _STUB_VERSION)
        ).model_version
    cur_preds = [r for r in preds if r.model_version == current_version]
    coverage = (
        round(sum(bool(r.is_actionable) for r in cur_preds) / len(cur_preds), 4)
        if cur_preds
        else 0.0
    )
    actionable_correct = [
        actual_map[(r.stock_id, r.prediction_date)] == r.label
        for r in cur_preds
        if r.is_actionable and (r.stock_id, r.prediction_date) in actual_map
    ]
    precision_actionable = (
        round(sum(actionable_correct) / len(actionable_correct), 4)
        if actionable_correct
        else None  # chưa có actual nào trên tập dám đoán → chưa chấm được
    )

    # overall/byLabel/series giữ semantics cũ: chấm trên MỌI version (lịch sử đầy đủ);
    # chỉ coverage/precision ở trên là theo version hiện hành.
    matched: list[tuple[date, bool, str]] = []  # (prediction_date, correct, predicted_label)
    for r in preds:
        actual = actual_map.get((r.stock_id, r.prediction_date))
        if actual is None:
            continue
        matched.append((r.prediction_date, actual == r.label, r.label))

    if not matched:
        return {
            "overall": 0.0,
            "last30": 0.0,
            "byLabel": {"tang": 0.0, "giam": 0.0, "di_ngang": 0.0},
            "series": [],
            "rollingByVersion": [],
            "modelVersion": current_version or "n/a",
            "coverage": coverage,
            "precisionActionable": precision_actionable,
            "detail": "chưa có actual_results để chấm — cần biết close T+1 thực tế",
        }

    overall = sum(c for _, c, _ in matched) / len(matched)

    # by label
    by_label: dict[str, list[bool]] = defaultdict(list)
    for _, correct, plabel in matched:
        by_label[plabel].append(correct)
    by_label_acc = {
        lbl: round(sum(v) / len(v), 4) if v else 0.0
        for lbl, v in (
            ("tang", by_label["tang"]),
            ("giam", by_label["giam"]),
            ("di_ngang", by_label["di_ngang"]),
        )
    }

    # rolling accuracy theo ngày (series)
    per_day: dict[date, list[bool]] = defaultdict(list)
    for d, correct, _ in matched:
        per_day[d].append(correct)
    series = [
        {"date": d.isoformat(), "accuracy": round(sum(v) / len(v), 4)}
        for d, v in sorted(per_day.items())
    ]
    last30 = round(sum(c for _, c, _ in matched[-30:]) / len(matched[-30:]), 4) if matched else 0.0

    return {
        "overall": round(overall, 4),
        "last30": last30,
        "byLabel": by_label_acc,
        "series": series,
        "rollingByVersion": _rolling_by_version(preds, actual_map),
        "modelVersion": current_version or "n/a",
        "coverage": coverage,
        "precisionActionable": precision_actionable,
    }
