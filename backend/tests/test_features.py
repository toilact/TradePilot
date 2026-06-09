"""Test feature builder: no-leakage (quan trọng nhất), indicator đúng, sentiment join, label ±1%."""

from datetime import date, datetime, timedelta

import pandas as pd

from features.builder import build_features, load_training_frame
from models.database import DailySentiment, PriceHistory, Stock


def _prices(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    d0 = datetime.fromisoformat(start)
    return pd.DataFrame(
        {"date": [d0 + timedelta(days=i) for i in range(len(closes))], "close": closes}
    )


# ── No-leakage (bất biến #1) ──────────────────────────────────────────────────


def test_no_leakage_feature_independent_of_future():
    """Feature hàng T KHÔNG đổi khi thay close của các ngày > T."""
    closes = [float(i) for i in range(1, 31)]
    base = build_features(_prices(closes), pd.DataFrame())

    altered = closes.copy()
    altered[25:] = [999.0, 998.0, 997.0, 996.0, 995.0]  # đổi tương lai (ngày > 24)
    changed = build_features(_prices(altered), pd.DataFrame())

    feat = ["ma7", "ma20", "rsi14", "macd", "macd_signal"]
    # Hàng T=20 (idx 20) chỉ phụ thuộc quá khứ→20 → phải y hệt dù tương lai đổi.
    pd.testing.assert_frame_equal(base.loc[:20, feat], changed.loc[:20, feat], check_dtype=False)


def test_last_row_label_is_null():
    """Hàng cuối thiếu close T+1 → label None (loại khỏi train)."""
    df = build_features(_prices([10.0, 11.0, 12.0]), pd.DataFrame())
    assert df["label"].iloc[-1] is None
    assert df["label"].iloc[0] is not None


# ── Indicator đúng (ground-truth tính tay) ────────────────────────────────────


def test_ma_values():
    df = build_features(_prices([1, 2, 3, 4, 5, 6, 7, 8]), pd.DataFrame())
    assert df["ma7"].iloc[6] == 4.0  # mean(1..7)
    assert df["ma7"].iloc[7] == 5.0  # mean(2..8)
    assert pd.isna(df["ma7"].iloc[5])  # warm-up < 7 phiên → NaN
    assert pd.isna(df["ma20"].iloc[7])  # chưa đủ 20 phiên


def test_macd_constant_series_is_zero():
    df = build_features(_prices([5.0] * 30), pd.DataFrame())
    assert abs(df["macd"].iloc[-1]) < 1e-9
    assert abs(df["macd_signal"].iloc[-1]) < 1e-9


def test_rsi_monotonic_up_is_100():
    df = build_features(_prices([float(i) for i in range(1, 25)]), pd.DataFrame())
    # chuỗi tăng đơn điệu → avg_loss=0 → RSI=100 (sau warm-up 14)
    assert df["rsi14"].iloc[20] == 100.0
    assert pd.isna(df["rsi14"].iloc[5])  # warm-up


# ── Sentiment join (ngày không tin → 0) ───────────────────────────────────────


def test_sentiment_join_fills_zero():
    prices = _prices([10.0, 11.0, 12.0], start="2026-01-01")
    sentiment = pd.DataFrame(
        {"date": [datetime(2026, 1, 2)], "sentiment_agg": [0.5], "news_count": [3]}
    )
    df = build_features(prices, sentiment)
    assert df["sentiment_agg"].tolist() == [0.0, 0.5, 0.0]  # 1/1 và 3/1 không tin → 0
    assert df["news_count"].tolist() == [0, 3, 0]


def test_no_sentiment_frame_all_zero():
    df = build_features(_prices([10.0, 11.0]), pd.DataFrame())
    assert df["sentiment_agg"].tolist() == [0.0, 0.0]
    assert df["news_count"].tolist() == [0, 0]


# ── Label khớp labeling ±1% ───────────────────────────────────────────────────


def test_label_matches_threshold():
    # 100→101.5 (+1.5% > 1% → tang); 101.5→101.0 (-0.49% → di_ngang); 101.0→? cuối None
    df = build_features(_prices([100.0, 101.5, 101.0]), pd.DataFrame())
    assert df["label"].iloc[0] == "tang"
    assert df["label"].iloc[1] == "di_ngang"
    assert df["label"].iloc[2] is None


def test_label_down():
    df = build_features(_prices([100.0, 98.0]), pd.DataFrame())  # -2% → giam
    assert df["label"].iloc[0] == "giam"


# ── load_training_frame (đọc DB) ──────────────────────────────────────────────


async def test_load_training_frame(session_factory, monkeypatch):
    # builder import SessionLocal nội bộ từ models.database → patch ở đó.
    import models.database as db

    monkeypatch.setattr(db, "SessionLocal", session_factory)

    async with session_factory() as s:
        stock = Stock(symbol="VCB", name="VCB", exchange="HOSE")
        s.add(stock)
        await s.flush()
        for i, close in enumerate([10.0, 11.0, 12.0, 13.0]):
            s.add(
                PriceHistory(
                    stock_id=stock.id,
                    date=date(2026, 1, 1) + timedelta(days=i),
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=100,
                )
            )
        s.add(
            DailySentiment(
                stock_id=stock.id, date=date(2026, 1, 2), sentiment_agg=0.5, news_count=2
            )
        )
        await s.commit()

    df = await load_training_frame("VCB")
    assert len(df) == 4
    assert list(df["date"].dt.date) == [date(2026, 1, 1) + timedelta(days=i) for i in range(4)]
    # sentiment join đúng: chỉ ngày 2/1 có tin
    assert df.loc[df["date"] == datetime(2026, 1, 2), "sentiment_agg"].iloc[0] == 0.5
    assert df.loc[df["date"] == datetime(2026, 1, 1), "news_count"].iloc[0] == 0


async def test_load_training_frame_unseeded(session_factory, monkeypatch):
    import models.database as db

    monkeypatch.setattr(db, "SessionLocal", session_factory)
    df = await load_training_frame("ZZZ")
    assert df.empty
