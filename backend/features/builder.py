"""Feature builder dùng chung train (ml/) + inference (backend). Backend là nguồn sự thật.

Biến price_history (OHLCV) + daily_sentiment → DataFrame feature + nhãn cho TFT.

BẤT BIẾN CHỐNG LEAKAGE (quan trọng nhất):
  - Feature ngày T chỉ dùng thông tin ≤ T. Indicator (MA/RSI/MACD) rolling/ewm nhìn quá khứ→T,
    KHÔNG shift âm, KHÔNG center. → giá trị feature hàng T độc lập với giá các ngày > T.
  - LABEL ngày T = label_from_close(close[T], close[T+1]) — chỉ NHÃN dùng tương lai, tách khỏi
    feature. Hàng cuối (thiếu T+1) → label = None, caller train phải loại.
  - Ngày không tin → sentiment_agg=0, news_count=0.
  - Trả frame theo date TĂNG DẦN (walk-forward); KHÔNG shuffle.

Caller train: dropna các cột feature (warm-up indicator đầu chuỗi là NaN) + loại hàng label None.
"""

from __future__ import annotations

import pandas as pd

from services.labeling import label_from_close

FEATURE_COLS = ("ma7", "ma20", "rsi14", "macd", "macd_signal", "sentiment_agg", "news_count")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder. Chỉ dùng quá khứ→hiện tại (ewm Wilder = rolling nhân quả)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing = ewm với alpha=1/period (min_periods=period để warm-up ra NaN)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss=0 → rs=inf → rsi=100 (toàn tăng). Giữ NaN ở warm-up.
    return rsi.where(avg_loss != 0, 100.0).where(avg_gain.notna(), other=pd.NA)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """MACD = EMA12 - EMA26; signal = EMA9 của MACD. EMA nhân quả (không nhìn tương lai)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def build_features(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 1 mã + daily_sentiment → frame feature + label (theo date tăng dần).

    `prices`: cột date, close (tối thiểu). `sentiment`: cột date, sentiment_agg, news_count.
    Trả cột: date, close, ma7, ma20, rsi14, macd, macd_signal, sentiment_agg, news_count, label.
    """
    if prices.empty:
        return pd.DataFrame(columns=["date", "close", *FEATURE_COLS, "label"])

    df = prices[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)  # walk-forward order

    # --- Indicator (chỉ nhìn quá khứ→T) ---
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["rsi14"] = _rsi(df["close"], 14)
    df["macd"], df["macd_signal"] = _macd(df["close"])

    # --- Sentiment join (ngày thiếu → 0) ---
    if sentiment is not None and not sentiment.empty:
        s = sentiment[["date", "sentiment_agg", "news_count"]].copy()
        s["date"] = pd.to_datetime(s["date"])
        df = df.merge(s, on="date", how="left")
    else:
        df["sentiment_agg"] = 0.0
        df["news_count"] = 0
    df["sentiment_agg"] = df["sentiment_agg"].fillna(0.0)
    df["news_count"] = df["news_count"].fillna(0).astype(int)

    # --- Label: close T+1 vs close T (CHỈ nhãn dùng tương lai) ---
    next_close = df["close"].shift(-1)
    df["label"] = [
        label_from_close(c, n) if pd.notna(n) else None
        for c, n in zip(df["close"], next_close, strict=True)
    ]

    return df[["date", "close", *FEATURE_COLS, "label"]]


async def load_training_frame(symbol: str) -> pd.DataFrame:
    """Đọc DB cho `symbol` → build_features. I/O wrapper mỏng (import nội bộ tránh vòng lặp)."""
    from sqlalchemy import select

    from models.database import DailySentiment, PriceHistory, SessionLocal, Stock

    symbol = symbol.upper()
    async with SessionLocal() as session:
        stock_id = (
            await session.execute(select(Stock.id).where(Stock.symbol == symbol))
        ).scalar_one_or_none()
        if stock_id is None:
            return pd.DataFrame(columns=["date", "close", *FEATURE_COLS, "label"])

        price_rows = (
            await session.execute(
                select(PriceHistory.date, PriceHistory.close)
                .where(PriceHistory.stock_id == stock_id)
                .order_by(PriceHistory.date)
            )
        ).all()
        sent_rows = (
            await session.execute(
                select(
                    DailySentiment.date, DailySentiment.sentiment_agg, DailySentiment.news_count
                ).where(DailySentiment.stock_id == stock_id)
            )
        ).all()

    prices = pd.DataFrame(price_rows, columns=["date", "close"])
    sentiment = pd.DataFrame(sent_rows, columns=["date", "sentiment_agg", "news_count"])
    return build_features(prices, sentiment)
