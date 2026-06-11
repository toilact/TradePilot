"""Feature builder dùng chung train (ml/) + inference (backend). Backend là nguồn sự thật.

Biến price_history (OHLCV) + daily_sentiment → DataFrame feature + nhãn cho LightGBM.

BẤT BIẾN CHỐNG LEAKAGE (quan trọng nhất):
  - Feature ngày T chỉ dùng thông tin ≤ T. Indicator (MA/RSI/MACD) và return/volatility
    (pct_change/rolling) nhìn quá khứ→T, KHÔNG shift âm, KHÔNG center. → giá trị feature
    hàng T độc lập với giá các ngày > T.
  - LABEL ngày T = label_from_close(close[T], close[T+1]) — chỉ NHÃN dùng tương lai, tách khỏi
    feature. Hàng cuối (thiếu T+1) → label = None, caller train phải loại.
  - Ngày không tin → sentiment_agg=0, news_count=0.
  - Trả frame theo date TĂNG DẦN (walk-forward); KHÔNG shuffle.

FEATURE_COLS = toàn bộ cột builder xuất ra (superset, gồm cả ma7/ma20 cho stub_v0).
LGBM_V4_FEATURES = feature list MODEL lgbm_v4 (chốt grill 2026-06-11, xem notebook 05 cell cuối):
  bỏ ma7/ma20 (redundant với close — thay bằng dist_ma20/ma_ratio), bỏ sentiment_agg/news_count
  (stub toàn 0, M8 thêm lại), bỏ index features (MI giả do lặp 30 mã/ngày). Train + inference
  PHẢI import tuple này — không tự chế danh sách (chống train/serve skew).

Caller train: dropna các cột feature (warm-up indicator đầu chuỗi là NaN) + loại hàng label None.
"""

from __future__ import annotations

import pandas as pd

from services.labeling import label_from_close

FEATURE_COLS = (
    "ma7",
    "ma20",
    "rsi14",
    "macd",
    "macd_signal",
    "ret_1d",
    "ret_5d",
    "abs_ret_1d",
    "vol_5",
    "vol_20",
    "dist_ma20",
    "ma_ratio",
    "sentiment_agg",
    "news_count",
    "sector",
)

LGBM_V4_FEATURES = (
    "rsi14",
    "macd",
    "macd_signal",
    "ret_1d",
    "ret_5d",
    "abs_ret_1d",
    "vol_5",
    "vol_20",
    "dist_ma20",
    "ma_ratio",
    "sector",  # static categorical
)


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


def build_features(
    prices: pd.DataFrame, sentiment: pd.DataFrame, sector: str | None = None
) -> pd.DataFrame:
    """OHLCV 1 mã + daily_sentiment → frame feature + label (theo date tăng dần).

    `prices`: cột date, close (tối thiểu). `sentiment`: cột date, sentiment_agg, news_count.
    `sector`: ngành tĩnh của mã (categorical, lặp mọi hàng; None nếu chưa seed).
    Trả cột: date, close, *FEATURE_COLS, label.
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

    # --- Return / volatility (v4 — chỉ nhìn quá khứ→T, fill_method=None: NaN không pad) ---
    df["ret_1d"] = df["close"].pct_change(fill_method=None)
    df["ret_5d"] = df["close"].pct_change(5, fill_method=None)
    df["abs_ret_1d"] = df["ret_1d"].abs()
    df["vol_5"] = df["ret_1d"].rolling(5).std()
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["dist_ma20"] = (df["close"] - df["ma20"]) / df["ma20"]
    df["ma_ratio"] = df["ma7"] / df["ma20"]

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

    # --- Sector (static categorical — không đổi theo ngày, không thể leak) ---
    df["sector"] = sector

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
        stock = (
            await session.execute(select(Stock.id, Stock.sector).where(Stock.symbol == symbol))
        ).one_or_none()
        if stock is None:
            return pd.DataFrame(columns=["date", "close", *FEATURE_COLS, "label"])
        stock_id, sector = stock

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
    return build_features(prices, sentiment, sector=sector)
