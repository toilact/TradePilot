"""PhoBERT sentiment inference + tổng hợp daily_sentiment.

Backend CHỈ inference — model train ở ml/ rồi export sang ml/artifacts/phobert_sentiment/.
Bất biến: ngày không có tin → sentiment_agg = 0.

TODO Phase 1.2:
  - load model từ ../ml/artifacts/phobert_sentiment/
  - chấm điểm news.sentiment_score
  - tổng hợp trung bình theo (stock, date) → daily_sentiment
"""

from __future__ import annotations

MODEL_PATH = "../ml/artifacts/phobert_sentiment"


def score_text(text: str) -> float:
    """Trả sentiment score [-1, 1] cho 1 đoạn tin. (chưa implement)"""
    raise NotImplementedError("Phase 1.2: load PhoBERT + inference")


async def build_daily_sentiment(symbol: str) -> int:
    """Tổng hợp daily_sentiment cho `symbol`. Trả số ngày đã ghi. (chưa implement)"""
    raise NotImplementedError("Phase 1.2: aggregate theo ngày, 0 nếu không có tin")
