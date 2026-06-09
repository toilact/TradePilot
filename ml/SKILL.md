---
name: ml-train-and-export
description: Train PhoBERT sentiment và TFT model trên Kaggle rồi export artifacts cho backend. Dùng khi sửa feature engineering, fine-tune sentiment, train/backtest TFT, hoặc cập nhật model_version.
---

# Skill: Train & export models

## Quy trình sentiment (PhoBERT)
1. Auto-label ~500 câu tin bằng LLM → 3 lớp.
2. Người review nhanh mẫu LLM gán nhãn không chắc.
3. Fine-tune `vinai/phobert-base` trên Kaggle GPU.
4. Eval (F1 theo lớp), export `artifacts/phobert_sentiment/`.

## Quy trình TFT
1. Build dataset: join `price_history` + `daily_sentiment` theo (stock_id, date).
2. Feature: MA7/MA20, RSI, MACD, sentiment_agg, news_count; target = nhãn ±1% tại T+1.
3. `stock_id` làm static categorical (global model).
4. Train `pytorch-forecasting` TemporalFusionTransformer.
5. **Backtest walk-forward** 90 ngày gần nhất → report accuracy + confusion matrix.
6. Export checkpoint + tăng `model_version`.

## Bất biến — không được vi phạm
- Walk-forward only; không random split, không shuffle theo thời gian.
- Feature ngày T chỉ dùng dữ liệu ≤ 16:00 phiên T (không leak close T+1 vào input).
- Nhãn: Tăng >+1% / Giảm <-1% / Đi ngang [-1%,+1%].

## Bàn giao cho backend
- Backend load từ `artifacts/`, chỉ inference. Mọi thay đổi feature phải đồng bộ với `backend/services/sentiment.py` và feature builder.
- Khi đổi feature/model → bump `model_version` để truy vết predictions.

## Compute
- Train: Kaggle (T4/P100). Inference: localhost (MacBook M5) chạy được.
