# ML — Sentiment + TFT

Train trên Kaggle GPU, export artifacts cho backend inference. **Không** chạy training trong backend.

## Notebooks
- `notebooks/01_data_exploration.ipynb` — khảo sát giá + tin
- `notebooks/02_sentiment_finetune.ipynb` — fine-tune PhoBERT
- `notebooks/03_tft_training.ipynb` — train TFT

## Artifacts (output cho backend)
- `artifacts/phobert_sentiment/` — weights model sentiment
- `artifacts/tft_model/` — TFT checkpoint + `model_version`

## Sentiment model
- Base: `vinai/phobert-base`, fine-tune 3 lớp (tích cực/tiêu cực/trung lập).
- Data: ~500 câu **auto-label bằng LLM** (GPT/Gemini), người review mẫu sai.
- Lib: `transformers` + `torch`.

## TFT model
- **1 model global** cho toàn bộ mã; `stock_id` là static categorical (Phase 1 chỉ VCB nhưng giữ kiến trúc này).
- Lib: `pytorch-forecasting`.
- Features: MA7/MA20, RSI, MACD, `sentiment_agg`, `news_count`.
- Target: nhãn 3 lớp ±1% tại T+1.

## Quy tắc bắt buộc
- **Backtest walk-forward** (train quá khứ → test tương lai). CẤM random split — sẽ leak tương lai.
- Feature ngày T chỉ từ thông tin có trước 16:00 phiên T.
- Ngày không tin → sentiment feature = 0.
- Accuracy mục tiêu > 50% (baseline ngẫu nhiên 33%).

## Workflow (xem root CLAUDE.md cho quy ước chung)
- Tooling: `uv` + `ruff`. Notebook chạy Kaggle; code dùng chung (feature builder) để chung repo, test bằng `pytest`.
- Secrets: `.env` + `.env.example` (OpenAI/Gemini key cho auto-label, Supabase để đọc data train).
- Test bắt buộc: hàm gán nhãn ±1%, builder feature không leak (assert không dùng close T+1 trong input), aggregation sentiment.
- Model governance: retrain hàng tháng HOẶC khi rolling accuracy 30 ngày tụt dưới ngưỡng; bump `model_version` mỗi lần; lưu accuracy theo version.
- Đồng bộ: đổi feature/định nghĩa sentiment phải cập nhật cả `backend/services/sentiment.py` + feature builder backend.
