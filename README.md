# TradePilot

[![CI](https://github.com/toilact/TradePilot/actions/workflows/ci.yml/badge.svg)](https://github.com/toilact/TradePilot/actions/workflows/ci.yml)

Web dự đoán cổ phiếu Việt Nam: với mỗi mã, dự đoán phiên **T+1** sẽ **Tăng / Giảm / Đi ngang**,
dựa trên giá lịch sử (OHLCV) + sentiment tin tức tiếng Việt.

> ⚠️ **Đây không phải khuyến nghị đầu tư.** Dự án học tập / cá nhân (portfolio + học ML).

## Định nghĩa nhãn

Nhãn cho ngày T tính từ % thay đổi giá đóng cửa **T+1 so với T**:

| % thay đổi (close T+1 vs close T) | Nhãn |
|-----------------------------------|------|
| > +1%   | **Tăng** (`tang`) |
| < −1%   | **Giảm** (`giam`) |
| trong [−1%, +1%] | **Đi ngang** (`di_ngang`) |

Baseline thực tế ~50% (lớp "đi ngang" chiếm đa số), **không phải 33%**.

## Kiến trúc

```
[CafeF / RSS]      [vnstock API]
     │                  │
     ▼                  ▼
 [Crawler]        [Price Fetcher]
     └────────┬─────────┘
              ▼
      [PostgreSQL / Supabase]
              │
     ┌────────┴────────┐
     ▼                 ▼
[PhoBERT Sentiment]  [TFT Model]   ← train trên Kaggle GPU
     └────────┬────────┘
              ▼
       [Prediction Store]
              │
              ▼
       [FastAPI Backend]
              │
              ▼
       [Next.js Frontend] → Vercel
```

**Nguyên tắc:** Backend là nguồn sự thật; ml chỉ train + export; frontend chỉ hiển thị.

## Cấu trúc thư mục

```
TradePilot/
├── backend/    # FastAPI + pipeline + API
│   ├── api/         # routes: predictions, stocks (history/accuracy), auth
│   ├── services/    # price_fetcher, crawler, sentiment, labeling, read_api,
│   │                #   actual_results, scheduler, stock_seed
│   ├── features/    # builder.py — feature TFT (chống leakage), dùng chung train+inference
│   ├── models/      # SQLAlchemy 2.0 async (9 bảng)
│   ├── scripts/     # seed_predictions, fill_actual_results, export_training_data
│   ├── migrations/  # Alembic
│   └── tests/       # pytest (53 test)
├── frontend/   # Next.js 15 + TailwindCSS (App Router)
│   └── app/         # trang chủ, stock/[symbol], accuracy, watchlist
├── ml/         # notebook train Kaggle + artifacts
│   ├── notebooks/   # 03_tft_training.py + 03_tft_training_kaggle.py
│   └── artifacts/   # checkpoint + metrics (weights gitignore)
└── docs/       # ADR + quy ước agent
```

## Tech stack

| Lớp | Công nghệ |
|-----|-----------|
| Frontend | Next.js 15 · React 19 · TailwindCSS · TypeScript |
| Backend  | FastAPI · SQLAlchemy 2.0 async · asyncpg · APScheduler · Python 3.12 |
| Database | Supabase (PostgreSQL), migration bằng Alembic |
| ML       | PhoBERT (sentiment) · Temporal Fusion Transformer (`pytorch-forecasting`), train trên Kaggle |
| Dữ liệu  | `vnstock` (OHLCV) · crawler CafeF + RSS (tin tức) |
| Tooling  | `uv` + `ruff` (Python) · `eslint` + `prettier` (frontend) |

## Trạng thái

Walking skeleton đã thông end-to-end (data → feature → model → API → frontend).

**Đã có:**
- Pipeline giá: `vnstock` → `price_history` (30 mã VN30, ~96.8k phiên), upsert idempotent.
- Crawler CafeF + RSS → `news` + `news_stocks` (lịch sự: robots.txt + delay + UA).
- Feature builder chống data leakage (MA7/MA20, RSI14, MACD, sentiment) + test no-leakage.
- Notebook TFT global (30 mã, split walk-forward theo thời gian) — chạy được trên Kaggle.
- API thật đọc DB: `/api/predictions`, `/api/stocks/{symbol}/history`, `/api/accuracy`.
- Accuracy job: `fill_actual_results` chấm nhãn thực tế (close T vs close phiên kế tiếp).
- Frontend 4 trang đọc data thật từ API.

**Chưa có / đang dở (trung thực):**
- **TFT v1 bị collapse** — đoán 100% "đi ngang", test acc 0.479 < baseline 0.504. Giữ làm
  *baseline tham chiếu*, **chưa dùng để dự đoán thật**. Frontend đang dùng dự đoán **stub**
  (`stub_v0`, so MA7 vs MA20), không phải TFT.
- **PhoBERT chưa train** → `sentiment_agg` hiện toàn 0 (model thiếu tín hiệu phi-giá).
- Trang accuracy còn rỗng tới khi dự đoán có phiên kế tiếp để chấm.
- Chưa wire scheduler 16:00 · chưa CI · chưa deploy · auth/watchlist là khung (Phase 3).

## Bất biến (không vi phạm)

1. **Nhãn ±1%** thống nhất ở train, backtest, hiển thị.
2. **Chống data leakage:** feature ngày T chỉ dùng thông tin ≤ 16:00 phiên T; nhãn là close T+1.
   Backtest **walk-forward**, cấm random split.
3. 1 bài báo → nhiều mã qua `news_stocks`. Ngày không có tin → sentiment = 0.
4. Backend là nguồn sự thật; ml chỉ train + export; frontend chỉ hiển thị.
5. Mọi nơi hiển thị dự đoán phải có disclaimer "Đây không phải khuyến nghị đầu tư".

## Chạy local

### Backend (FastAPI)

```bash
cd backend
uv sync                                    # cài deps (cần Python 3.12)
cp .env.example .env                       # điền DATABASE_URL (Supabase), keys...
uv run alembic upgrade head                # tạo schema
uv run uvicorn main:app --reload           # http://localhost:8000/docs
```

Script tiện ích:
```bash
uv run python -m scripts.seed_predictions     # seed dự đoán stub (để frontend có data)
uv run python -m scripts.fill_actual_results  # chấm nhãn thực tế (accuracy)
```

Kiểm thử:
```bash
uv run ruff check . && uv run pytest -q        # 53 test
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev                                 # http://localhost:3000
```

Cần backend chạy ở `localhost:8000` (hoặc đặt `NEXT_PUBLIC_API_URL` trong `.env.local`).

### Train model (Kaggle)

1. Export panel: `cd backend && uv run python -m scripts.export_training_data` → `ml/data/training_panel.csv`.
2. Upload CSV lên Kaggle Dataset, mở `ml/notebooks/03_tft_training_kaggle.py` (8 cell copy-paste), bật GPU.
3. Download `tft_v1.ckpt` + `metrics_tft_v1.json` về `ml/artifacts/tft_model/`.

## Quy ước phát triển

- **Git:** branch riêng → PR → review + merge. KHÔNG commit thẳng `main`. Conventional Commits.
- **Secrets:** `.env` (gitignore) + `.env.example` mỗi module. Không hardcode/commit secret.
- **Test bắt buộc:** hàm gán nhãn ±1%, kiểm tra leakage, aggregation sentiment, parser crawler.

## License

Dự án học tập cá nhân — không có giấy phép phân phối.
