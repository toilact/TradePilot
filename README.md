# TradePilot

[![CI](https://github.com/toilact/TradePilot/actions/workflows/ci.yml/badge.svg)](https://github.com/toilact/TradePilot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](backend/pyproject.toml)
[![Next.js](https://img.shields.io/badge/next.js-15-black)](frontend/package.json)
[![Model](https://img.shields.io/badge/model-LightGBM%20lgbm__v4-green)](ml/artifacts/lgbm_model/MODEL_CARD_lgbm_v4.md)

Hệ thống dự đoán cổ phiếu Việt Nam (VN30): với mỗi mã, dự đoán phiên **T+1** sẽ
**Tăng / Giảm / Đi ngang** — kèm **confidence gating**: chỉ phát tín hiệu khi model đủ
tự tin, còn lại trung thực hiển thị *"Không đủ tín hiệu"*.

Pipeline tự chạy **16:05 mỗi ngày giao dịch**: lấy giá → chấm điểm dự đoán hôm qua →
crawl tin → inference → gửi tổng kết Telegram. Độ chính xác được theo dõi công khai
trên trang `/accuracy`.

> ⚠️ **Đây không phải khuyến nghị đầu tư.** Dự án học tập / cá nhân (portfolio + học ML),
> xây dựng solo cùng agent AI.

---

## Bài toán & định nghĩa nhãn

Nhãn cho ngày T tính từ % thay đổi giá đóng cửa **T+1 so với T**:

| % thay đổi (close T+1 vs close T) | Nhãn |
|-----------------------------------|------|
| > +1% | **Tăng** (`tang`) |
| < −1% | **Giảm** (`giam`) |
| trong [−1%, +1%] | **Đi ngang** (`di_ngang`) |

Lớp "đi ngang" chiếm ~50% → baseline luôn-đoán-đi-ngang đã đạt ~0.50 accuracy.
Vì vậy tiêu chí đánh giá là **macro-F1** (cả 3 lớp phải sống) + **precision trên tập
dám đoán** (coverage × precision), *không phải* accuracy thô.

## Confidence gating — điểm khác biệt chính

Model xuất xác suất 3 lớp, được **calibrate bằng temperature scaling** (T = 0.6148, fit
trên validation). Chỉ khi xác suất cao nhất ≥ **threshold 0.60** mới phát tín hiệu:

| Threshold | Precision (test 2025) | Coverage |
|-----------|----------------------|----------|
| 0.40 | 0.503 | 81.7% |
| **0.60** (production) | **0.667** | **21.4%** |
| 0.70 | 0.736 | 10.9% |

Đánh đổi có chủ đích: *nói ít, đúng nhiều* — 21% số phiên có tín hiệu nhưng precision 0.667,
so với baseline ~0.50. Chi tiết quyết định: [ADR 0002](docs/adr/0002-confidence-gated-predictions.md).

## Hành trình model (trung thực)

| Version | Model | Kết quả | Ghi chú |
|---------|-------|---------|---------|
| `stub_v0` | MA7 vs MA20 | — | Stub để frontend chạy end-to-end sớm |
| `tft_v1`, `tft_v2` | Temporal Fusion Transformer (Kaggle GPU) | **Collapse** — macro-F1 0.22, đoán ~100% đi ngang | Loss đứng im; cả class-weight lẫn oversample đều không cứu. Giữ làm bằng chứng |
| `lgbm_v3` | LightGBM global | macro-F1 **0.4018** | Cùng feature/split — chứng minh tín hiệu CÓ, TFT không khai thác được |
| `lgbm_v4` (production) | LightGBM + 7 feature volatility/momentum + sector | macro-F1 **0.4234**, cả 3 lớp F1 > 0 | `vol_20`/`vol_5` thống trị feature importance — "biên độ trước, hướng sau" |

Bài học đắt giá nhất: **MI của index features trên panel lặp 30 mã/ngày là tín hiệu giả**
(0.194 → 0.001 khi đo đúng trên 1 mã) — đã loại toàn bộ index features sau EDA.
Chi tiết: [model card lgbm_v4](ml/artifacts/lgbm_model/MODEL_CARD_lgbm_v4.md).

## Kiến trúc

```
[CafeF + RSS]        [vnstock API]
      │                    │
      ▼                    ▼
  [Crawler]         [Price Fetcher]         ┌────────────────────────────────┐
      └─────────┬──────────┘                │ Daily pipeline 16:05 (launchd) │
                ▼                           │ prices → actual_results        │
     [Supabase PostgreSQL] ◄────────────────│ → news → sentiment             │
                │                           │ → inference → Telegram         │
      ┌─────────┴──────────┐                └────────────────────────────────┘
      ▼                    ▼
[Sentiment (stub→PhoBERT)] [LightGBM lgbm_v4 + temperature + gating]
      └─────────┬──────────┘
                ▼
        [predictions + actual_results]
                ▼
        [FastAPI read API]  — structlog JSON · Sentry
                ▼
        [Next.js frontend]  — 4 trạng thái: Tăng/Giảm/Đi ngang/Không đủ tín hiệu
```

**Phân vai bất di bất dịch:** backend là nguồn sự thật · `ml/` chỉ train + export artifact ·
frontend chỉ hiển thị. Feature dùng chung 1 builder (`backend/features/builder.py`) cho cả
train lẫn inference — **chống train/serve skew** bằng assert feature list khớp metrics.

## Chống data leakage (bất biến số 1)

- Feature ngày T chỉ dùng thông tin có **trước 16:00 phiên T**; nhãn là close T+1.
- Split **walk-forward theo thời gian** (train 2010–2024 / val / test) — cấm random split.
- Test tự động bắt leakage trong CI; mọi feature giá đều được phủ kiểm tra no-leakage.

## Tech stack

| Lớp | Công nghệ |
|-----|-----------|
| Frontend | Next.js 15 (App Router) · React 19 · TailwindCSS · TypeScript |
| Backend | FastAPI · SQLAlchemy 2.0 async · Alembic · APScheduler · Python 3.12 |
| Database | Supabase (PostgreSQL) |
| ML | LightGBM (train local, vài giây) · PhoBERT sentiment (kế hoạch, train Kaggle) |
| Dữ liệu | `vnstock` (OHLCV điều chỉnh) · crawler CafeF + RSS (lịch sự: robots.txt, delay, UA rõ) |
| Observability | structlog JSON · Sentry (backend + frontend) · alert Telegram |
| Tooling | `uv` + `ruff` · `eslint` + `prettier` · pytest (109 test) · GitHub Actions CI chặn merge |

## Cấu trúc thư mục

```
TradePilot/
├── backend/          # FastAPI — nguồn sự thật
│   ├── api/          #   routes: predictions, stocks, auth
│   ├── services/     #   price_fetcher, crawler, sentiment, inference (gating),
│   │                 #   scheduler (pipeline 16:00), notifier, actual_results, read_api
│   ├── features/     #   builder.py — feature dùng chung train + inference (chống skew)
│   ├── models/       #   SQLAlchemy 2.0 async (9 bảng)
│   ├── migrations/   #   Alembic (mọi đổi schema = 1 migration)
│   ├── scripts/      #   run_daily_pipeline, export_training_data, launchd plist
│   └── tests/        #   pytest — nhãn ±1%, leakage, gating, scheduler, notifier...
├── frontend/         # Next.js 15 — trang chủ, stock/[symbol], accuracy, watchlist
├── ml/               # train + export (KHÔNG serve)
│   ├── notebooks/    #   04_lgbm_training.py (tham số hoá version) + EDA
│   └── artifacts/    #   model card + metrics JSON (commit) · weights (gitignore)
└── docs/adr/         # Architecture Decision Records
```

## Chạy local

### Backend

```bash
cd backend
uv sync                          # deps mặc định (KHÔNG gồm lightgbm — xem ghi chú)
cp .env.example .env             # điền DATABASE_URL (Supabase), Sentry/Telegram tuỳ chọn
uv run alembic upgrade head
uv run uvicorn main:app --reload # http://localhost:8000/docs
```

```bash
uv run ruff check . && uv run pytest -q   # 109 test
```

> Dep ML nằm trong group riêng (`uv sync --group inference`) — CI/deploy mặc định không cài
> ([ADR 0002](docs/adr/0002-confidence-gated-predictions.md)).

### Pipeline hàng ngày (1 lệnh end-to-end)

```bash
cd backend
uv run --group inference python -m scripts.run_daily_pipeline
```

Tự động hoá: template launchd tại `backend/scripts/launchd/com.tradepilot.daily.plist`
(16:05 thứ 2–6, chạy bù sau sleep, log `~/Library/Logs/tradepilot-daily.log`).

### Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:3000 — cần backend ở localhost:8000
```

### Train lại model (local, vài giây)

```bash
cd backend && uv run python -m scripts.export_training_data   # → ml/data/training_panel.csv
cd ..
uv run --with lightgbm --with scikit-learn --with pandas --with scipy \
    python ml/notebooks/04_lgbm_training.py                   # → artifacts + metrics JSON
```

## Trạng thái & lộ trình

**Đã hoàn thành:** dữ liệu VN30 ~97k phiên · crawler tin tức · LightGBM `lgbm_v4` production
(macro-F1 0.4234, gating threshold 0.60) · API + frontend 4 trạng thái · accuracy tracking
(coverage + precision-on-actionable) · observability (structlog JSON, Sentry, Telegram) ·
pipeline tự động 16:05 (launchd) · CI 3 job chặn merge.

**Kế hoạch:** Redis cache (Upstash) → deploy (Vercel + Render) → PhoBERT sentiment thật
(hiện stub = 0) → auth + watchlist → drift monitor + rolling threshold.

**Giới hạn đã biết (đọc model card trước khi dùng):** holdout 2026 cho thấy độ bền
regime-shift còn yếu (macro-F1 0.396); sentiment chưa đóng góp tín hiệu; threshold production
cố định — sẽ chuyển rolling khi có drift monitor.

## Quy ước phát triển

- **Git:** branch riêng → PR → review + merge (CI chặn merge khi fail). KHÔNG commit thẳng `main`. Conventional Commits.
- **Secrets:** `.env` (gitignore) + `.env.example` mỗi module. Không hardcode/commit secret.
- **Schema:** mọi thay đổi qua Alembic migration, không SQL tay.
- **Test bắt buộc:** nhãn ±1%, leakage, gating, sentiment aggregation, parser crawler (HTML fixture).
- **Crawl lịch sự:** robots.txt, rate-limit, User-Agent rõ; chỉ lưu link + metadata, không tái xuất bản nội dung.

## License

Dự án học tập cá nhân — không có giấy phép phân phối.
