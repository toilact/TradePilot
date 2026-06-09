# TradePilot — Kế hoạch triển khai

## Context

Xây dựng web app dự đoán giá cổ phiếu Việt Nam (tăng/giảm/đi ngang T+1) dựa trên dữ liệu giá lịch sử và phân tích sentiment tin tức tiếng Việt. Tiếp cận theo "walking skeleton" — làm end-to-end với 1 mã (VCB) trước, rồi mở rộng ra top 100 mã.

**Mục đích:** Dự án học tập / cá nhân (portfolio + học ML). Web cần hiển thị **disclaimer rõ ràng "Đây không phải khuyến nghị đầu tư"** ở mọi trang dự đoán, nhưng không cần lo pháp lý tư vấn tài chính nặng.

## Tiến độ (2026-06-09)
- ✅ **Khung backend** (FastAPI chạy, 9 bảng ORM + constraints, labeling ±1% + test, Python 3.12) — service/api còn stub.
- ✅ **Khung frontend** (Next.js 15, 4 trang, build sạch, design "Ethereal Glass" + bảng màu chốt ở ADR 0001) — dùng mock data.
- ✅ **Phase 1.1 — giá:** Alembic setup → migration tạo 9 bảng chạy thật trên Supabase → `price_fetcher` (vnstock) lấy OHLCV VCB → `price_history` (upsert idempotent). Có `stock_seed`, `db/upsert` đa dialect, test transform/validate/idempotent.
- ✅ **Phase 1.1 — tin:** crawler CafeF lấy tin VCB → `news` + `news_stocks` (lịch sự: robots.txt + delay + UA, idempotent theo url, content=NULL). 20 bài VCB đã vào Supabase. FireAnt hoãn (cần token API).
- 🔜 **Đang làm:** Phase 1.2 — sentiment (auto-label LLM → fine-tune PhoBERT → chấm `news.sentiment_score` → tổng hợp `daily_sentiment`).
- Chi tiết trạng thái: xem `CONTEXT.md`.

## Định nghĩa nhãn (quan trọng — nền tảng của toàn pipeline)

Nhãn cho ngày T được tính từ % thay đổi giá đóng cửa T+1 so với T:

| % thay đổi (close T+1 vs close T) | Nhãn |
|-----------------------------------|------|
| > +1% | **Tăng** |
| < -1% | **Giảm** |
| trong khoảng [-1%, +1%] | **Đi ngang** |

Nhãn dùng để (a) gán nhãn dữ liệu train, (b) backtest accuracy, (c) hiển thị cho người dùng. Baseline ngẫu nhiên ≈ 33%.

---

## Quyết định kiến trúc đã xác nhận

| Hạng mục | Quyết định |
|----------|-----------|
| Thị trường | Việt Nam — HOSE, HNX, UPCOM |
| Mục tiêu dự đoán | 3-class: Tăng / Giảm / Đi ngang (T+1) |
| Dữ liệu đầu vào | OHLCV + Sentiment từ tin tức/diễn đàn |
| Nguồn dữ liệu giá | `vnstock` Python library (TCBS/SSI) |
| Nguồn tin tức | CafeF.vn + FireAnt.vn |
| ML Model | TFT (Temporal Fusion Transformer) — **1 model global**, `stock_id` static feature |
| Sentiment label data | Auto-label bằng LLM (GPT/Gemini), người review mẫu sai |
| Định nghĩa nhãn | Ngưỡng ±1% (Tăng / Giảm / Đi ngang) |
| Tin tức lịch sử | Crawl tối đa; ngày không có tin → sentiment = 0 (trung tính) |
| Sentiment NLP | PhoBERT (`vinai/phobert-base`) fine-tuned, chạy local |
| Train compute | Kaggle GPU (T4/P100) |
| Inference | Localhost (MacBook M5) |
| Frontend | Next.js + TailwindCSS |
| Backend | FastAPI (Python) |
| Database | Supabase (PostgreSQL) |
| Cache | Upstash (Redis) |
| Auth | NextAuth.js — Email/Password + Google OAuth |
| Deploy Frontend | Vercel |
| Deploy Backend | Localhost (dev), Railway (production sau) |
| Scope mã CK | Top 100 (VN30 + blue-chip thanh khoản cao) |
| Tần suất cập nhật | 1 lần/ngày lúc ~16:00 sau đóng cửa phiên |

---

## Kiến trúc hệ thống

```
[CafeF / FireAnt]  [vnstock API]
       │                  │
       ▼                  ▼
  [Crawler Service]  [Price Fetcher]
       │                  │
       └────────┬─────────┘
                ▼
         [PostgreSQL / Supabase]
                │
       ┌────────┴────────┐
       ▼                 ▼
 [PhoBERT Sentiment]  [TFT Model]
       │                 │
       └────────┬────────┘
                ▼
         [Prediction Store]
                │
                ▼
         [FastAPI Backend] ←→ [Redis / Upstash]
                │
                ▼
         [Next.js Frontend] → [Vercel]
```

---

## Cấu trúc thư mục

```
TradePilot/
├── frontend/   # Next.js: page.tsx, stock/[symbol], accuracy, watchlist
├── backend/    # FastAPI: api/, services/(crawler,price_fetcher,sentiment,scheduler), models/
├── ml/         # notebooks/(exploration, sentiment_finetune, tft_training) + artifacts/
└── docker-compose.yml
```

Mỗi module có `CLAUDE.md` (context/rules) + `SKILL.md` (quy trình). Root `CLAUDE.md` chứa bất biến + engineering workflow chung.

---

## Database Schema (Supabase / PostgreSQL)

> Tạo & quản lý schema bằng **Alembic** (autogenerate từ ORM `backend/models/database.py`). Mỗi thay đổi schema = 1 migration version. Không dùng `create_all`/SQL tay.

```sql
stocks (id, symbol, name, exchange, sector, is_active)
price_history (id, stock_id, date, open, high, low, close, volume)
news (id, title, content, url, source, published_at, sentiment_score)   -- 1 bài nhiều mã
news_stocks (id, news_id, stock_id)                                     -- N-N
daily_sentiment (id, stock_id, date, sentiment_agg, news_count)         -- input TFT; 0 nếu ko tin
predictions (id, stock_id, prediction_date, target_date, label, confidence, model_version)
actual_results (id, stock_id, date, label)                             -- fill khi biết close T+1
users (id, email, name, provider, created_at)
watchlist (id, user_id, stock_id, created_at)
```

---

## Kế hoạch theo phase

### Phase 1 — Walking Skeleton với VCB (1–2 tuần)
Pipeline end-to-end cho duy nhất mã VCB.

**1.1 Data pipeline**
- [x] Setup Alembic → migration tạo 9 bảng trên Supabase
- [x] `vnstock` → OHLCV VCB → `price_history` (upsert theo stock_id+date, idempotent)
- [x] Crawler CafeF → `news` + `news_stocks` (lịch sự + idempotent, content=NULL). 2 nguồn: trang theo mã + RSS chuyên mục map về mã (match symbol/tên, N-N). 118 bài, 6 mã seed. FireAnt hoãn (API cần token).

**1.2 Sentiment model**
- [x] **Mắt xích backend (C):** `score_news` + `build_daily_sentiment` (gộp theo ngày/mã, idempotent, ngày-không-tin=0) + test. `score_text` STUB trả 0.0 — chờ PhoBERT. Verify: VCB 17 ngày, FPT 13 ngày `daily_sentiment`.
- [ ] Auto-label ~500 câu bằng LLM → review mẫu sai (A)
- [ ] Fine-tune `vinai/phobert-base` trên Kaggle → `ml/artifacts/phobert_sentiment/` (B)
- [ ] Thay `score_text` stub bằng PhoBERT thật → re-score + rebuild `daily_sentiment`

**1.3 TFT Model**
- [ ] Feature: MA7/MA20, RSI, MACD, `sentiment_agg`, `news_count`
- [ ] **Chống leakage:** ngày T chỉ dùng tin `published_at` ≤ 16:00 phiên T; target close T+1
- [ ] Train TFT global (`pytorch-forecasting`), `stock_id` static categorical
- [ ] **Backtest walk-forward** 90 ngày → accuracy; export checkpoint + `model_version`

**1.4 FastAPI backend** _(khung ✅, cần nối logic)_
- [x] Khung routes `/api/predictions`, `/api/stocks/{symbol}/history`, `/api/accuracy` (stub) + scheduler khung
- [ ] Nối logic thật trả data từ DB (qua Redis cache)

**1.5 Next.js frontend (MVP)** _(khung ✅)_
- [x] Trang chủ + `/stock/[symbol]` + `/accuracy` + `/watchlist`, design + disclaimer mọi trang
- [ ] Đổi `USE_MOCK=false` khi API thật sẵn sàng; deploy Vercel

### Phase 2 — Mở rộng Top 100 mã (2–3 tuần)
- [ ] Crawler 100 mã (VN30 + VNMID + blue-chip)
- [ ] Train 1 TFT global cho 100 mã (không per-stock)
- [ ] Trang chủ đầy đủ + bộ lọc (sàn/ngành/tín hiệu) + trang accuracy có biểu đồ

### Phase 3 — Auth + Watchlist + Polish (1–2 tuần)
- [ ] NextAuth (Email + Google) + JWT validation FastAPI
- [ ] Trang watchlist + notification tín hiệu mạnh
- [ ] Dark mode, responsive mobile

---

## Engineering workflow

- **Git:** agent nhận issue `ready-for-agent` → branch → PR → người duyệt+merge. Không commit thẳng `main`. Conventional Commits, `Closes #N`.
- **Secrets:** `.env` (gitignore) + `.env.example` mỗi module. Không hardcode/commit secret.
- **Tooling:** `uv` + `ruff` (Python); `eslint` + `prettier` (frontend).
- **Testing:** `pytest` trọng điểm — nhãn ±1%, no-leakage, aggregation sentiment, crawler parser (HTML fixture).
- **CI/CD:** GitHub Actions chạy lint + test trên PR, chặn merge nếu fail. Vercel preview frontend.
- **Observability:** structured logging JSON + Sentry; alert khi daily job fail hoặc crawl 0 tin.
- **Data governance:** crawl lịch sự (robots.txt, rate-limit, delay, UA); chỉ lưu link + sentiment; retention tin N tháng.
- **Model governance:** retrain hàng tháng hoặc khi rolling accuracy 30 ngày tụt; bump `model_version`; lưu accuracy theo version.

---

## Thư viện chính

| Package | Mục đích |
|---------|----------|
| `vnstock` | OHLCV từ TCBS/SSI |
| `beautifulsoup4` + `httpx` | Crawl CafeF, FireAnt |
| `transformers` + `torch` | PhoBERT fine-tune + inference |
| `openai` / `google-generativeai` | LLM auto-label sentiment (1 lần) |
| `pytorch-forecasting` | TFT model |
| `apscheduler` | Daily job |
| `sqlalchemy[asyncio]` + `asyncpg` | PostgreSQL ORM async (greenlet) |
| `alembic` | Migration schema (autogenerate từ ORM) |
| `redis` + `upstash-redis` | Cache |
| `fastapi` + `uvicorn` | Backend API |
| `next` + `tailwindcss` + `next-auth` | Frontend + auth |
| `recharts` / `lightweight-charts` | Biểu đồ |

---

## Verification

1. **Pipeline:** `python -m backend.services.crawler --symbol VCB` → Supabase có data
2. **Sentiment:** 5 câu tin → label đúng chiều
3. **Model:** backtest walk-forward VCB → accuracy > 50% (baseline 33%)
4. **API:** `curl localhost:8000/api/predictions?symbol=VCB` → JSON hợp lệ
5. **Frontend:** Vercel URL → bảng dự đoán → click VCB vào trang chi tiết
6. **Auth:** đăng nhập Google → thêm VCB watchlist → refresh vẫn còn
7. **Scheduler:** trigger manual 16:00 → prediction mới trong DB
8. **No-leakage test:** assert feature builder không chứa close T+1 trong input
