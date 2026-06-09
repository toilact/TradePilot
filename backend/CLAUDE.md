# Backend — FastAPI

API + data pipeline cho TradePilot. Cùng Python với ML để gọi inference trực tiếp.

## Stack
- FastAPI + uvicorn (async), SQLAlchemy + asyncpg → Supabase (PostgreSQL)
- Redis (Upstash) cache kết quả dự đoán
- APScheduler: job 16:00 hàng ngày (crawl → sentiment → predict)

## Cấu trúc
- `main.py` — app entry
- `api/` — `predictions.py`, `stocks.py`, `auth.py`
- `services/` — `crawler.py` (CafeF+FireAnt), `price_fetcher.py` (vnstock), `sentiment.py` (PhoBERT inference), `scheduler.py`
- `models/database.py` — SQLAlchemy models + Supabase connection

## Schema chính
`stocks`, `price_history`, `news`, `news_stocks` (N-N), `daily_sentiment`, `predictions`, `actual_results`, `users`, `watchlist`. Định nghĩa ở `models/database.py` (SQLAlchemy 2.0 async, có unique constraints để upsert).

## Migration
- Dùng **Alembic** (autogenerate từ ORM). Mỗi đổi schema = 1 migration; KHÔNG `create_all`/SQL tay trên Supabase.
- Lệnh: `uv run alembic revision --autogenerate -m "..."` → review → `uv run alembic upgrade head`.

## Quy tắc bắt buộc
- **Chống data leakage:** dự đoán ngày T chỉ dùng tin `published_at ≤ 16:00 phiên T`; nhãn là close T+1.
- **Nhãn:** Tăng >+1%, Giảm <-1%, còn lại Đi ngang.
- Ngày không có tin → `daily_sentiment.sentiment_agg = 0`.
- Đọc model từ `../ml/artifacts/` — backend chỉ inference, KHÔNG train ở đây.
- Cache prediction qua Redis; chỉ chạy inference khi cache miss.

## Endpoints
- `GET /api/predictions?symbol=` — dự đoán mới nhất
- `GET /api/stocks/{symbol}/history` — giá + sentiment timeline
- `GET /api/accuracy` — thống kê độ chính xác
- Auth: NextAuth JWT validation (Email + Google)

## Workflow (xem root CLAUDE.md cho quy ước chung)
- **Python pin 3.12** (`.python-version`) — KHÔNG dùng 3.13/3.14 (vỡ wheel `torch`/`vnstock`). SQLAlchemy dùng extra `[asyncio]`.
- Tooling: `uv` (deps + lock) + `ruff` (lint+format). `uv run pytest` để test.
- Secrets: `.env` (gitignore) + `.env.example`. Cần: Supabase URL/key, OpenAI/Gemini key, Upstash Redis, Google OAuth.
- Test bắt buộc: hàm gán nhãn ±1%, kiểm tra leakage (chỉ tin ≤16:00 phiên T), aggregation `daily_sentiment`, parser crawler (HTML fixture).
- Logging JSON + Sentry. Daily job phải log thành/bại + số mã; alert khi fail hoặc crawl 0 tin.
- Crawl lịch sự: robots.txt, rate-limit + delay, User-Agent rõ; chỉ lưu link + sentiment (không full text).
