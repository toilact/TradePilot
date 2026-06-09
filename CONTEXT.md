# TradePilot — Context (handoff)

Domain context + trạng thái dự án để bắt đầu một session mới. Đọc file này + `PLAN.md` + các `CLAUDE.md` là đủ để tiếp tục.

## Dự án là gì

Web dự đoán cổ phiếu Việt Nam: với mỗi mã, dự đoán phiên T+1 sẽ **Tăng / Giảm / Đi ngang**, dựa trên giá lịch sử (OHLCV) + sentiment tin tức tiếng Việt. Dự án **học tập/cá nhân** (solo + agent AI). Không phải khuyến nghị đầu tư.

## Domain glossary (ngôn ngữ chung)

- **T / T+1** — phiên giao dịch hiện tại / phiên kế tiếp. Dự đoán luôn cho T+1.
- **Nhãn (label)** — Tăng (close T+1 > +1%), Giảm (< -1%), Đi ngang ([-1%, +1%]). Baseline ngẫu nhiên ≈ 33%.
- **OHLCV** — Open/High/Low/Close/Volume, dữ liệu giá theo ngày (nguồn `vnstock`).
- **Sentiment** — điểm cảm xúc tin tức (-1..1) do PhoBERT chấm; tổng hợp theo ngày/mã thành `daily_sentiment` (0 nếu ngày không có tin).
- **TFT** — Temporal Fusion Transformer; **1 model global** cho mọi mã, `stock_id` là static feature.
- **model_version** — phiên bản model; bump mỗi lần retrain để truy vết predictions/accuracy.
- **Top 100** — VN30 + blue-chip thanh khoản cao (scope mục tiêu). Phase 1 chỉ làm VCB.

## Bất biến (KHÔNG được vi phạm)

1. **Nhãn ±1%** thống nhất ở train, backtest, hiển thị.
2. **Chống data leakage:** feature ngày T chỉ dùng thông tin có trước 16:00 phiên T; nhãn là close T+1. Backtest **walk-forward**, cấm random split/shuffle thời gian.
3. 1 bài báo → nhiều mã qua `news_stocks`. Ngày không tin → sentiment = 0.
4. **Backend** là nguồn sự thật; **ml** chỉ train+export; **frontend** chỉ hiển thị (không tự tính nhãn/sentiment).
5. Mọi nơi hiển thị dự đoán phải có disclaimer "Đây không phải khuyến nghị đầu tư".
6. Không commit `.env`/secret. Không train trong backend.

## Trạng thái hiện tại (cập nhật 2026-06-09)

**Giai đoạn: Phase 1.1 — mắt xích giá đã chạy thật (vnstock → Supabase). Tiếp: crawler tin.**

Đã có:
- ✅ `git init` (branch `main`, **chưa commit**, chưa có GitHub remote) + `.gitignore`
- ✅ Tài liệu: `PLAN.md`, `CONTEXT.md`, root + 3 module `CLAUDE.md`/`SKILL.md`, `docs/agents/`, **`docs/adr/0001`** (design system)
- ✅ **Backend khung** (`backend/`): FastAPI chạy được (`/health` OK), 9 bảng ORM (SQLAlchemy 2.0 async, có unique constraints), `labeling.py` (nhãn ±1%) + test pass, service/api đều là stub có TODO. **Python pin 3.12** (`.python-version`); `uv`+`ruff` sạch.
- ✅ **Frontend khung** (`frontend/`): Next.js 15 + Tailwind, 4 trang (chủ/chi tiết/accuracy/watchlist) + components, build sạch. Dùng **mock data** (`lib/api.ts` cờ `USE_MOCK=true`). Design "Ethereal Glass" + bảng màu đã chốt (xem ADR 0001).
- ✅ **Alembic** setup (`alembic.ini`, `migrations/env.py` lấy URL từ settings) + migration `f6e9b2c93653` (9 bảng) **đã chạy thật trên Supabase**.
- ✅ **`price_fetcher`** (vnstock `vci` → OHLCV) + `stock_seed` (metadata mã) + `db/upsert` (đa dialect PG/SQLite, idempotent theo unique constraint). Đã fetch giá VCB vào `price_history` thật. Test: transform/validate/idempotent (SQLite in-memory).
- ✅ **`.env` thật** đã cấu hình (DATABASE_URL asyncpg → Supabase), pipeline giá verify chạy được.

Chưa có:
- ❌ Logic còn lại: crawler (CafeF/FireAnt), sentiment (PhoBERT), TFT — vẫn stub `NotImplementedError`
- ❌ API chưa nối DB thật (Phase 1.4), frontend còn `USE_MOCK=true`
- ❌ Chưa commit, chưa GitHub remote, chưa CI, chưa ml notebooks
- ⚠️ Lưu ý debt nhỏ (xem review): `validate()` chưa chặn close/open ngoài [low,high]; `sync_database_url` chưa có test.

## Quyết định đã chốt (đừng hỏi lại)

Thị trường VN · nhãn 3-class ±1% T+1 · OHLCV + sentiment · `vnstock` + crawl CafeF/FireAnt · PhoBERT + TFT global · sentiment data auto-label bằng LLM · Kaggle train, localhost inference · Next.js + FastAPI + Supabase + Upstash + NextAuth(Email+Google) · Vercel + Railway(sau) · top 100 mã · cập nhật 16:00/ngày · solo + agent · `uv`+`ruff` / `eslint`+`prettier` · pytest trọng điểm · GitHub Actions CI · Sentry · crawl lịch sự · retrain hàng tháng.

**Quyết định mới (2026-06-09):**
- **Migration:** dùng **Alembic** (autogenerate từ ORM), không dùng `create_all`/SQL tay.
- **Backend chạy Python 3.12** (3.14 vỡ wheel ML); SQLAlchemy dùng extra `[asyncio]` (greenlet).
- **Design system:** gold = brand, emerald/đỏ/xám = data only (ADR 0001).

## Bước tiếp theo gợi ý (theo thứ tự)

1. ~~Phase 1.1 `price_fetcher`~~ ✅ xong (giá VCB đã vào Supabase).
2. **Crawler CafeF/FireAnt cho VCB** (+ HTML fixture để test parser) → `news` + `news_stocks`. Crawl lịch sự (robots.txt, rate-limit, UA), chỉ lưu link + metadata.
3. Sentiment (Phase 1.2) → TFT (Phase 1.3) → nối API (Phase 1.4) → đổi frontend `USE_MOCK=false`.
4. Sau khi có data thật: commit + tạo GitHub remote + CI.

## File cần đọc khi vào việc

- `PLAN.md` — kế hoạch đầy đủ theo phase + verification
- `CLAUDE.md` (root) — bất biến + engineering workflow
- `backend/CLAUDE.md`, `ml/CLAUDE.md`, `frontend/CLAUDE.md` — context từng module
- `*/SKILL.md` — quy trình thao tác từng module
- `docs/agents/*` — issue tracker + triage convention
