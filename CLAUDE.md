# TradePilot

Web dự đoán cổ phiếu Việt Nam (Tăng/Giảm/Đi ngang T+1) từ giá lịch sử + sentiment tin tức tiếng Việt. Dự án **học tập/cá nhân** (solo + agent AI).

## Module
- `frontend/` — Next.js + Tailwind (UI). Xem `frontend/CLAUDE.md`.
- `backend/` — FastAPI + pipeline + API. Xem `backend/CLAUDE.md`.
- `ml/` — PhoBERT sentiment + TFT, train trên Kaggle. Xem `ml/CLAUDE.md`.

Luồng: nhãn ±1% → gán nhãn data → daily_sentiment + giá → TFT global → predictions → API → frontend.

## Bất biến toàn dự án (mọi module phải tuân thủ)
- **Nhãn:** Tăng >+1%, Giảm <-1%, còn lại Đi ngang (close T+1 vs T).
- **Chống data leakage:** feature ngày T chỉ dùng thông tin có trước 16:00 phiên T; nhãn là close T+1. Backtest **walk-forward**, cấm random split.
- 1 bài báo → nhiều mã qua `news_stocks`; ngày không tin → sentiment = 0.
- Backend là nguồn sự thật; ml chỉ train+export; frontend chỉ hiển thị.
- Mọi nơi hiển thị dự đoán phải có disclaimer "Đây không phải khuyến nghị đầu tư".

## Engineering workflow

### Git & code review (solo + agent)
- Agent nhận issue gắn `ready-for-agent` → làm trên **branch riêng** → mở **PR** → người duyệt+merge. KHÔNG commit thẳng vào `main`.
- Conventional Commits (`feat:`, `fix:`, `chore:`...). Mỗi PR link issue (`Closes #N`).
- Issue tracker = GitHub Issues; triage labels: xem `docs/agents/`.

### Secrets
- Mỗi module có `.env` (**gitignore**) + `.env.example` (chỉ tên biến, không giá trị).
- **TUYỆT ĐỐI** không commit `.env`, không hardcode secret (Supabase, OpenAI/Gemini, Google OAuth, Upstash).

### Tooling
- Python (backend, ml): `uv` (package + lock) + `ruff` (lint + format).
- Frontend: `eslint` + `prettier`.

### Testing
- `pytest` trọng điểm — bắt buộc test: hàm gán nhãn ±1%, kiểm tra leakage, aggregation sentiment, parser crawler (dùng HTML fixture). UI test thủ công cho v1.

### CI/CD (GitHub Actions)
- Mỗi PR tự chạy `ruff` + `pytest` (backend/ml) và `eslint` (frontend); **chặn merge nếu fail**.
- Frontend auto-deploy Vercel (preview/PR). Backend deploy Railway thêm ở Phase production.

### Observability
- Structured logging (JSON). Sentry bắt exception ở backend + frontend.
- Daily job log thành/bại + số mã xử lý; **alert** (email/Telegram) khi job fail hoặc crawl 0 tin.

### Data governance
- Crawl lịch sự: tôn trọng `robots.txt`, rate-limit + delay, User-Agent rõ ràng, cache tránh lặp.
- Chỉ lưu **link + sentiment_score** (không tái xuất bản nguyên văn bài báo).
- Retention: giá giữ vô thời hạn; nội dung tin giữ N tháng rồi chỉ giữ metadata.

### Model governance
- Retrain hàng tháng HOẶC khi rolling accuracy 30 ngày tụt dưới ngưỡng.
- Mỗi lần retrain bump `model_version`; lưu accuracy theo version. Trang `accuracy/` vừa cho người dùng vừa để giám sát drift.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
