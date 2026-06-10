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

**Giai đoạn: Pipeline data→feature đã thông end-to-end. Tiếp: train TFT trên Kaggle → nối API.**

GitHub remote: `github.com/toilact/TradePilot`. Đã có 4 PR (#1 crawler, #2 sentiment, #3 feature builder đã merge vào `main`; #4 TFT notebook đang mở). Mỗi việc 1 branch → PR.

Đã có (chạy thật, có test, verify trên Supabase):
- ✅ **Khung backend + frontend** (FastAPI + 9 bảng ORM; Next.js 15 4 trang mock). Python 3.12, `uv`+`ruff`.
- ✅ **Alembic** + migration `f6e9b2c93653` (9 bảng) trên Supabase.
- ✅ **`price_fetcher`** (vnstock) + `stock_seed` + `db/upsert` (đa dialect, idempotent). **Đã fetch giá 30 mã VN30 (~96833 phiên)** vào `price_history`.
- ✅ **`crawler`** (CafeF): 2 nguồn — trang theo mã + RSS chuyên mục map về mã (`match_stocks`, N-N). Lịch sự (robots.txt+delay+UA), idempotent theo url, `content=NULL`. ~118 bài. FireAnt hoãn (API cần token).
- ✅ **`sentiment`** mắt xích backend: `score_news` + `build_daily_sentiment` (gộp ngày/mã, idempotent, ngày-không-tin=0). `score_text` **STUB trả 0.0** — chờ PhoBERT. VCB 17 ngày + FPT 13 ngày `daily_sentiment`.
- ✅ **`backend/features/builder.py`** (dùng chung train+inference): MA7/MA20/RSI14/MACD + sentiment + label ±1%. **Chống leakage** (feature ≤T, label tách riêng, hàng cuối=None) + test no-leakage. `load_training_frame(symbol)`.
- ✅ **`ml/notebooks/03_tft_training.py`** (PR #4): TFT global 30 mã, split THỜI GIAN (train<2024/val 2024/test≥2025), baseline + confusion + F1, export checkpoint. Smoke-test pass. Dep ML chỉ ở `ml/requirements.txt` (backend KHÔNG có torch).
- ✅ **`.env` thật** (Supabase asyncpg). `CLAUDE.md` đã untrack (không publish lên GitHub).

Chưa có:
- ❌ **Train TFT thật** — notebook mới smoke-test; cần chạy `RUN_TRAINING=True` trên **Kaggle GPU** (việc của user, ngoài session). Chưa có checkpoint thật.
- ❌ **PhoBERT thật** (Phase 1.2 A+B): auto-label LLM + fine-tune Kaggle. `score_text` còn stub → `sentiment_agg` toàn 0.
- ❌ API chưa nối DB thật (Phase 1.4), frontend còn `USE_MOCK=true`. Chưa CI.

⚠️ **Sự thật cần nhớ (đừng ảo tưởng accuracy):**
- **Baseline thực tế ~53%** (lớp "đi ngang" đa số), KHÔNG phải 33%. TFT phải vượt 53% mới có giá trị.
- **Sentiment hiện vô dụng** (toàn 0, phủ ~3 tháng/17 năm) — model học từ feature GIÁ. Giữ pipeline để thay PhoBERT sau.
- **API classification pytorch-forecasting** (`CrossEntropy`/`output_size=3`) đổi theo version → notebook ghi TODO xác minh trên Kaggle + fallback GRU/LSTM, KHÔNG bịa.
- Debt nhỏ: `price_fetcher.validate()` chưa chặn close/open ngoài [low,high]; `sync_database_url` chưa có test; `stocks.name` là tên pháp lý (không phải brand) → `match_stocks` theo tên phủ thấp, chủ yếu dựa symbol.

## Quyết định đã chốt (đừng hỏi lại)

Thị trường VN · nhãn 3-class ±1% T+1 · OHLCV + sentiment · `vnstock` + crawl CafeF/FireAnt · PhoBERT + TFT global · sentiment data auto-label bằng LLM · Kaggle train, localhost inference · Next.js + FastAPI + Supabase + Upstash + NextAuth(Email+Google) · Vercel + Railway(sau) · top 100 mã · cập nhật 16:00/ngày · solo + agent · `uv`+`ruff` / `eslint`+`prettier` · pytest trọng điểm · GitHub Actions CI · Sentry · crawl lịch sự · retrain hàng tháng.

**Quyết định mới (2026-06-09):**
- **Migration:** dùng **Alembic** (autogenerate từ ORM), không dùng `create_all`/SQL tay.
- **Backend chạy Python 3.12** (3.14 vỡ wheel ML); SQLAlchemy dùng extra `[asyncio]` (greenlet).
- **Design system:** gold = brand, emerald/đỏ/xám = data only (ADR 0001).
- **Feature builder** đặt ở `backend/features/` (backend là nguồn sự thật); ml import từ đây.
- **Scope train Phase 1:** mở rộng từ VCB → **30 mã VN30** (TFT global thật). FireAnt hoãn.
- **Dep ML** (torch/pytorch-forecasting) CHỈ ở `ml/`, KHÔNG vào backend (backend chỉ inference).
- **`CLAUDE.md` không publish** — gitignore + untrack (giữ local).

## Bước tiếp theo gợi ý (theo thứ tự)

1. ✅ Xong: price_fetcher (30 mã VN30) · crawler CafeF/RSS · sentiment aggregate (stub) · feature builder · notebook TFT.
2. **Chạy notebook TFT trên Kaggle GPU** (`RUN_TRAINING=True`): xác minh API classification pytorch-forecasting, train → accuracy out-of-sample > 53% → export checkpoint + `model_version` vào `ml/artifacts/tft_model/`. (Việc của user; có thể cần export panel ra CSV để Kaggle khỏi nối Supabase.)
3. **Phase 1.2 A+B (PhoBERT thật):** auto-label ~500 câu bằng LLM → fine-tune Kaggle → thay `score_text` stub → re-score + rebuild `daily_sentiment`.
4. **Phase 1.4 (nối API):** backend đọc `predictions`/`price_history` qua Redis cache → đổi frontend `USE_MOCK=false` → deploy. (Cần checkpoint từ bước 2; có thể làm trước với predictions stub.)
5. CI (GitHub Actions: ruff + pytest chặn merge).

## File cần đọc khi vào việc

- `PLAN.md` — kế hoạch đầy đủ theo phase + verification
- `CLAUDE.md` (root) — bất biến + engineering workflow
- `backend/CLAUDE.md`, `ml/CLAUDE.md`, `frontend/CLAUDE.md` — context từng module
- `*/SKILL.md` — quy trình thao tác từng module
- `docs/agents/*` — issue tracker + triage convention
