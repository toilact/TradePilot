# Frontend — Next.js

UI cho TradePilot. Đọc dữ liệu từ FastAPI backend, deploy Vercel.

## Stack
- Next.js 15 (App Router) + TailwindCSS 3 + TypeScript
- NextAuth.js — Email/Password + Google OAuth (Phase 3)
- Chart: **SVG thuần** (`components/sparkline.tsx`, `components/price-chart.tsx`) — không thêm lib chart nặng

## Trạng thái
Khung đã dựng, build sạch. Dữ liệu hiện là **mock** (`lib/api.ts`, cờ `USE_MOCK=true`).
Khi backend Phase 1.4 trả data thật → đổi `USE_MOCK=false`.

## Routes (app/)
- `page.tsx` — trang chủ: bảng dự đoán + badge Tăng/Giảm/Đi ngang + confidence
- `stock/[symbol]/` — chi tiết: chart giá + sentiment timeline + top tin + lịch sử dự đoán
- `accuracy/` — thống kê độ chính xác mô hình theo thời gian
- `watchlist/` — danh sách theo dõi (yêu cầu đăng nhập)

## Bộ lọc (trang chủ)
Sàn (HOSE/HNX/UPCOM), ngành, tín hiệu (lọc chỉ mã dự đoán Tăng).

## Quy tắc bắt buộc
- **Disclaimer "Đây không phải khuyến nghị đầu tư"** ở footer + mọi nơi hiển thị dự đoán. Bắt buộc.
- Hiển thị `confidence` và `model_version` kèm dự đoán để minh bạch.
- Backend = nguồn sự thật duy nhất; frontend KHÔNG tự tính nhãn/sentiment.

## Design system (xem ADR `docs/adr/0001-design-system-bang-mau.md` cho lý do)
Phong cách "Ethereal Glass" dark + bento + double-bezel. Token sống trong `tailwind.config.ts` + `app/globals.css`.

**Bất biến màu — KHÔNG trộn vai trò brand vs data:**

| Token | Giá trị | Vai trò |
|-------|---------|---------|
| nền `--bg` | `#0F1115` | nền than ngả xanh nhẹ |
| chữ chính | `#E8EAED` | tương phản cao; chữ phụ white/50–65 |
| `gold` | `#E8C39E` | **THƯƠNG HIỆU** — logo, eyebrow, accent UI, quầng nền |
| `up` | `#34D399` emerald | **DỮ LIỆU** — Tăng |
| `down` | `#FB7185` rose | **DỮ LIỆU** — Giảm |
| `flat` | `#A1A1AA` xám | **DỮ LIỆU** — Đi ngang |

- Dùng gold cho brand/UI; emerald/đỏ/xám CHỈ cho tín hiệu dữ liệu. Không nhầm lẫn.
- Font: `font-display` (Space Grotesk) cho tiêu đề, `font-sans` (Plus Jakarta Sans) cho body. Không Inter/Roboto/Arial.
- Card luôn dùng `.bezel-shell` + `.bezel-core` (double-bezel). Eyebrow dùng class `.eyebrow`.
- Motion: easing tùy chỉnh `ease-fluid`/`ease-spring`; scroll reveal qua `<Reveal>`. Chỉ animate `transform`/`opacity`.

## API backend (dev: localhost:8000)
- `GET /api/predictions?symbol=`
- `GET /api/stocks/{symbol}/history`
- `GET /api/accuracy`

## Roadmap
v1: trang chủ + chi tiết + accuracy (✅ khung). Auth/watchlist thật + responsive polish ở Phase 3.

## Workflow (xem root CLAUDE.md cho quy ước chung)
- Tooling: `eslint` + `prettier`. UI test thủ công cho v1.
- Secrets: `.env.local` (gitignore) + `.env.example`. Cần: `NEXT_PUBLIC_API_URL`, NextAuth secret, Google OAuth client.
- CI: `eslint` chạy trên PR; Vercel auto-deploy preview mỗi PR.
- Sentry bắt lỗi phía client.
