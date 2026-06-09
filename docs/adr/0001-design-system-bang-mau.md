# 0001 — Design system & bảng màu frontend

**Status:** Accepted
**Date:** 2026-06-09

## Context

Frontend dùng phong cách "Ethereal Glass" (dark, glass card double-bezel, bento layout). Bản đầu nền đen tuyền `#050505` bị đánh giá quá tối/nặng. Cần chốt bảng màu và nguyên tắc dùng màu để (a) chữa cảm giác tối, (b) tránh agent/người sau vô tình phá vỡ tính nhất quán — đặc biệt là nhầm lẫn màu thương hiệu với màu tín hiệu dữ liệu.

## Decision

### Bảng màu
- **Nền:** `#0F1115` (than ngả xanh nhẹ) — dịu hơn đen tuyền, vẫn "dark sang".
- **Chữ chính:** `#E8EAED` (gần trắng); chữ phụ theo tầng white/50–65 (tương phản cao, dễ đọc số liệu).
- **Card (double-bezel):** lõi sáng hơn nền rõ rệt (`rgba(255,255,255,0.055)` + inset highlight) để "nổi" lên.
- **Hairline:** `rgba(255,255,255,0.12)`.

### Nguyên tắc dùng màu (QUAN TRỌNG — bất biến UI)
- **Gold/champagne (`#E8C39E`) = màu THƯƠNG HIỆU.** Dùng cho logo, eyebrow, accent UI, quầng sáng nền.
- **Emerald (`#34D399`) / Rose đỏ (`#FB7185`) / Xám (`#A1A1AA`) = màu DỮ LIỆU.** Chỉ dùng cho tín hiệu: Tăng / Giảm / Đi ngang và sentiment.
- **Không trộn hai vai trò:** không dùng emerald cho nút/accent thương hiệu; không dùng gold cho tín hiệu dữ liệu.
- Nền chỉ có 2 quầng champagne mờ (không emerald/violet) để không "cướp" sự chú ý khỏi tín hiệu dữ liệu.

### Typography
- Display: **Space Grotesk**. Body: **Plus Jakarta Sans**. (Không dùng Inter/Roboto/Arial.)

## Consequences
- **Dễ hơn:** người dùng phân biệt tức thì "màu UI" vs "tín hiệu giá" — chuẩn UX tài chính; quyết định màu sau này có khung rõ ràng để theo.
- **Khó hơn / ràng buộc:** mọi thành phần UI mới phải tuân nguyên tắc brand-vs-data; thêm màu mới cần cân nhắc rơi vào vai trò nào.
- Token cụ thể sống trong `frontend/tailwind.config.ts` + `frontend/app/globals.css`; bảng tra cứu nhanh nằm trong `frontend/CLAUDE.md`.
