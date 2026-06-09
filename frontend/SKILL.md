---
name: frontend-build-pages
description: Xây dựng và sửa các trang Next.js của TradePilot (bảng dự đoán, chi tiết mã, accuracy, watchlist). Dùng khi thêm UI, tích hợp API backend, chart giá/sentiment, hoặc auth.
---

# Skill: Build frontend pages

## Nguyên tắc dữ liệu
- Mọi số liệu (nhãn, confidence, sentiment, accuracy) lấy từ backend API. KHÔNG tính lại ở client.
- Fetch trong Server Components khi có thể; cache theo prediction date.

## Từng trang
- **Trang chủ:** bảng 100 mã, cột mã/giá/nhãn/confidence; filter sàn+ngành+tín hiệu. Badge màu theo nhãn.
- **Chi tiết `stock/[symbol]`:** chart giá (recharts/lightweight-charts) + overlay sentiment timeline + danh sách tin + lịch sử dự đoán vs thực tế.
- **Accuracy:** biểu đồ độ chính xác theo 30/90 ngày, theo nhãn (confusion matrix gọn).
- **Watchlist:** chỉ khi đăng nhập; thêm/xóa mã, lưu qua backend.

## Bất biến — không được vi phạm
- Disclaimer "không phải khuyến nghị đầu tư" phải xuất hiện mọi nơi có dự đoán.
- Màu: Tăng=xanh, Giảm=đỏ, Đi ngang=xám (nhất quán toàn app).
- Luôn hiện confidence + model_version cạnh dự đoán.

## Auth (Phase 3)
- NextAuth.js: Email/Password + Google. JWT gửi kèm request tới FastAPI cho route watchlist.

## Deploy
- Vercel (frontend). Backend dev ở localhost:8000 — cấu hình `NEXT_PUBLIC_API_URL`.
