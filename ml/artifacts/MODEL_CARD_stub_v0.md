# Model Card — stub_v0 (KHÔNG PHẢI MODEL THẬT)

> Hồi tố 2026-06-16 (M10 model governance). `stub_v0` là **placeholder** để frontend chạy
> end-to-end SỚM trước khi có model thật — KHÔNG phải dự đoán ML và KHÔNG bao giờ là production.

## Là gì

| | |
|---|---|
| Loại | Quy tắc heuristic (không train, không xác suất) |
| Logic | So **MA7 vs MA20** phiên gần nhất: `(MA7−MA20)/MA20` > +0.5% → `tang`, < −0.5% → `giam`, còn lại `di_ngang` |
| Sinh bởi | `backend/scripts/seed_predictions.py` (`stub_label`) |
| Confidence | Cố định (giả) — KHÔNG calibrate; `is_actionable=false` (server_default) → UI luôn "Không đủ tín hiệu" |
| model_version | `stub_v0` |

## Vai trò & vòng đời

- Dùng ở Phase 1.4 để dựng luồng predictions → API → frontend trước khi có model.
- Khi model thật ghi cùng ngày, **stub bị đè**: `read_api._pred_order` ưu tiên non-stub khi trùng
  `prediction_date` (stub giữ lại làm lịch sử, không hiển thị).
- Không có metrics file (không phải kết quả ML). Không nằm trong frontier/gating.

## Versioning (đếm GLOBAL monotonic)

`v0 stub_v0` → `v1/v2` TFT (collapse, xem `tft_model/MODEL_CARD_tft_v1.md`) → `v3+` LightGBM
(production: `lgbm_model/MODEL_CARD_lgbm_v4.md`).

## Lưu ý governance

Mọi nơi hiển thị stub vẫn phải có disclaimer "Đây không phải khuyến nghị đầu tư" (như mọi prediction).
stub_v0 KHÔNG được tính vào coverage/precision gating của version production (read_api lọc theo version hiện hành).
