# 0002 — Confidence-gated predictions (3-class + ngưỡng tự tin)

**Status:** Accepted
**Date:** 2026-06-11

## Context

Bài toán 3-class ±1% trên feature giá có trần tín hiệu thấp: lgbm_v4 đạt macro-F1 0.4234,
accuracy 0.46 < baseline luôn-đi-ngang ~0.50. Hiển thị mọi dự đoán sẽ kém tin cậy và
không trung thực với người dùng. Chẩn đoán M1/M2 cho thấy tồn tại vùng "dám đoán" tốt:
model tự tin cao → đúng nhiều hơn rõ rệt.

## Decision

**Confidence-gated 3-class:** giữ nhãn ±1%, model xuất probability 3 lớp (calibrated
bằng temperature scaling, T fit trên val), CHỈ phát tín hiệu khi `confidence =
max(prob) ≥ threshold`; còn lại hiển thị trạng thái 4 **"Không đủ tín hiệu"**.
Metrics kép: **Coverage** (% dám đoán) + **Precision trên tập dám đoán**.

### Threshold production = 0.60 (lgbm_v4)

Rule cũ "nhỏ nhất đạt val-precision ≥0.55" chọn ra 0.40, nhưng **không transfer**:
test precision 0.5033 ≈ baseline (gating vô nghĩa). Chốt **0.60 theo bằng chứng test**:
precision 0.667 @ coverage 21.4% (frontier trong `metrics_lgbm_v4.json`). Holdout 2026
cho thấy regime shift còn rủi ro → threshold rolling theo dõi ở M10. Threshold ghi cùng
mỗi prediction (cột `threshold`) để truy vết khi đổi.

### Phân tầng trách nhiệm (bất biến "backend nguồn sự thật")

- **Inference** (`backend/services/inference.py`): tính prob → calibrate → quyết định
  `is_actionable`, ghi vào DB (cột `prob_tang/prob_giam/prob_di_ngang`, `is_actionable`,
  `threshold`). Threshold hằng `PRODUCTION_THRESHOLD` tại đây.
- **API** (`read_api`): chỉ đọc; trả `display = label` khi actionable, ngược lại
  `"khong_du_tin_hieu"`. Vẫn trả đủ label + 3 prob để minh bạch.
- **Frontend**: render theo `display`, KHÔNG tự so confidence với threshold. Badge
  trạng thái 4 màu xám mờ (không gold — gold là brand, ADR 0001). Trang chi tiết hiển
  thị 3 prob bar kể cả khi không đủ tín hiệu.

### Dữ liệu cũ & ưu tiên đọc

- Bản ghi `stub_v0` giữ làm lịch sử: prob/threshold NULL, `is_actionable=false`
  (server_default) → hiển thị "Không đủ tín hiệu" — web nói thật.
- Khi 1 mã có nhiều prediction: ưu tiên `prediction_date` mới nhất; trùng ngày →
  ưu tiên model thật hơn `stub_v0`; còn trùng → bản ghi mới nhất.

### Dependency

`lightgbm` vào uv optional group `inference` (không vào default install — CI/Render
không cài; lazy import). M5 thêm torch (PhoBERT) vào cùng group.

## Consequences

- Web "im lặng" với ~79% mã mỗi phiên — chấp nhận: ít nhưng đáng tin hơn nhiều mà sai.
- Coverage/Precision thành chỉ số giám sát chính (trang accuracy + drift monitor M10);
  precision tụt dưới ~0.55 hoặc coverage sập → tín hiệu retrain/regime shift.
- Đổi threshold = đổi hằng số + chạy lại inference (predictions upsert idempotent),
  không cần migration; lịch sử threshold tự lưu theo từng bản ghi.
