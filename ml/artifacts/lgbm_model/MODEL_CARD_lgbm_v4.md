# Model Card — lgbm_v4

> Theo template `MODEL_CARD_lgbm_v3.md`. 1 card + 1 metrics JSON mỗi version; weights gitignore,
> card + metrics commit.

## Tóm tắt

| | |
|---|---|
| **model_version** | `lgbm_v4` (đếm global: v0 stub → v1, v2 TFT → v3 → v4 LightGBM + market features) |
| **Family** | LightGBM (gradient boosting, multiclass) — `LGBMClassifier` defaults + `class_weight="balanced"`, seed 42, `sector` categorical native |
| **Ngày train** | 2026-06-11, local Mac (vài giây — không cần Kaggle) |
| **Train script** | `ml/notebooks/04_lgbm_training.py` (tham số hoá `MODEL_VERSION`; chạy lại 100%: panel CSV + seed cố định) |
| **Artifacts** | `lgbm_v4.txt` (native Booster, gitignore) + `metrics_lgbm_v4.json` (commit) |
| **Kết quả** | **PASS + VƯỢT GATE v3** — macro-F1 **0.4234** > v3 **0.4018** (+5.4%), cả 3 lớp F1 > 0, gating mạnh |
| **Trạng thái** | Đạt DoD M2. Đủ điều kiện wire frontend ở M3 (thay `stub_v0`). |

## Thay đổi so với v3

Thêm **7 feature volatility/momentum nội tại** (`ret_1d, ret_5d, abs_ret_1d, vol_5, vol_20,
dist_ma20, ma_ratio`) + **`sector`** (ICB level 2, static categorical). Bỏ `ma7/ma20` tuyệt đối
(redundant với close). **Loại cả 4 index feature** — EDA notebook 05 phát hiện MI 0.194 của index là
GIẢ (lặp 30 mã/ngày; trên 1 mã = 0.001). Bỏ `sentiment_agg/news_count` khỏi feature model (stub
chết, thêm lại M8). Chi tiết: notebook 05 cell cuối + TICKLIST M2 + grill 2026-06-11.

## Data & split (walk-forward, cấm random split)

- Panel: `ml/data/training_panel.csv` — 96,203 hàng sau lọc, 30 mã VN30, 2008→2026-06-09.
- Split THỜI GIAN: train **2010 → 2024** (74,591; cắt 2008-2009 chế độ dị thường, 4.4% data) ·
  val 2024 (7,500) · test 2025 (7,470).
- **Vì sao test 2025 (giống v3) chứ không test 2026:** đã thử split dịch tiến test 2026 → cả v3 lẫn
  v4 đều ~0.40 vì 2026 mỏng (3,090 dòng/5 tháng) + lệch chế độ (di_ngang 42% vs train 53%) → chôn
  cải thiện thật. So 2×2 (feature × test period) chứng minh: trên **cùng test 2025, v4 0.4234 >
  v3 0.4018**; feature mới tốt thật, vấn đề là tập 2026. → test 2025 cho gate công bằng; 2026 báo
  cáo riêng làm holdout (dưới).
- Features (11): xem `LGBM_V4_FEATURES` trong `backend/features/builder.py` (nguồn sự thật, train
  import — chống train/serve skew). Label ±1% close T+1 (builder chống leakage).

## Kết quả test (out-of-sample 2025)

- **macro-F1 0.4234** (PASS ≥ 0.36 VÀ > v3 0.4018) · accuracy 0.4775 (THAM CHIẾU — baseline
  luôn-di-ngang 0.5365).
- Per-class F1: di_ngang **0.626** · giam **0.323** · tang **0.321** (cân hơn v3 0.61/0.30/0.29).
- Pred-dist: di_ngang 48.6% / giam 28.7% / tang 22.8% (không lớp nào > 80%).
- **Feature importance (gain) — volatility thống trị:** `vol_20` **27672** > `vol_5` **11482** >
  abs_ret_1d 7323 > ret_5d 7225 > ret_1d 6116 > dist_ma20 5995 > rsi14 4798 > ma_ratio 4764 >
  macd_signal 4233 > macd 4030 > sector 3614. Xác nhận luận đề EDA "biên độ trước, hướng sau":
  vol_5/vol_20 là tín hiệu chính, vượt xa rsi14 (feature mạnh nhất của v3).

## Calibration & confidence-gating

- Temperature scaling fit trên val: **T = 0.6148** (< 1 → sharpen).
- Threshold (rule: nhỏ nhất đạt precision ≥ 0.55 & coverage ≥ 20% trên VAL): **0.40**.
- **Threshold transfer val→test ĐÃ TỐT (chữa được bệnh v3):** v3 thr 0.60 chỉ đạt test precision
  0.5521; **v4 thr 0.60 → test precision 0.667 @ coverage 21.4%**. Transfer ổn định ở mọi mức:

  | thr | val precision/cov | test precision/cov |
  |---|---|---|
  | 0.40 | 0.587 / 0.804 | **0.503** / 0.817 |
  | 0.55 | 0.727 / 0.325 | **0.640** / 0.298 |
  | 0.60 | 0.750 / 0.245 | **0.667** / 0.214 |
  | 0.70 | 0.791 / 0.134 | **0.736** / 0.109 |

- Gate M2 "tồn tại threshold ≥0.50/coverage ≥20% trên test" ĐẠT mạnh hơn v3.
- Frontier đầy đủ (val + test, 0.40→0.80/0.01) trong `metrics_lgbm_v4.json`.
- ⚠️ **PRODUCTION override (M3 — ADR 0002): threshold = 0.60**, KHÔNG dùng 0.40 trong
  `metrics json` (rule val không transfer — test precision 0.503 ≈ baseline). Nguồn sự thật
  production: `backend/services/inference.py::PRODUCTION_THRESHOLD` (ghi kèm mỗi prediction
  vào cột `threshold`). Temperature vẫn đọc từ metrics json.

## Holdout 2026 H1 (vô nhiễm — không tune, đọc kỹ)

Áp model + T + threshold đã chốt lên 2026 (3,090 dòng, chưa từng dùng để chọn gì):
- macro-F1 **0.3963** · accuracy 0.4019 · pred-dist di_ngang 29% / giam 42% / tang 28% (méo về giam).
- Tại thr 0.40: precision **0.4125** @ coverage 81.6%.
- **Diễn giải:** chế độ 2026 khác hẳn train (di_ngang 42% vs 53%, thị trường động hơn). class_weight
  balanced + train chế độ di_ngang-cao → gặp 2026 đoán giam/tang nhiều → lệch. **v4 tốt hơn v3 trong
  chế độ tương tự train, NHƯNG chưa bền khi chế độ đổi mạnh.** Đây là regime-shift (đã biết từ v3),
  không phải bug. Dữ kiện cho M10 (drift monitor) + nhắc nhở: threshold production cần rolling.

## Giới hạn (đọc trước khi dùng)

- **Holdout 2026 cho thấy độ bền regime còn yếu** — model chưa chống được chế độ đổi mạnh. Theo dõi
  drift (M10); cân nhắc retrain rolling.
- **Trần feature giá vẫn còn:** acc < baseline (đánh đổi với macro-F1 — đúng bản chất ±1%).
- Sentiment features chưa có (M8 PhoBERT → có thể là v5).
- Threshold production nên chọn lại trên cửa sổ gần hiện tại (rolling), không cố định val 2024.
- KHÔNG phải khuyến nghị đầu tư.

## Load model (backend M5)

```python
import lightgbm as lgb
booster = lgb.Booster(model_file="ml/artifacts/lgbm_model/lgbm_v4.txt")
# sector phải là pandas category cùng tập mức như lúc train; thứ tự cột = feature_names.
probs = booster.predict(X[FEATURE_NAMES])  # [N,3] theo classes_order trong metrics JSON
```

`classes_order`, `feature_names`, `temperature` đọc từ `metrics_lgbm_v4.json` — KHÔNG hardcode.
Threshold production = `backend/services/inference.py::PRODUCTION_THRESHOLD` (0.60, ADR 0002 —
threshold trong metrics json là output của rule trên val, không dùng cho production).
Feature list lấy từ `backend/features/builder.py::LGBM_V4_FEATURES`.
