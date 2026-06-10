# Model Card — lgbm_v3

> Template model card cho mọi version sau (đổi số liệu, giữ mục). Quy ước: 1 card + 1 metrics JSON
> mỗi version; weights gitignore (local/registry), card + metrics commit.

## Tóm tắt

| | |
|---|---|
| **model_version** | `lgbm_v3` (đếm global: v0 stub → v1, v2 TFT → v3 LightGBM) |
| **Family** | LightGBM (gradient boosting, multiclass) — `LGBMClassifier` defaults + `class_weight="balanced"`, seed 42 |
| **Ngày train** | 2026-06-10, local Mac (vài giây — không cần Kaggle) |
| **Train script** | `ml/notebooks/04_lgbm_training.py` (chạy lại được 100%: panel CSV + seed cố định) |
| **Artifacts** | `lgbm_v3.txt` (native Booster, gitignore) + `metrics_lgbm_v3.json` (commit) |
| **Kết quả** | **PASS** — macro-F1 **0.4018** ≥ 0.36, cả 3 lớp F1 > 0, không collapse |
| **Trạng thái** | KHÔNG wire frontend (chờ gate M2 — market features). Frontend vẫn `stub_v0`. |

## Vì sao LightGBM thay TFT

`tft_v1` VÀ `tft_v2` (class weight + oversample) đều **collapse** — loss đứng im, model không học
(xem `metrics_tft_v1.json`, `metrics_tft_v2.json` + CONTEXT "Sự thật cần nhớ"). LightGBM proxy lúc
chẩn đoán đạt macro-F1 0.398 trên CÙNG feature + CÙNG split → chuyển model thật sang LightGBM.
v3 tái lập thành công: **0.4018**.

## Data & split (walk-forward, cấm random split)

- Panel: `ml/data/training_panel.csv` (export `backend/scripts/export_training_data.py`) —
  96,233 hàng sau lọc warm-up/label-None, 30 mã VN30, 2008→2026.
- Split THỜI GIAN mốc chung: train < 2024-01-01 (78,173) · val 2024 (7,500) · test ≥ 2025-01-01 (10,560).
- Features (7, y hệt notebook 03 — KHÔNG feature mới ngoài builder): ma7, ma20, rsi14, macd,
  macd_signal, sentiment_agg, news_count. Label ±1% close T+1 (builder chống leakage).

## Kết quả test (out-of-sample ≥ 2025)

- **macro-F1 0.4018** (PASS ≥ 0.36) · accuracy 0.4611 (THAM CHIẾU — baseline luôn-di-ngang 0.5039).
- Per-class F1: di_ngang **0.6148** · giam **0.3032** · tang **0.2875** (cả 3 > 0 — hết collapse).
- Pred-dist: di_ngang 53.4% / giam 23.8% / tang 22.8% (không lớp nào > 80%).
- Feature importance (gain): **rsi14 18420** > macd 13979 > macd_signal 12414 > ma7 10095 > ma20 8736 >
  **sentiment_agg 0.0 = news_count 0.0** (sentiment stub chết — đúng dự đoán, chờ PhoBERT M8).

## Calibration & confidence-gating

- Temperature scaling fit trên val: **T = 0.4185** (< 1 → model underconfident, calibration sharpen).
- Threshold theo rule đã chốt (nhỏ nhất đạt precision ≥ 0.55 & coverage ≥ 20% trên VAL): **0.40**.
- ⚠️ **REGIME SHIFT val→test (dữ kiện quan trọng cho M2/M3):** threshold 0.40 đạt 0.5731 trên val
  nhưng chỉ **0.4765 trên test** (< baseline!) — val 2024 "dễ" hơn test 2025. Threshold chọn trên
  val năm trước KHÔNG transfer.
- **Tin tốt:** trên test VẪN tồn tại vùng gating có giá trị: **thr 0.60 → precision 0.5521 @
  coverage 20.8%** (> baseline 0.5039) — gate M2 "tồn tại threshold ≥0.50/coverage ≥20% trên test"
  ĐẠT. Tín hiệu confidence có thật kể cả với feature giá thuần.
- Frontier đầy đủ (val + test, 0.40→0.80/0.01) trong `metrics_lgbm_v3.json`.

## Giới hạn (đọc trước khi dùng)

- **Trần feature giá thuần:** không thể vừa acc > 50% vừa macro-F1 cao (chẩn đoán 2026-06-10).
  Đập trần = market features (M2) + sentiment thật (M8).
- Sentiment features hiện vô dụng (stub = 0).
- Threshold production phải chọn lại sau M2 (regime shift ở trên).
- KHÔNG phải khuyến nghị đầu tư.

## Load model (backend M5)

```python
import lightgbm as lgb
booster = lgb.Booster(model_file="ml/artifacts/lgbm_model/lgbm_v3.txt")
probs = booster.predict(X[FEATURES])  # [N,3] theo classes_order trong metrics JSON
```

`classes_order`, `feature_names`, `temperature`, `threshold` đọc từ `metrics_lgbm_v3.json` —
KHÔNG hardcode.
