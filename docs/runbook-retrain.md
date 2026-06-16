# Runbook — Retrain model LightGBM (M10 — model governance)

Quy trình chuẩn để retrain + lên đời model production. Đủ chi tiết để **1 phiên agent mới
chạy được mà không cần hỏi**. Train LOCAL (vài giây) — KHÔNG cần Kaggle cho LightGBM.

> Bất biến: backend nguồn sự thật · `ml/` chỉ train+export · walk-forward, cấm random split ·
> feature dùng chung `backend/features/builder.py` (chống train/serve skew) · Conventional Commits + PR.

## Khi nào retrain (trigger)

1. **Định kỳ:** hàng tháng.
2. **Drift alert:** Telegram "⚠️ DRIFT — CẦN RETRAIN" từ `services/drift_monitor.py` (bước 7 pipeline).
   Tín hiệu (rolling 30 phiên, version production): precision-dám-đoán < 0.50 **HOẶC** 1 lớp > 70%
   pred-dist (tiền-collapse) **HOẶC** coverage < 10%. Xem `backend/services/drift_monitor.py`.
3. **Watchdog freshness** (`/healthz?check=freshness` → 503) KHÔNG phải trigger retrain — đó là báo
   "Mac quên chạy pipeline", chỉ cần chạy lại pipeline.

## Các bước

### 1. Export panel training mới (LOCAL, đọc Supabase)
```bash
cd backend && uv run --group pipeline python -m scripts.export_training_data
# → ml/data/training_panel.csv (cột = features.builder.LGBM_V4_FEATURES + label)
```

### 2. Đặt version mới + config train
- Đặt tên version kế tiếp theo đếm GLOBAL monotonic: `lgbm_v4` → `lgbm_v5` (v0 stub, v1/v2 TFT, v3+ LightGBM).
- Thêm 1 entry vào `CONFIGS` trong `ml/notebooks/04_lgbm_training.py` (copy từ `lgbm_v4`, đổi feature/param nếu có).

### 3. Train LOCAL
```bash
cd /Users/chithanhdaica/Documents/TradePilot
# Mac Apple Silicon: lightgbm cần libomp (Homebrew). Chỉnh path theo cài đặt của bạn.
export DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib"   # hoặc /opt/homebrew/opt/libomp/lib
MODEL_VERSION=lgbm_v5 uv run --with lightgbm --with scikit-learn --with pandas --with scipy \
    python ml/notebooks/04_lgbm_training.py
# → ml/artifacts/lgbm_model/lgbm_v5.txt (gitignore) + metrics_lgbm_v5.json (commit)
```

### 4. So frontier + quyết định ship
Mở `ml/artifacts/lgbm_model/metrics_lgbm_v5.json`, đối chiếu version hiện tại:
- **Pass criteria (KHÔNG dùng "acc > 50%"):** macro-F1 ≥ 0.36 · cả 3 lớp F1 > 0 · không lớp nào > 80% pred-dist.
- **Gate lên đời:** macro-F1 mới ≥ macro-F1 production VÀ `frontier_test` tồn tại threshold đạt
  precision ≥ 0.55 & coverage ≥ 20% (fallback knee precision ≥ 0.50). Threshold/temperature tune trên VAL.
- **Kết quả âm tính cũng ghi nhận** (như TFT v1/v2, sentiment M8) — không ship thì viết model card nêu lý do.

### 5. Viết model card
`ml/artifacts/lgbm_model/MODEL_CARD_lgbm_v5.md` — template = `MODEL_CARD_lgbm_v4.md` (split, baseline,
feature importance, calibration T + threshold production, holdout, bài học).

### 6. Bump production (chỉ khi qua gate bước 4)
- `backend/services/inference.py`: `MODEL_VERSION = "lgbm_v5"` (đọc `lgbm_v5.txt` + `metrics_lgbm_v5.json`;
  `PRODUCTION_THRESHOLD` cập nhật theo frontier mới nếu đổi).
- `load_metadata` tự assert `feature_names == LGBM_V4_FEATURES` — nếu đổi feature, sync builder TRƯỚC.
- Cập nhật badge model trong `README.md` + `CONTEXT.md`.

### 7. PR
Branch `feat/retrain-lgbm-v5` → commit `feat(ml): retrain lgbm_v5 ...` → CI 4 job xanh → review → merge.
Predictions production tự đổi sang version mới ở phiên pipeline kế tiếp (stub/version cũ giữ lịch sử,
bị non-stub đè khi trùng ngày — xem `read_api._pred_order`).

## Registry
`ml/artifacts/lgbm_model/` là registry: mỗi version có `metrics_lgbm_vN.json` (commit) + `MODEL_CARD_lgbm_vN.md`
+ `lgbm_vN.txt` (gitignore — chỉ máy train). TFT collapse v1/v2 lưu ở `ml/artifacts/tft_model/`,
stub_v0 ở `ml/artifacts/MODEL_CARD_stub_v0.md`. Đủ cho solo — KHÔNG cần MLflow.

## Sau khi retrain
Theo dõi `/accuracy` (chart rolling theo version) + drift alert vài phiên để xác nhận version mới
không drift/collapse trên data live.
