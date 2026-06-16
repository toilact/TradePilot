# Model Card — TFT v1 / v2 (LƯU TRỮ — COLLAPSED, đã bỏ)

> Hồi tố 2026-06-16 (M10 model governance). Đây là **bằng chứng âm tính**: TFT không học được
> trên feature giá thuần của bài toán này. Model production hiện tại là LightGBM — xem
> `ml/artifacts/lgbm_model/MODEL_CARD_lgbm_v4.md`.

## Tóm tắt

| | tft_v1 | tft_v2 |
|---|---|---|
| Family | Temporal Fusion Transformer (pytorch-forecasting), 1 model global, `stock_id` static | nt |
| Train compute | Kaggle GPU T4 | nt |
| accuracy (test) | 0.479 | 0.478 |
| **macro-F1 (test)** | **0.2159** | **0.2208** |
| F1 `di_ngang` / `giam` / `tang` | 0.648 / **0.0** / **0.0** | 0.647 / **0.0** / ~0.0 |
| pred-dist | **100% di_ngang** (collapse hoàn toàn) | ~99.5% di_ngang |
| Kết luận | ❌ COLLAPSE | ❌ COLLAPSE |

Baseline luôn-đi-ngang trên test: acc 0.5039 (di_ngang 50.4% / tang 25.5% / giam 24.2%).
Pass criteria: macro-F1 ≥ 0.36 + cả 3 lớp F1 > 0 + không lớp nào > 80% pred-dist → **CẢ HAI ĐỀU TRƯỢT**.

## Triệu chứng

- **v1:** đoán **100% `di_ngang`** (confusion matrix: 2 cột `giam`/`tang` toàn 0) → macro-F1 0.216.
- **v2 (chống collapse):** CrossEntropy class-weight nghịch đảo tần suất (power 1.0) + tripwire pred-dist
  > 80% → fallback Cách C oversample. **Vẫn collapse**: `giam` F1 = 0, `tang` recall ~0.008, loss đứng im.
- Kết luận: weight không bị "nuốt" — optimization **kẹt**, model không học gì từ chuỗi này.

## Vì sao bỏ TFT (quyết định theo bằng chứng — 2026-06-10)

Cùng feature/split, **LightGBM `class_weight=balanced` ra macro-F1 0.398 > 0.36** (cả 3 lớp sống) →
**tín hiệu CÓ TỒN TẠI, TFT không khai thác được**. Chốt: bỏ TFT, chuyển model thật sang LightGBM
(`lgbm_v3` 0.4018 → `lgbm_v4` 0.4234). Chi tiết: `MODEL_CARD_lgbm_v3/v4.md`, CONTEXT.md, plan.md (M1).

## Artifacts (giữ làm bằng chứng, KHÔNG dùng inference)

- `metrics_tft_v1.json`, `metrics_tft_v2.json` (commit) · `tft_v1.ckpt`, `tft_v2.ckpt` (gitignore).
- Notebook: `ml/notebooks/03_tft_training.py` (LƯU TRỮ).

## Bài học (governance)

- Đo macro-F1 + pred-dist + F1 từng lớp, KHÔNG dùng accuracy thô (baseline đi-ngang đã ~0.50).
- Một model deep "đứng im loss" trên data có tín hiệu (chứng minh bằng baseline cây) → đổi họ model,
  đừng cố cứu bằng tinh chỉnh. Drift monitor (M10) cũng bắt pred-dist 1 lớp > 70% = tiền-collapse vì lý do này.
