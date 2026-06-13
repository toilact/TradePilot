# Model Card — lgbm_v5 (LightGBM + sentiment THẬT) · KẾT QUẢ ÂM TÍNH

**TL;DR:** v5 = v4 + `sentiment_agg` + `news_count` (sentiment thật từ PhoBERT v3, sau khi re-score
3960 tin). Trên **cùng split** với v4, macro-F1 test **0.4236 vs v4 0.4234** — chênh +0.0002 =
**nhiễu**. Sentiment KHÔNG cải thiện model giá. **Quyết định: KHÔNG wire v5 — giữ v4 production.**
Đây là bước E của M8; governance: kết quả âm tính cũng là kết quả.

## Bối cảnh

M8 thay sentiment stub (toàn 0) bằng PhoBERT thật. Câu hỏi bước E: *sentiment thật có giúp dự đoán
giá T+1 không?* v4 ([MODEL_CARD_lgbm_v4.md](MODEL_CARD_lgbm_v4.md)) cố ý BỎ sentiment vì lúc đó stub
chết. Giờ `daily_sentiment` đã có giá trị thật (PhoBERT v3, re-score 3960 tin, phủ 2012-2026) →
thêm lại 2 cột và đo đóng góp ròng.

## Thiết lập (so táo-táo với v4)

- Feature: `LGBM_V4_FEATURES` + `sentiment_agg` + `news_count` (13 cột). Mọi thứ khác GIỐNG v4.
- Split walk-forward GIỐNG v4: train [2010, 2024) / val 2024 / test 2025 / holdout 2026.
- `LGBMClassifier(class_weight="balanced")`, seed 42, default lightgbm 4.x.
- Panel: `training_panel.csv` re-export sau re-score. Sentiment≠0: train 1452 / val 325 / test 788
  hàng — **thưa** (~2-3% tổng hàng; đa số phiên không tin → sentiment_agg=0 theo bất biến).

## Kết quả

| Model | test macro-F1 | test acc | holdout 2026 macro-F1 | passed (≥0.36, 3 lớp>0) |
|-------|--------------:|---------:|----------------------:|:---:|
| lgbm_v4 (giá thuần) | 0.4234 | 0.4775 | — | ✅ |
| **lgbm_v5 (+sentiment)** | **0.4236** | 0.4743 | 0.3898 | ✅ |

*Lưu ý: v4 mốc là bản committed (panel cũ); v5 trên panel re-export (+90 hàng giá mới). Chênh hàng
<0.1% → không ảnh hưởng kết luận; chênh macro-F1 0.0002 nằm sâu trong nhiễu.*

**Feature importance (gain) — sentiment ở đáy:**

```
vol_20  27549  | vol_5  11237 | ret_5d 7367 | abs_ret_1d 7361 | ret_1d 5781
dist_ma20 5585 | rsi14  4726 | ma_ratio 4720 | macd_signal 4214 | macd 3940 | sector 3531
sentiment_agg  680  ⬅   news_count  48  ⬅   (hai feature THẤP NHẤT)
```

`sentiment_agg` gain 680 ≈ 2.5% của `vol_20`; `news_count` gần như không dùng. Model học chủ yếu từ
**biến động giá** (vol_20 áp đảo) — đúng như v4.

## Vì sao sentiment vô dụng ở đây (chẩn đoán trung thực)

1. **Thưa:** ~97% hàng không có tin → `sentiment_agg=0`, feature không mang tín hiệu xuyên panel.
   Tin chủ yếu 2021+ (CafeF phân trang cạn lịch sử), nhiều mã·phiên train trống.
2. **Sentiment title-only còn nhiễu:** PhoBERT v3 mới macro-F1 0.7155 (chưa qua gate 0.75); nút thắt
   neu. Tín hiệu đầu vào yếu → khó kỳ vọng kéo được model giá.
3. **Quan hệ bản chất yếu:** sentiment tiêu đề gộp-ngày → hướng giá ±1% T+1 là liên hệ vốn lỏng;
   feature giá nội tại (volatility) có trần tín hiệu cao hơn nhiều.

## Quyết định

- **Production GIỮ v4.** Không wire v5 (backend vẫn load `lgbm_v4.txt`).
- Artifact v5 (`lgbm_v5.txt` + `metrics_lgbm_v5.json`) giữ làm bằng chứng kết quả âm tính.

## Khi nào nên thử lại

- **Phủ tin dày hơn:** cào sâu nhiều nguồn/nhiều năm để >20-30% phiên có tin (giờ ~2-3%). Coverage
  là nút thắt số 1, không phải chất lượng PhoBERT.
- **Đổi cách dùng sentiment:** thay vì daily_agg thô → rolling 3-5 ngày, momentum sentiment, hoặc
  chỉ subset mã·giai đoạn có tin dày (đo lift cục bộ thay vì trộn loãng toàn panel).
- **PhoBERT vượt gate 0.75** (giảm nhiễu đầu vào) rồi lặp lại bước E.

## Tái lập

```bash
cd backend && uv run python -m scripts.export_training_data   # panel sentiment thật
MODEL_VERSION=lgbm_v5 DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib" \
  uv run --group inference --with scikit-learn python ../ml/notebooks/04_lgbm_training.py
```

Config `lgbm_v5` ở [04_lgbm_training.py](../../notebooks/04_lgbm_training.py) (CONFIGS).
