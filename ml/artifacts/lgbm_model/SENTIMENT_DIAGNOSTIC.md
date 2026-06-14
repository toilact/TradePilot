# Sentiment diagnostic (M8 Pha 1) — KẾT QUẢ ÂM TÍNH

**Ngày:** 2026-06-14 · **Notebook:** `ml/notebooks/06_sentiment_diagnostic.py` ·
**Số liệu thô:** `diagnostic_sentiment.json`

## Câu hỏi
Sentiment tin tức (PhoBERT v3, title-only) có nâng được **độ chính xác dự đoán giá T+1** của
LightGBM không? (Mục tiêu B — precision-trên-actionable, KHÔNG phải macro-F1 PhoBERT.)

## Bối cảnh
- `lgbm_v5` (sentiment làm feature) đã FAIL trước đây (macro-F1 0.4236 vs v4 0.4234 = noise),
  nghi do coverage thấp (~3.8% phiên-mã).
- **Pha 1a sector-mapping** nâng coverage **3.8% → 9.6% tổng** (2022: 6→20%, 2025: 12→30%,
  2026: 22→47%) KHÔNG cần cào thêm. → tạo điều kiện thuận lợi nhất để test lại trần tín hiệu.
- **Diagnostic-first:** chứng minh trần trên data hiện có TRƯỚC khi đốt công cào (Pha 2).

## Kết quả — FAIL cả 3 tầng độc lập (news-rows 2022+, 6499 hàng)

| Tầng | Đo gì | Kết quả |
|------|-------|---------|
| **0 — signal** | sentiment_agg ↔ nhãn T+1 | **FAIL**: mean sentiment_agg gần như y hệt 3 lớp (di_ngang 0.340 / giảm 0.327 / tăng 0.336); MI thật 0.178 ≈ MI null-shuffle-trong-ngày 0.162 (**tỉ lệ 1.09** → 91% là artifact "nhớ ngày"); 1-mã TCB MI sụp 0.178→**0.025** (~7×); quintile p(tang) không đơn điệu |
| **1a — feature** | LGBM news-rows có vs không sentiment | **FAIL**: thêm sentiment làm macro-F1 **XẤU đi** (0.390 → 0.367, Δ **−0.023**) trên test out-of-time 2026 |
| **1b — gating (gate quyết định)** | Option C thật: lgbm_v4 đông cứng + override is_actionable theo sentiment_extreme | **FAIL**: baseline precision 0.627@cov0.20; **mọi** τ∈{.5,.6,.7,.8} × 2 chế độ đều tệ hơn — *chỉ-hạ* giảm precision nhẹ (−0.002→−0.007), *hạ+nâng* **sụp precision** (−0.05→−0.08). Kể cả tin PhoBERT cực kỳ tự tin (\|extreme\|≥0.8) cũng không nhận ra dự đoán đúng |

## Kết luận
**Tín hiệu sentiment title-only KHÔNG dự báo hướng giá T+1** — kể cả khi coverage đã đẩy lên
20–47% vùng gần đây và test bằng ĐÚNG cơ chế ship (Option C gating). Giả thuyết cạnh tranh đặt ra
lúc grill ("title-only vốn yếu, bất kể coverage") **thắng**. MI "có vẻ cao" trước đây là artifact
lặp-ngày do sector-mapping (1 tin → N mã cùng ngày cùng điểm), null-test đã bóc trần.

## Quyết định
- **DỪNG M8-sentiment-cho-dự-đoán:** KHÔNG cào dày (Pha 2), KHÔNG wire Option C (Pha 3),
  KHÔNG retrain lgbm_v5. Production giữ **lgbm_v4 giá-thuần** (sentiment vẫn ngoài `LGBM_V4_FEATURES`).
- **GIỮ** thành quả phụ trợ (cải thiện đúng đắn, độc lập với kết luận âm tính):
  - Pha 0: vá leakage 16:00 + dồn tin cuối tuần vào phiên kế (`effective_trading_day`) + cột
    `sentiment_extreme`. `daily_sentiment` giờ đúng hơn (trang lịch sử mã hiển thị sentiment dày hơn).
  - Pha 1a: sector-mapping (coverage 3.8→9.6%) — hạ tầng sẵn nếu sau này thử **sentiment full-article**
    (không phải title-only) hoặc horizon dài hơn T+1.
- **Hướng tương lai khả dĩ** (nếu muốn theo đuổi sentiment tiếp, ngoài phạm vi M8 này): bỏ trần
  title-only → đọc nội dung bài (đụng governance không-tái-xuất-bản), hoặc đổi nhãn/horizon
  (T+3/T+5) để tin có thời gian tác động, hoặc tín hiệu sentiment dạng *biến động* thay vì *hướng*.

> Đúng tinh thần DoD M8 gốc: "kết quả âm tính cũng là kết quả". Diagnostic-first tiết kiệm nhiều
> ngày cào mà chắc chắn ra noise.
