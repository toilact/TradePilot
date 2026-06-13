# Model Card — PhoBERT Sentiment (tin tài chính VN)

Phân loại sentiment tiêu đề tin theo **góc-giá-T+1** (tích cực/trung lập/tiêu cực), fine-tune
`vinai/phobert-base` 3 lớp. Backend [services/sentiment.py](../../../backend/services/sentiment.py)
chấm điểm = `p_pos − p_neg ∈ [-1,1]` rồi gộp `daily_sentiment` (feature cho LightGBM).

> **Trạng thái: CHƯA SHIP (standalone gate).** Gate **macro-F1 ≥ 0.75 trên val** (xem ml/CLAUDE.md).
> v1/v2/v3 đều chưa đạt nhưng đang lên; **v3 = bản dùng hiện tại** trong artifact, đã re-score
> `daily_sentiment` thật (**3960 tin / 30 mã** — sau deep-crawl) để thử LightGBM v5 (bước E).
> LightGBM v4 production chưa dùng sentiment → tích hợp này KHÔNG ảnh hưởng dự đoán đang chạy.
> Quyết định ship sentiment thật = qua **frontier LightGBM v5 vs v4**, không phải chỉ gate standalone.

## Kết quả theo version (val)

| Version | n_train | n_val | Accuracy | macro-F1 | F1 neg | F1 neu | F1 pos | Gate 0.75 |
|---------|--------:|------:|---------:|---------:|-------:|-------:|-------:|:---------:|
| phobert_v1 | 466 | 117 | 0.7179 | 0.6468 | 0.4615 | 0.6914 | 0.7874 | ❌ |
| phobert_v2 | 536 | 134 | 0.7164 | 0.7045 | 0.6441 | 0.7174 | 0.7521 | ❌ |
| **phobert_v3** | **1136** | **284** | **0.7289** | **0.7155** | **0.6947** | **0.6442** | **0.8075** | ❌ |

`train_dist` v3: pos 663 / neu 559 / neg 198 (data gấp đôi qua `crawl_deep.py` + tự label).
`label_order = [neg, neu, pos]`, `id2label {0:neg,1:neu,2:pos}`.

**Confusion matrix v3** (hàng = thật, cột = đoán; thứ tự neg/neu/pos):

```
        neg  neu  pos
neg  [  33    5    1 ]
neu  [  21   67   24 ]
pos  [   2   24  107 ]
```

## Tiến triển v1 → v2 → v3

- **neg đã chữa xong:** F1 0.46 → 0.64 → **0.69** (recall 0.85) nhờ gấp đôi data (neg 99 → 198).
- macro-F1 0.6468 → 0.7045 → **0.7155** (early-stop epoch 6/8). Plateau ~0.71-0.72.
- **Nút thắt dịch sang `neu`** (F1 0.64, recall 0.60): neu thật 112 → 21 lọt neg, 24 lọt pos. Do
  class-weight nay over-predict neg (neg precision chỉ 0.59). neu là lớp mơ-hồ, ranh giới với
  pos/neg vốn nhập nhằng → khó cải thiện chỉ bằng thêm data.
- Có thể đang gần **trần của sentiment chỉ-từ-tiêu-đề** (v1 cố ý KHÔNG word-segmentation).

## Golden sentences (local) — v3 PASS

5/5 đúng chiều (v2 trước fail câu "HPG tăng trần/mua ròng" = −0.29; **v3 = +0.88**):

| Nhãn kỳ vọng | Điểm v3 | Câu |
|---|---:|---|
| pos | +0.9232 | Vietcombank báo lãi kỷ lục, cổ phiếu lập đỉnh |
| pos | +0.8791 | HPG tăng trần, khối ngoại mua ròng mạnh |
| neg | −0.9173 | Cổ phiếu BIDV giảm sàn, nhà đầu tư bán tháo |
| neg | −0.8704 | Doanh nghiệp thua lỗ nặng, nợ xấu phình to |
| neu | +0.6894 | FPT chốt danh sách cổ đông trả cổ tức (neu nhưng nghiêng pos vì "trả cổ tức") |

`test_phobert_golden_sentences_directionality` PASS (9 passed). Test `skipif` khi thiếu artifact.

## Next step

1. ~~LightGBM v5 (bước E)~~ **ĐÃ LÀM → kết quả ÂM TÍNH:** v5 (+sentiment thật) macro-F1 test
   0.4236 vs v4 0.4234 = nhiễu; sentiment ở đáy feature importance. KHÔNG wire — giữ v4. Lý do:
   tin THƯA (~2-3% phiên có tin) + sentiment title-only còn nhiễu. Xem
   [MODEL_CARD_lgbm_v5.md](../lgbm_model/MODEL_CARD_lgbm_v5.md). → Muốn sentiment có ích cho giá:
   **cào tin dày hơn nhiều** (coverage là nút thắt, không phải gate PhoBERT).
2. Nếu muốn đẩy sentiment qua gate 0.75: bottleneck giờ là **neu** (không phải neg) — làm dày/rõ
   neu hoặc giảm cường độ class-weight (neg nay không còn hiếm); cân nhắc word-segmentation (đánh
   đổi train/serve skew + JVM, xem notebook).

## Tái lập

- Notebook: [ml/notebooks/02_sentiment_finetune.py](../../notebooks/02_sentiment_finetune.py)
  (Kaggle GPU T4, class-weighted CrossEntropy, early-stop theo F1 macro val).
- Data train: `ml/data/sentiment_labeled.csv` (gitignore) — autolabel LLM + soát tay.
- Quy trình đầy đủ: [ml/PHOBERT_M8_GUIDE.md](../../PHOBERT_M8_GUIDE.md).
