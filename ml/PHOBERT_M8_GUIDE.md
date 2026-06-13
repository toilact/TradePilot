# M8 — PhoBERT sentiment: hướng dẫn từ autolabel → Kaggle → thay stub

Quy trình thay `score_text` stub (trả 0.0) bằng PhoBERT thật. Các bước **A, C, E** chạy LOCAL;
bước **B** chạy trên **Kaggle GPU** (thủ công). Gate ship: **macro-F1 ≥ 0.75** trên val.

> Trạng thái 2026-06-13: A đã chạy 420/578 title (Gemini free 20 req/ngày/key chặn). Thêm
> `GEMINI_API_KEY_2/_3` vào `backend/.env` để xoay key gán nốt. Lớp `neg` mỏng (59) → cân nhắc
> cào thêm tin tiêu cực trước khi kỳ vọng đạt gate.

## A — Autolabel + soát nhãn (LOCAL)

```bash
cd backend
# Gán nhãn (xoay vòng mọi key trong .env khi 1 key hết quota/ngày). RESUME được — chạy lại
# nhiều ngày, chỉ gán title còn thiếu. Đổi model khi cần:
uv run --group pipeline python -m scripts.autolabel_sentiment
uv run --group pipeline python -m scripts.autolabel_sentiment --model gemini-2.0-flash

# Kiểm định + flag nhãn nghi sai để soát tập trung (GOVERNANCE — bắt buộc trước fine-tune):
uv run python -m scripts.validate_labels
```

Output (đều `gitignore` trong `ml/data/`):
- `sentiment_labeled.csv` — (title, label) data train. CHỈ chứa nhãn hợp lệ (ERROR bị loại).
- `labels_to_review.csv` — nhãn nghi sai chiều (heuristic từ khoá) → mở soát + sửa tay.
- `labels_sample.csv` — mẫu/lớp để spot-check.

Sửa nhãn sai trực tiếp trong `sentiment_labeled.csv` (cột `label`: pos/neu/neg) trước khi upload.

## B — Fine-tune trên Kaggle GPU (THỦ CÔNG)

1. Tạo **Kaggle Dataset** mới, upload `ml/data/sentiment_labeled.csv` (đặt tên vd `tradedataset`).
2. Tạo **Kaggle Notebook**, Settings → Accelerator = **GPU T4**, Add Data = dataset vừa tạo.
3. Mở `ml/notebooks/02_sentiment_finetune.py`, **copy-paste từng cell** (`# %%`) vào notebook,
   chạy tuần tự. (Sửa biến `CSV` ở Cell 2 nếu tên dataset khác — notebook có fallback tự tìm.)
4. Cell 5 in class weight (chống mất cân bằng pos≫neg); Cell 6 train weighted CrossEntropy.
5. **Cell 7 in GATE:** `✅ PASS` nếu macro-F1 ≥ 0.75 và cả 3 lớp F1 > 0; ngược lại `❌ CHƯA ĐẠT`.
   - **KHÔNG ĐẠT** → đừng ship. Cào thêm tin (nhất là tiêu cực) → gán nhãn → train lại. Ghi
     kết quả vào model card (kết quả âm tính cũng là kết quả).
6. **Cell 8** lưu `phobert_sentiment/` (model + tokenizer + `metrics_phobert.json`). Download CẢ
   thư mục về máy local.

## C — Thay stub bằng PhoBERT (LOCAL)

1. Đặt thư mục tải về vào `ml/artifacts/phobert_sentiment/` (chứa `config.json`, `model.safetensors`,
   tokenizer..., `metrics_phobert.json`). Thư mục này `gitignore`.
2. Backend `services/sentiment.py` **tự động** load PhoBERT khi có artifact (degrade về 0.0 nếu
   thiếu). Re-score TOÀN BỘ tin cũ (đang mang 0.0 của stub) + rebuild daily_sentiment mọi mã:
   ```bash
   cd backend
   DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib" \
     uv run --group inference python -m services.sentiment --all
   ```
3. Kiểm tra 5 câu vàng đúng chiều (test tự bật khi có artifact):
   ```bash
   uv run --group inference python -m pytest tests/test_sentiment.py -q
   ```

## D — Crawl thêm tin (tăng coverage + data train)

Hai đường, đều governance (chỉ lưu title+url, `published_at` chuẩn để chống leakage, idempotent):

1. **Theo mã (đòn bẩy chính cho VN30):** `services/crawler.py` lấy ~20 bài mới nhất/mã từ
   `cafef.vn/<symbol>.html`. Chạy vòng 30 mã định kỳ. Để lấy SÂU lịch sử cần thêm phân trang
   archive theo mã (chưa có — việc mở rộng tiếp theo).
2. **Ingest từ firecrawl:** agent cào trang chuyên mục bằng firecrawl → JSON `[{title,url}]` →
   `uv run python -m scripts.ingest_titles ../ml/data/firecrawl_titles.json`. Script decode
   `published_at` từ ID bài CafeF + match mã + persist. **Lưu ý:** trang chuyên mục đa phần tin
   vĩ mô → tỉ lệ khớp VN30 THẤP (thử nghiệm: 37 tin chỉ ~1 khớp mã mới). Hiệu quả cho VN30 kém
   hơn crawl theo mã; hữu ích khi mở rộng scope mã (M11).

Mục tiêu: (a) lớp `neg` dày hơn cho fine-tune, (b) tăng % phiên có `news_count>0`. Sau khi cào →
lặp lại A (autolabel phần mới, RESUME tự bỏ tin đã gán) → B → C.

## E — Retrain LightGBM v5 với sentiment thật (LOCAL)

Sau khi `daily_sentiment` hết toàn-0, thêm `sentiment_agg`/`news_count` vào feature set và train v5,
so frontier với v4 (xem `04_lgbm_training.py`). **Chỉ thay production nếu frontier v5 > v4.** Ghi
model card v5 + kết luận sentiment có/không cải thiện (kể cả kết quả âm tính).

## Bất biến cần giữ
- Chỉ lưu **title + score** (không full text) — governance.
- Điểm sentiment = `p_pos − p_neg` ∈ [-1,1]; backend đọc `id2label` từ config model (không hard-code).
- Backend nguồn sự thật; ml chỉ train+export; dep torch/transformers ở uv group `inference`
  (KHÔNG vào default install / Render).
