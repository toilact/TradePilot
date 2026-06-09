---
name: backend-daily-pipeline
description: Chạy và bảo trì pipeline dự đoán hàng ngày của TradePilot backend (crawl → sentiment → predict → store). Dùng khi làm việc với crawler, scheduler, sentiment inference, hoặc API dự đoán.
---

# Skill: Daily prediction pipeline

Pipeline chạy 1 lần/ngày lúc ~16:00 (sau đóng cửa) cho top 100 mã.

## Thứ tự bước (scheduler.py)
1. `price_fetcher.py` — fetch OHLCV ngày T qua `vnstock`, lưu `price_history`.
2. `crawler.py` — crawl tin mới CafeF + FireAnt, map mã qua `news_stocks`.
3. `sentiment.py` — PhoBERT inference cho tin mới → `news.sentiment_score`.
4. Tổng hợp `daily_sentiment` (trung bình theo ngày/mã; 0 nếu không tin).
5. TFT inference → `predictions` (label + confidence + model_version).
6. Invalidate Redis cache các mã vừa cập nhật.
7. Khi có close T+1 thật → fill `actual_results` để tính accuracy.

## Bất biến (invariants) — không được vi phạm
- Inference ngày T chỉ dùng feature có `published_at ≤ 16:00 phiên T`.
- Nhãn ±1%: Tăng >+1% / Giảm <-1% / Đi ngang [-1%,+1%].
- Một bài báo → nhiều mã qua `news_stocks`, KHÔNG nhồi 1 stock_id vào `news`.

## Chạy thủ công (test)
- Crawl 1 mã: `python -m services.crawler --symbol VCB`
- Trigger full job: gọi hàm scheduler thủ công, kiểm tra `predictions` có dòng mới.

## Lỗi thường gặp
- vnstock rate-limit → retry + backoff.
- CafeF/FireAnt đổi HTML → crawler selector hỏng, kiểm tra trước khi đổ lỗi cho data.
- Tin cũ khó crawl sâu → chấp nhận, sentiment=0 cho ngày trống.
