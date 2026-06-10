"""Auto-label sentiment cho title tin → ml/data/sentiment_labeled.csv (data train PhoBERT).

Cần GEMINI_API_KEY trong .env. Chạy: cd backend && uv run python -m scripts.autolabel_sentiment

Output CSV (title, label) — gitignore. GOVERNANCE: sau khi chạy, NGƯỜI review mẫu ~20 nhãn,
sửa thủ công nếu sai chiều rõ ràng trước khi fine-tune.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

OUTPUT = _REPO / "ml" / "data" / "sentiment_labeled.csv"


async def main() -> None:
    from sqlalchemy import select

    from models.database import News, SessionLocal
    from services.sentiment_label import label_titles

    async with SessionLocal() as session:
        rows = (await session.execute(select(News.title).where(News.title.is_not(None)))).all()
        titles = [t for (t,) in rows if t and t.strip()]
    # khử trùng lặp, giữ thứ tự
    seen: set[str] = set()
    titles = [t for t in titles if not (t in seen or seen.add(t))]
    print(f"Có {len(titles)} title (đã khử trùng). Đang gọi Gemini...")

    labels = await label_titles(titles)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"title": titles, "label": labels})
    df.to_csv(OUTPUT, index=False)

    print(f"\nDone: {len(df)} dòng → {OUTPUT}")
    print(f"Phân bố nhãn:\n{df['label'].value_counts()}")
    print("\n⚠️ GOVERNANCE: hãy mở CSV review ~20 mẫu, sửa nhãn sai chiều trước khi fine-tune.")


if __name__ == "__main__":
    asyncio.run(main())
