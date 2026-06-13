"""Auto-label sentiment cho title tin → ml/data/sentiment_labeled.csv (data train PhoBERT).

Cần GEMINI_API_KEY trong .env. Chạy: cd backend && uv run python -m scripts.autolabel_sentiment
  --model gemini-2.0-flash   # đổi model khi 1 model hết quota/NGÀY (free tier)

RESUME: chỉ gán title CHƯA có nhãn hợp lệ trong CSV (free tier giới hạn req/ngày → chạy nhiều
phiên). Output CSV (title, label) — gitignore. GOVERNANCE: sau khi chạy, soát mẫu bằng
`scripts.validate_labels`, sửa nhãn sai chiều trước khi fine-tune.
"""

from __future__ import annotations

import argparse
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

    from config import settings
    from models.database import News, SessionLocal
    from services.sentiment_label import ERROR_LABEL, LABELS, label_titles

    parser = argparse.ArgumentParser(description="Auto-label title sentiment (Gemini→OpenAI)")
    parser.add_argument(
        "--provider",
        choices=("auto", "gemini", "openai"),
        default="auto",
        help="auto = Gemini trước, title nào hết quota → fallback OpenAI",
    )
    parser.add_argument("--model", default=None, help="ghi đè model (theo provider)")
    args = parser.parse_args()

    async with SessionLocal() as session:
        rows = (await session.execute(select(News.title).where(News.title.is_not(None)))).all()
        titles = [t for (t,) in rows if t and t.strip()]
    seen: set[str] = set()  # khử trùng lặp, giữ thứ tự
    titles = [t for t in titles if not (t in seen or seen.add(t))]

    # RESUME: giữ nhãn hợp lệ đã có, chỉ gán title còn thiếu.
    done: dict[str, str] = {}
    if OUTPUT.exists():
        prev = pd.read_csv(OUTPUT).dropna(subset=["title", "label"])
        done = {r.title: r.label for r in prev.itertuples(index=False) if r.label in LABELS}
    todo = [t for t in titles if t not in done]
    print(f"{len(titles)} title (đã khử trùng); {len(done)} đã gán, {len(todo)} cần gán...")

    def _absorb(items: list[str], labels: list[str]) -> int:
        for t, lbl in zip(items, labels, strict=True):
            if lbl in LABELS:
                done[t] = lbl
        return sum(1 for lbl in labels if lbl == ERROR_LABEL)

    # Provider chính.
    primary = "gemini" if args.provider == "auto" else args.provider
    print(f"→ {primary}...")
    n_error = _absorb(todo, await label_titles(todo, provider=primary, model_name=args.model))

    # Fallback: title nào còn ERROR (Gemini hết quota) → thử OpenAI (nếu auto + có key).
    if args.provider == "auto" and primary == "gemini" and settings.openai_api_key:
        retry = [t for t in todo if t not in done]
        if retry:
            print(f"→ fallback OpenAI cho {len(retry)} title Gemini chưa gán...")
            n_error = _absorb(retry, await label_titles(retry, provider="openai"))

    # Ghi theo thứ tự DB (title đã gán), bỏ ERROR.
    out = pd.DataFrame({"title": t, "label": done[t]} for t in titles if t in done)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print(f"\nDone: {len(out)}/{len(titles)} title có nhãn hợp lệ → {OUTPUT}")
    print(f"Phân bố nhãn:\n{out['label'].value_counts()}")
    if n_error:
        print(f"\n⚠️ {n_error} title lỗi (hết quota?) — chạy lại (đổi --model) để gán nốt.")
    print("\n⚠️ GOVERNANCE: chạy `python -m scripts.validate_labels` để soát mẫu trước fine-tune.")


if __name__ == "__main__":
    asyncio.run(main())
