"""Kiểm định chất lượng sentiment_labeled.csv trước khi fine-tune PhoBERT (governance).

Pass tự động (KHÔNG thay người review, mà giúp người review nhanh + đúng chỗ):
  1. Thống kê phân bố lớp + tỉ lệ mất cân bằng → cảnh báo nếu lớp thiểu số quá mỏng.
  2. Flag nhãn NGHI SAI bằng heuristic từ khoá (vd title có "lỗ/giảm sàn" mà gán pos) →
     ml/data/labels_to_review.csv để người soát tập trung.
  3. Trích mẫu ngẫu nhiên mỗi lớp → ml/data/labels_sample.csv để spot-check.

Chạy: cd backend && uv run python -m scripts.validate_labels
Đây là HEURISTIC hỗ trợ — quyết định nhãn cuối là của người (đúng plan M8 governance).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "ml" / "data"
CSV = _DATA / "sentiment_labeled.csv"
REVIEW = _DATA / "labels_to_review.csv"
SAMPLE = _DATA / "labels_sample.csv"

# Từ khoá chiều rõ ràng (lowercase, không dấu trùng khớp tiếng Việt có dấu sẵn).
NEG_WORDS = (
    "giảm sàn",
    "bán mạnh",
    "bán tháo",
    "thua lỗ",
    "lỗ",
    "rủi ro",
    "lao dốc",
    "giảm sâu",
    "mất giá",
    "thoái vốn",
    "vi phạm",
    "xử phạt",
    "phá sản",
    "nợ xấu",
    "bị bán",
    "đỏ lửa",
    "giảm điểm",
    "sụt",
    "âm",
)
POS_WORDS = (
    "lãi kỷ lục",
    "lập đỉnh",
    "phá đỉnh",
    "tăng trưởng",
    "vượt đỉnh",
    "kỷ lục",
    "mua ròng",
    "bứt phá",
    "khởi sắc",
    "lãi lớn",
    "tăng trần",
    "nổi sóng",
    "lợi nhuận kỷ lục",
    "thăng hoa",
)


def _suspect(title: str, label: str) -> str | None:
    """Trả lý do nếu nhãn nghi sai chiều, ngược lại None."""
    t = title.lower()
    has_neg = any(w in t for w in NEG_WORDS)
    has_pos = any(w in t for w in POS_WORDS)
    if label == "pos" and has_neg and not has_pos:
        return "gán pos nhưng có từ tiêu cực"
    if label == "neg" and has_pos and not has_neg:
        return "gán neg nhưng có từ tích cực"
    if label == "neu" and (has_neg ^ has_pos):
        return "gán neu nhưng có từ chiều rõ"
    return None


def main() -> None:
    if not CSV.exists():
        sys.exit(f"Chưa có {CSV} — chạy scripts.autolabel_sentiment trước.")
    df = pd.read_csv(CSV).dropna(subset=["title", "label"])
    n = len(df)
    dist = df["label"].value_counts()
    print(f"=== {n} nhãn hợp lệ ===")
    for lbl in ("pos", "neu", "neg"):
        c = int(dist.get(lbl, 0))
        print(f"  {lbl}: {c:4d} ({c / n * 100:4.1f}%)")
    minc, maxc = int(dist.min()), int(dist.max())
    print(f"  mất cân bằng (max/min): {maxc / max(minc, 1):.1f}x")

    print("\n=== Đánh giá đủ/thiếu (heuristic) ===")
    if minc < 100:
        print(f"  ⚠️ lớp mỏng nhất {minc} < 100 — rủi ro KHÓ đạt gate macro-F1 ≥ 0.75.")
    if maxc / max(minc, 1) > 3:
        print("  ⚠️ mất cân bằng > 3x — cân nhắc class weight / oversample / cào thêm lớp thiểu số.")
    if minc >= 100 and maxc / max(minc, 1) <= 3:
        print("  ✓ phân bố tạm ổn cho fine-tune thử.")

    # Flag nghi sai.
    df["suspect"] = [_suspect(t, lbl) for t, lbl in zip(df["title"], df["label"], strict=True)]
    flagged = df[df["suspect"].notna()]
    flagged[["title", "label", "suspect"]].to_csv(REVIEW, index=False)
    print(f"\n=== {len(flagged)} nhãn NGHI SAI → {REVIEW} (soát + sửa tay) ===")
    for r in flagged.head(15).itertuples(index=False):
        print(f"  [{r.label}] {r.suspect}: {r.title[:70]}")

    # Mẫu ngẫu nhiên mỗi lớp để spot-check.
    sample = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 15), random_state=42), include_groups=True
    )
    sample[["title", "label"]].to_csv(SAMPLE, index=False)
    print(f"\n=== {len(sample)} mẫu spot-check → {SAMPLE} ===")
    print("\nSau khi soát/sửa REVIEW + SAMPLE, dùng CSV để fine-tune (notebook 02).")


if __name__ == "__main__":
    main()
