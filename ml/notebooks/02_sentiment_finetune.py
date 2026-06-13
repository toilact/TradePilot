# ruff: noqa: E402  (notebook chia cell — import nằm trong cell tương ứng, cố ý)
# %% [markdown]
# # 02 — Fine-tune PhoBERT sentiment (TradePilot) — BẢN KAGGLE
#
# Fine-tune `vinai/phobert-base` phân loại sentiment tin chứng khoán VN: **pos / neu / neg**.
#
# **CÁCH DÙNG:** copy-paste TỪNG CELL (`# %%`) vào Kaggle Notebook, bật GPU (T4), chạy tuần tự.
# Data: `sentiment_labeled.csv` (title,label) từ `backend/scripts/autolabel_sentiment.py` (đã
# soát qua `validate_labels.py`), upload lên Kaggle Dataset.
#
# ## ⚠️ GIỚI HẠN DATA (đọc trước khi kỳ vọng)
# Data 583 title, label theo rubric `ml/data/LABELING_RUBRIC.md` (góc A: tác động giá mã T+1;
# ngưỡng bảo thủ → mặc định neu). **Mất cân bằng**: pos 305 / neu 216 / **neg 62** (~10%). Đã xử lý
# bằng class weight, nhưng lớp `neg` mỏng → macro-F1 có thể KHÔNG đạt gate 0.75. Kết quả âm tính
# CŨNG là kết quả (ghi model card) — khi đó: cào thêm tin TIÊU CỰC (M8 part D) rồi train lại,
# KHÔNG nới rubric để ép neg (đầu độc nhãn). Với neg ~12 mẫu ở val (20%), macro-F1 nhiễu mạnh —
# cân nhắc Stratified K-Fold (Cell 3 có note) để ước lượng ổn định hơn.
#
# ## ⚠️ WORD SEGMENTATION (quyết định v1: KHÔNG segment)
# PhoBERT gốc khuyến nghị tách từ bằng VnCoreNLP. **v1 CỐ Ý tokenize TRỰC TIẾP** (cả train lẫn
# serve `backend/services/sentiment.py`) → KHÔNG train/serve skew, KHÔNG kéo JVM vào deploy path.
# Đánh đổi: dưới chất lượng tối ưu vài điểm F1. FUTURE WORK: nếu thêm py_vncorenlp thì PHẢI segment
# ở CẢ inference (sentiment.py), nếu không sẽ skew. Bottleneck hiện tại là lượng data + neg mỏng,
# không phải segmentation.
#
# ## Export khớp backend
# `services/sentiment.py` đọc `model.config.id2label` để tìm index pos/neg (điểm = p_pos − p_neg).
# Cell export lưu id2label/label2id qua `save_pretrained` → backend KHÔNG phụ thuộc thứ tự cứng.

# %% [markdown]
# ## Cell 1 — Cài deps

# %%
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40", "scikit-learn"],
    check=False,
)
print("✓ deps")

# %% [markdown]
# ## Cell 2 — Đọc data + làm sạch

# %%
import glob
from pathlib import Path

import pandas as pd

CSV = "/kaggle/input/tradedataset/sentiment_labeled.csv"  # đổi theo dataset của bạn
if not Path(CSV).exists():
    found = glob.glob("/kaggle/input/**/sentiment_labeled.csv", recursive=True)
    assert found, "Không tìm thấy sentiment_labeled.csv trong /kaggle/input/"
    CSV = found[0]
    print(f"⚠️ Dùng CSV tự tìm: {CSV}")

LABELS = ["neg", "neu", "pos"]  # index cố định: neg=0, neu=1, pos=2
df = pd.read_csv(CSV).dropna(subset=["title", "label"])
df["title"] = df["title"].astype(str).str.strip()
df = df[(df["title"] != "") & df["label"].isin(LABELS)]  # loại ERROR / rỗng
df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
print(f"{len(df)} dòng sạch (đã bỏ ERROR/rỗng/trùng). Phân bố:\n{df['label'].value_counts()}")
assert df["label"].nunique() == 3, "Cần đủ 3 lớp pos/neu/neg để train."

label2id = {lbl: i for i, lbl in enumerate(LABELS)}
df["label_id"] = df["label"].map(label2id)

# %% [markdown]
# ## Cell 2b — EDA + kiểm tra chất lượng data (text, KHÔNG feature engineering)
#
# Phân loại văn bản → model tự học đặc trưng; KHÔNG tạo cột feature, KHÔNG leak thời gian
# (text tĩnh). EDA ở đây = phân bố lớp, độ dài tiêu đề, và **kiểm rò rỉ nhãn** (cùng tiêu đề
# nhưng khác nhãn = mâu thuẫn nhãn → phải xử lý TRƯỚC split, nếu không train/val nhiễu).

# %%
# 1) Phân bố lớp (mất cân bằng → class weight ở Cell 5 + 6).
dist = df["label"].value_counts()
print("Phân bố lớp:")
for lbl in LABELS:
    n = int(dist.get(lbl, 0))
    print(f"  {lbl:3s}: {n:4d} ({n / len(df) * 100:5.1f}%)")
minority = dist.min()
if minority < 80:
    print(f"⚠️ Lớp hiếm nhất {minority} mẫu (<80) — gate macro-F1 0.75 rủi ro; "
          "cào thêm tin lớp đó thay vì nới rubric.")

# 2) Độ dài tiêu đề (ký tự + từ) theo lớp — sanity check, chọn max_length tokenizer.
df["n_char"] = df["title"].str.len()
df["n_word"] = df["title"].str.split().map(len)
print("\nĐộ dài tiêu đề (ký tự / từ) theo lớp:")
print(df.groupby("label")[["n_char", "n_word"]].agg(["mean", "max"]).round(1))
print(f"Tiêu đề dài nhất: {df['n_word'].max()} từ "
      f"(max_length=128 token ở Cell 4 phủ đủ — title ngắn).")

# 3) RÒ RỈ NHÃN: cùng tiêu đề (đã chuẩn hoá) nhưng gán khác nhãn → mâu thuẫn.
#    Đã drop_duplicates theo title thô ở Cell 2; check thêm sau khi chuẩn hoá khoảng trắng/hoa-thường.
_norm = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
conflict = df.groupby(_norm)["label"].nunique()
bad = conflict[conflict > 1].index
print(f"\nTiêu đề (chuẩn hoá) trùng nhưng KHÁC nhãn: {len(bad)}")
if len(bad):
    print(df[_norm.isin(bad)][["title", "label"]])
    print("⚠️ Xử lý mâu thuẫn nhãn trên TRƯỚC khi train (sửa nhãn trong sentiment_labeled.csv).")
else:
    print("✓ Không có mâu thuẫn nhãn.")

df = df.drop(columns=["n_char", "n_word"])  # cột EDA tạm, không đưa vào train

# %% [markdown]
# ## Cell 3 — Split train/val (stratified — phân loại văn bản, KHÔNG walk-forward)
#
# Text tĩnh → split NGẪU NHIÊN stratified (không walk-forward; walk-forward chỉ cho time-series
# giá). `neg` mỏng (~62) → val chỉ ~12 mẫu neg, macro-F1 1-lần-split nhiễu. Nếu muốn ước lượng ổn
# định hơn: thay bằng `StratifiedKFold(n_splits=5)` rồi trung bình macro-F1 (đắt hơn 5× GPU). Bản
# này giữ 1-lần-split cho đơn giản; đọc macro-F1 val như ước lượng có sai số, KHÔNG con số tuyệt đối.

# %%
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)
print(f"train={len(train_df)} val={len(val_df)}")
print(f"train dist:\n{train_df['label'].value_counts()}")
print(f"val dist:\n{val_df['label'].value_counts()}")
if val_df["label"].value_counts().min() < 5:
    print("⚠️ Lớp val < 5 mẫu — macro-F1 nhiễu mạnh, coi như tham khảo (cân nhắc K-Fold).")

# %% [markdown]
# ## Cell 4 — Tokenize (PhoBERT)

# %%
import torch
from transformers import AutoTokenizer

MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(texts):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=128)


train_enc = tokenize(train_df["title"])
val_enc = tokenize(val_df["title"])


class SentDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


train_ds = SentDataset(train_enc, train_df["label_id"])
val_ds = SentDataset(val_enc, val_df["label_id"])

# %% [markdown]
# ## Cell 5 — Class weights (chống mất cân bằng pos ≫ neg)

# %%
import numpy as np

counts = train_df["label_id"].value_counts().sort_index().to_numpy()
# inverse-frequency: lớp hiếm có trọng số lớn → model không bỏ qua neg.
class_weights = counts.sum() / (len(LABELS) * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("class_weights (neg, neu, pos):", class_weights.tolist())

# %% [markdown]
# ## Cell 6 — Fine-tune (GPU, weighted CrossEntropy)

# %%
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={i: lbl for lbl, i in label2id.items()},
    label2id=label2id,
)


class WeightedTrainer(Trainer):
    """CrossEntropy có class weight — chống collapse về lớp đa số (pos)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits.view(-1, 3), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


args = TrainingArguments(
    output_dir="/kaggle/working/phobert_out",
    num_train_epochs=6,  # data nhỏ → train lâu hơn chút, eval theo epoch để theo dõi overfit
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
)
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

# %% [markdown]
# ## Cell 7 — Đánh giá + GATE pass/fail (macro-F1 ≥ 0.75)

# %%
from sklearn.metrics import classification_report, confusion_matrix

pred = trainer.predict(val_ds)
y_pred = np.argmax(pred.predictions, axis=-1)
y_true = val_df["label_id"].to_numpy()

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(3))

print("accuracy:", round(acc, 4))
print("macro_f1:", round(macro_f1, 4))
print("confusion_matrix:\n", confusion_matrix(y_true, y_pred, labels=range(3)))
print(classification_report(y_true, y_pred, labels=range(3), target_names=LABELS, zero_division=0))

GATE = 0.75
all_classes_alive = bool((per_class_f1 > 0).all())
passed = bool(macro_f1 >= GATE and all_classes_alive)
print(f"\n{'✅ PASS' if passed else '❌ CHƯA ĐẠT'} — macro-F1 {macro_f1:.4f} (gate {GATE}), "
      f"3 lớp F1>0: {all_classes_alive}")
if not passed:
    print("→ KHÔNG ship. Cào thêm tin (nhất là tiêu cực) + gán nhãn, train lại. Ghi model card.")

# %% [markdown]
# ## Cell 8 — Export + metrics (chỉ chạy khi PASS, hoặc khi cố ý lưu baseline)

# %%
import json

OUT = "/kaggle/working/phobert_sentiment"
model.save_pretrained(OUT)  # lưu cả id2label/label2id → backend đọc để map pos/neg
tokenizer.save_pretrained(OUT)

metrics = {
    "model_version": "phobert_v1",
    "base_model": MODEL_NAME,
    "n_train": int(len(train_df)),
    "n_val": int(len(val_df)),
    "gate": GATE,
    "passed": passed,
    "accuracy": round(float(acc), 4),
    "macro_f1": round(float(macro_f1), 4),
    "per_class_f1": {
        lbl: round(float(f), 4) for lbl, f in zip(LABELS, per_class_f1, strict=True)
    },
    "label_order": LABELS,  # neg=0, neu=1, pos=2 — backend map p_pos − p_neg
    "train_dist": {k: int(v) for k, v in train_df["label"].value_counts().items()},
}
with open(f"{OUT}/metrics_phobert.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"✓ model + tokenizer + metrics → {OUT}")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("\n⬇️ Download CẢ thư mục phobert_sentiment → ml/artifacts/phobert_sentiment/ (gitignore).")
print("   Backend score_text tự load từ đó; re-score: uv run --group inference "
      "python -m services.sentiment --all")
