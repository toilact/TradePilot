# ruff: noqa: E402  (notebook chia cell — import nằm trong cell tương ứng, cố ý)
# %% [markdown]
# # 02 — Fine-tune PhoBERT sentiment (TradePilot) — BẢN KAGGLE
#
# Fine-tune `vinai/phobert-base` phân loại sentiment tin chứng khoán VN: **pos / neu / neg**.
#
# **CÁCH DÙNG:** copy-paste TỪNG CELL (`# %%`) vào Kaggle Notebook, bật GPU, chạy tuần tự.
# Data: `sentiment_labeled.csv` (title,label) — export từ `scripts/autolabel_sentiment.py`,
# upload lên Kaggle Dataset.
#
# ## ⚠️ GIỚI HẠN DATA
# Data auto-label bằng LLM + crawl giới hạn (vài trăm title) → accuracy chỉ THAM KHẢO. Mục tiêu
# là thông pipeline + có model thay stub score_text, KHÔNG kỳ vọng cao. Sau này nhiều data hơn
# → train lại.
#
# ## ⚠️ WORD SEGMENTATION
# PhoBERT gốc khuyến nghị tách từ bằng VnCoreNLP. Bản này dùng tokenizer TRỰC TIẾP (đơn giản,
# chấp nhận giảm chất lượng). Nếu muốn chuẩn: segment title bằng py_vncorenlp trước khi tokenize.

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
# ## Cell 2 — Đọc data

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

df = pd.read_csv(CSV).dropna(subset=["title", "label"])
df = df[df["label"].isin(["pos", "neu", "neg"])].reset_index(drop=True)
print(f"{len(df)} dòng. Phân bố:\n{df['label'].value_counts()}")

LABELS = ["neg", "neu", "pos"]  # index cố định: neg=0, neu=1, pos=2
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
df["label_id"] = df["label"].map(label2id)

# %% [markdown]
# ## Cell 3 — Split train/val (stratified — phân loại văn bản, KHÔNG walk-forward)

# %%
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)
print(f"train={len(train_df)} val={len(val_df)}")

# %% [markdown]
# ## Cell 4 — Tokenize (PhoBERT)

# %%
from transformers import AutoTokenizer

MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(texts):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=128)


train_enc = tokenize(train_df["title"])
val_enc = tokenize(val_df["title"])


import torch


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
# ## Cell 5 — Fine-tune (GPU)

# %%
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label={i: lbl for lbl, i in label2id.items()}, label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


args = TrainingArguments(
    output_dir="/kaggle/working/phobert_out",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
)
trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

# %% [markdown]
# ## Cell 6 — Đánh giá (data nhỏ → tham khảo)

# %%
from sklearn.metrics import classification_report, confusion_matrix

pred = trainer.predict(val_ds)
y_pred = np.argmax(pred.predictions, axis=-1)
y_true = val_df["label_id"].to_numpy()
print("accuracy:", accuracy_score(y_true, y_pred))
print("confusion_matrix:\n", confusion_matrix(y_true, y_pred, labels=range(3)))
print(classification_report(y_true, y_pred, labels=range(3), target_names=LABELS, zero_division=0))

# %% [markdown]
# ## Cell 7 — Export (download về ml/artifacts/phobert_sentiment/)

# %%
OUT = "/kaggle/working/phobert_sentiment"
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(f"✓ model + tokenizer → {OUT}")
print("⬇️ Download cả thư mục về ml/artifacts/phobert_sentiment/ ; backend score_text load từ đây.")
print(f"label order (id→nhãn): {LABELS}  # neg=0, neu=1, pos=2 — backend map sang [-1,0,1]")
