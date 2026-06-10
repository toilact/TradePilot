# ruff: noqa: E402  (notebook chia cell — import nằm trong cell tương ứng, cố ý)
# %% [markdown]
# # 03 — TFT global training (TradePilot) — BẢN KAGGLE
#
# Train **Temporal Fusion Transformer** (pytorch-forecasting) GLOBAL cho 30 mã VN30 →
# dự đoán phiên T+1: **Tăng / Giảm / Đi ngang** (ngưỡng ±1%).
#
# **CÁCH DÙNG:** copy-paste TỪNG CELL (phân tách bởi `# %%`) vào Kaggle Notebook rồi chạy.
# Cell 1 cài deps, cell 2 đọc CSV, ... cell cuối train + export. KHÔNG cần backend/Supabase —
# mọi feature đã có sẵn trong CSV (export từ `backend/scripts/export_training_data.py`).
#
# ## ⚠️ BASELINE THỰC TẾ ~50% (lớp "đi ngang" đa số), KHÔNG PHẢI 33%
# "Luôn đoán đi ngang" đạt ~50% accuracy trên test. → TFT phải vượt **~50%** mới có giá trị.
#
# ## ⚠️ SENTIMENT HIỆN TOÀN 0 (PhoBERT chưa train) → model học từ feature GIÁ.

# %% [markdown]
# ## Cell 1 — Cài đặt dependencies (chạy 1 lần)

# %%
# Kaggle thường có sẵn torch + lightning. Chỉ cần pytorch-forecasting.
# Nếu lỗi version, thử ghim: pip install pytorch-forecasting==1.0.0
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pytorch-forecasting>=1.0"],
    check=False,
)
print("✓ deps installed")

# %% [markdown]
# ## Cell 2 — Cấu hình + đọc panel CSV
# CSV đã có sẵn: date, symbol, stock_id, close, ma7, ma20, rsi14, macd, macd_signal,
# sentiment_agg, news_count, label.

# %%
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_VERSION = "tft_v1"  # bump mỗi lần retrain (lưu accuracy theo version)
ARTIFACT_DIR = Path("/kaggle/working")  # Kaggle output dir → download sau khi chạy
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Mốc split theo THỜI GIAN (chung mọi mã) — walk-forward, không random.
TRAIN_END = "2024-01-01"  # train: date < TRAIN_END
VAL_END = "2025-01-01"  # val: [TRAIN_END, VAL_END) ; test: >= VAL_END

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]  # fmt: skip

FEATURES = ["ma7", "ma20", "rsi14", "macd", "macd_signal", "sentiment_agg", "news_count"]

# Đường dẫn CSV trên Kaggle. Nếu sai, glob fallback tự tìm file *.csv trong /kaggle/input/.
PANEL_CSV = "/kaggle/input/tradedataset/training_panel.csv"
if not Path(PANEL_CSV).exists():
    found = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    assert found, "Không tìm thấy file CSV nào trong /kaggle/input/ — kiểm tra đã add dataset chưa."
    PANEL_CSV = found[0]
    print(f"⚠️ Dùng CSV tự tìm: {PANEL_CSV}")

raw = pd.read_csv(PANEL_CSV)
raw["date"] = pd.to_datetime(raw["date"])
print(f"Đọc {len(raw)} hàng từ {PANEL_CSV}")
print(raw.head())

# %% [markdown]
# ## Cell 3 — Build panel: lọc + encode stock_id + time_idx
# - Lọc đúng 30 mã VN30, loại warm-up (NaN feature) + label None.
# - `stock_id` = code số CỐ ĐỊNH theo `sorted(VN30)` (ổn định dù thiếu mã).
# - `time_idx` tăng dần per mã (đã sort theo date).

# %%
if "symbol" not in raw.columns:  # fallback CSV cũ chỉ có stock_id = symbol-string
    raw["symbol"] = raw["stock_id"]

panel = raw[raw["symbol"].isin(VN30)].copy()
panel = panel.dropna(subset=FEATURES)
panel = panel[panel["label"].notna()]
panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

codes = pd.Categorical(panel["symbol"], categories=sorted(VN30)).codes
panel["stock_id"] = codes.astype(str)  # group_id dạng str cho TimeSeriesDataSet
panel["time_idx"] = panel.groupby("symbol").cumcount()

# Sanity: symbol ↔ stock_id phải 1-1 (chống misalign entity embedding).
assert panel.groupby("symbol")["stock_id"].nunique().eq(1).all()
assert panel.groupby("stock_id")["symbol"].nunique().eq(1).all()
print(f"Panel: {len(panel)} hàng / {panel['symbol'].nunique()} mã")

# %% [markdown]
# ## Cell 4 — Split theo thời gian (walk-forward, KHÔNG random)

# %%
d = pd.to_datetime(panel["date"])
train = panel[d < TRAIN_END]
val = panel[(d >= TRAIN_END) & (d < VAL_END)]
test = panel[d >= VAL_END]

# Walk-forward assert: train < val < test theo thời gian.
if len(train) and len(val):
    assert pd.to_datetime(train["date"]).max() < pd.to_datetime(val["date"]).min()
if len(val) and len(test):
    assert pd.to_datetime(val["date"]).max() < pd.to_datetime(test["date"]).min()
print(
    f"Split — train={len(train)} (<{TRAIN_END}), "
    f"val={len(val)} ({TRAIN_END}–{VAL_END}), test={len(test)} (>={VAL_END})"
)

# %% [markdown]
# ## Cell 5 — Baseline (cận dưới để so sánh)
# Nếu TFT không vượt "luôn đoán đi ngang", model CHƯA học được gì.

# %%
majority = train["label"].value_counts().idxmax()
acc_majority = float((test["label"] == majority).mean())
acc_flat = float((test["label"] == "di_ngang").mean())
baseline = {
    "majority_class": majority,
    "acc_majority_on_test": round(acc_majority, 4),
    "acc_always_flat_on_test": round(acc_flat, 4),
    "test_label_dist": test["label"].value_counts(normalize=True).round(4).to_dict(),
}
print("Baseline:", json.dumps(baseline, ensure_ascii=False, indent=2))

# %% [markdown]
# ## Cell 6 — Tạo TimeSeriesDataSet + train TFT (GPU)
#
# **⚠️ XÁC MINH API:** pytorch-forecasting classification dùng target categorical + `CrossEntropy` +
# `output_size=3`. API đổi theo version. Nếu cell này lỗi → đọc message + chỉnh `output_size`/loss.
#
# **max_encoder_length=40:** mã ít lịch sử trong test có thể bị silent-drop (xem số sample Cell 7).

# %%
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import CrossEntropy

MAX_ENCODER_LENGTH = 40

training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="label",
    group_ids=["stock_id"],
    static_categoricals=["stock_id"],
    time_varying_unknown_reals=FEATURES,
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=1,  # dự đoán T+1
    target_normalizer=NaNLabelEncoder(),  # target phân loại
    allow_missing_timesteps=True,
    add_relative_time_idx=True,
)
validation = TimeSeriesDataSet.from_dataset(training, val, stop_randomization=True)
train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=2)
val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=2)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=32,
    attention_head_size=2,
    dropout=0.2,
    loss=CrossEntropy(),
    output_size=3,  # 3 lớp — xác minh đúng version
    log_interval=0,
)
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
)
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("✓ train xong")

# %% [markdown]
# ## Cell 7 — Đánh giá out-of-sample (accuracy + confusion + per-class F1)
#
# ALIGNMENT: dùng `predict(return_x=True)` → y_true (decoder_target) ĐÚNG thứ tự với y_pred.
# LABEL ORDER: lấy từ `training.target_normalizer.classes_` (thứ tự THẬT model học).

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

test_ds = TimeSeriesDataSet.from_dataset(training, test, stop_randomization=True)
test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=2)
print(f"Test samples thực sự predict: {len(test_ds)} / {len(test)} hàng test "
      f"(chênh = mã thiếu ≥{MAX_ENCODER_LENGTH} phiên context bị drop)")

# mode="prediction" → CrossEntropy.to_prediction() trả CLASS INDEX (không argmax kép).
out = tft.predict(test_loader, mode="prediction", return_x=True)
y_pred = np.asarray(out.output.cpu()).ravel()
y_true = np.asarray(out.x["decoder_target"].cpu()).ravel()

# Nếu version trả [N,3] probabilities (ndim==2) → bỏ comment dòng argmax dưới:
# y_pred = np.asarray(out.output.cpu()).argmax(axis=-1).ravel()
assert y_pred.ndim == 1, f"predict() trả shape {out.output.shape} — thêm .argmax(-1) (xem comment)."

classes = [str(c) for c in training.target_normalizer.classes_]
idx = list(range(len(classes)))
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=idx).tolist()
report = classification_report(
    y_true, y_pred, labels=idx, target_names=classes, output_dict=True, zero_division=0
)
print(f"TFT accuracy out-of-sample: {acc:.4f}  (baseline {acc_flat:.4f})")
print("labels_order:", classes)
print("confusion_matrix:", cm)

# %% [markdown]
# ## Cell 8 — Export checkpoint + metrics (download từ /kaggle/working)

# %%
ckpt = ARTIFACT_DIR / f"{MODEL_VERSION}.ckpt"
trainer.save_checkpoint(ckpt)  # đầy đủ hparams + weights + dataset params
print(f"✓ checkpoint → {ckpt}")
# Backend load lại: TemporalFusionTransformer.load_from_checkpoint(ckpt)

metrics = {
    "model_version": MODEL_VERSION,
    "baseline": baseline,
    "tft": {
        "accuracy": round(float(acc), 4),
        "labels_order": classes,  # backend map index→nhãn theo thứ tự này
        "confusion_matrix": cm,
        "per_class": report,
    },
}
(ARTIFACT_DIR / f"metrics_{MODEL_VERSION}.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2)
)
print(f"✓ metrics → {ARTIFACT_DIR / f'metrics_{MODEL_VERSION}.json'}")
print("\n⬇️ Download 2 file trên từ tab Output của Kaggle về ml/artifacts/tft_model/")
