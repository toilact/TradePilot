# ruff: noqa: E402  (notebook chia cell — import nằm trong cell tương ứng, cố ý)
# %% [markdown]
# # 03 — TFT global training (TradePilot)
#
# Train **Temporal Fusion Transformer** (pytorch-forecasting) GLOBAL cho 30 mã VN30 →
# dự đoán phiên T+1: **Tăng / Giảm / Đi ngang** (ngưỡng ±1%).
#
# ## Mục tiêu: accuracy OUT-OF-SAMPLE sát thực tế
# - Walk-forward theo THỜI GIAN tuyệt đối: train quá khứ → test tương lai. KHÔNG random split.
# - Split mốc chung mọi mã: train `< 2024`, val `2024`, test `>= 2025`.
# - So với baseline (majority-class, "luôn đi ngang") để biết model có thực sự tốt không.
#
# ## ⚠️ BASELINE THỰC TẾ ~53%, KHÔNG PHẢI 33%
# Vì lớp "đi ngang" chiếm đa số (ngưỡng ±1% gom nhiều phiên), "luôn đoán đi ngang" đạt ~53%
# accuracy trên test (đo thật trên VCB+FPT). → TFT phải vượt **~53%** mới THỰC SỰ có giá trị,
# không phải 33% (ngẫu nhiên đều). Đây mới là thước đo "sát thực tế". Đừng mừng nếu chỉ ~50%.
#
# ## ⚠️ CẢNH BÁO DỮ LIỆU (đọc kỹ, đừng ảo tưởng)
# - **sentiment_agg hiện TOÀN 0** (PhoBERT chưa train — đang stub) và chỉ phủ ~3 tháng/17 năm.
#   → 2 cột sentiment_agg/news_count gần như VÔ DỤNG lúc này. Giữ trong pipeline để khi thay
#   PhoBERT thật là có ngay; ĐỪNG kỳ vọng chúng tăng accuracy ở lần train này.
# - Model thật sự học từ feature GIÁ (ma7/ma20/rsi14/macd/macd_signal).
#
# ## ⚠️ CHẠY Ở ĐÂU
# - File này thiết kế cho **Kaggle GPU (T4/P100)**. Cell train được bọc `if RUN_TRAINING:`
#   (mặc định False) để chạy thử nhẹ ở local không nổ GPU. Trên Kaggle: đặt `RUN_TRAINING=True`.
# - Trên Kaggle, thay phần đọc DB bằng cách load CSV panel đã export (xem cell "Export panel"),
#   hoặc set biến môi trường `DATABASE_URL` của Supabase trong Kaggle Secrets.

# %%
from __future__ import annotations

import json

# %% [markdown]
# ## Cell 0 — Cài dep (Kaggle KHÔNG có sẵn pytorch-forecasting)
# Chạy cell này TRƯỚC khi train trên Kaggle. Local smoke-test (RUN_TRAINING=False) bỏ qua được.
# %%
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "pytorch-forecasting>=1.0", "lightning>=2.2", "nest-asyncio"],
    check=False,
)
print("✓ deps (torch/CUDA Kaggle có sẵn; chỉ cài pytorch-forecasting + lightning + nest-asyncio)")

import pandas as pd

# _REPO: local = từ __file__ (ml/notebooks/03..); Kaggle = không có __file__ → dùng cwd.
# Backend chỉ cần khi đọc DB (PANEL_CSV=None). Trên Kaggle dùng CSV → khối backend bỏ qua an toàn.
_REPO = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path.cwd()
_BACKEND = _REPO / "backend"
if _BACKEND.exists() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

MODEL_VERSION = "tft_v2"  # bump mỗi lần retrain (lưu accuracy theo version). v2 = chống collapse.
# Artifacts: local = ml/artifacts/tft_model; Kaggle = /kaggle/working (tải về sau khi chạy).
ARTIFACT_DIR = (
    _REPO / "ml" / "artifacts" / "tft_model"
    if (_REPO / "ml").exists()
    else Path("/kaggle/working/tft_model")
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# %%
try:
    import torch
    import torch.nn.functional as F
    from pytorch_forecasting.metrics import CrossEntropy

    class WeightedCrossEntropy(CrossEntropy):
        def __init__(self, weight: torch.Tensor | None = None, **kwargs):
            super().__init__(**kwargs)
            if weight is not None:
                self.register_buffer("weight", weight)
            else:
                self.weight = None

        def loss(self, y_pred, target):
            return F.cross_entropy(
                y_pred.view(-1, y_pred.size(-1)),
                target.view(-1),
                weight=self.weight,
            )
except ImportError:
    WeightedCrossEntropy = None

# --- TFT v2 chống collapse (xem grill 2026-06-10 + CONTEXT) -------------------------------------
# v1 COLLAPSE: đoán 100% di_ngang (CrossEntropy không trọng số + lớp di_ngang ~50%). Chẩn đoán
# local (LightGBM proxy) xác nhận: (a) thêm class weight → thoát collapse, đoán cả 3 lớp;
# (b) feature GIÁ THUẦN có TRẦN — không thể vừa acc>50% vừa macro-F1 cao (sentiment hiện =0).
#
# TIÊU CHÍ PASS v2 (KHÔNG phải acc>50% — bất khả thi với feature hiện tại):
#   macro-F1 >= 0.36 VÀ cả 3 lớp F1 > 0 (thoát collapse). Accuracy chỉ THAM CHIẾU (chấp nhận <50%).
PASS_MACRO_F1_MIN = 0.36

# Class weight = nghịch đảo tần suất, CHỈ tính trên TRAIN (chống leakage), chuẩn hoá mean=1.
# CLASS_WEIGHT_POWER: 1.0 = balanced đầy đủ (tối đa macro-F1, khớp tiêu chí); 0.5 = dịu hơn
# (đỡ tụt accuracy); 0.0 = không weight = collapse. Mặc định 1.0.
CLASS_WEIGHT_POWER = 1.0
# Tripwire: sau train, nếu 1 lớp chiếm > ngưỡng này trong dự đoán val → coi như VẪN collapse
# (weight bị version nuốt im lặng) → raise để chuyển Cách C (oversample). Tránh tốn trọn vòng GPU.
COLLAPSE_PRED_FRAC = 0.80

# Đường dẫn CSV panel (export từ local bằng backend/scripts/export_training_data.py).
# Khi chạy Kaggle: upload ml/data/training_panel.csv lên Kaggle Dataset (cell Cell 0 đã add).
# TỰ DÒ đường dẫn (đừng hardcode — Kaggle path tuỳ cách upload, hay sai). Local: trỏ file repo.
import glob as _glob

_found = _glob.glob("/kaggle/input/**/training_panel.csv", recursive=True)
_local = _REPO / "ml" / "data" / "training_panel.csv"  # _REPO đã tính an toàn ở trên
PANEL_CSV: Path | None = (
    Path(_found[0]) if _found else (_local if _local.exists() else None)
)  # None → đọc từ DB (cần DATABASE_URL); có CSV → dùng file
print(f"PANEL_CSV = {PANEL_CSV}")

# Mốc split theo THỜI GIAN (chung mọi mã) — walk-forward, không random.
TRAIN_END = "2024-01-01"  # train: date < TRAIN_END
VAL_END = "2025-01-01"  # val:   TRAIN_END <= date < VAL_END ; test: date >= VAL_END

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]  # fmt: skip

FEATURES = ["ma7", "ma20", "rsi14", "macd", "macd_signal", "sentiment_agg", "news_count"]
LABELS = ["tang", "giam", "di_ngang"]


def class_weights_from_train(
    train: pd.DataFrame, power: float = CLASS_WEIGHT_POWER
) -> dict[str, float]:
    """Trọng số lớp = (nghịch đảo tần suất)^power, CHỈ tính trên TRAIN, chuẩn hoá mean=1.

    Train-only là BẮT BUỘC: tính trên test/toàn panel = leak phân bố tương lai vào training.
    Trả dict {nhãn: weight}; caller map sang tensor theo ĐÚNG thứ tự target_normalizer.classes_
    (KHÔNG hardcode thứ tự LABELS — NaNLabelEncoder fit theo data).
    """
    import numpy as np

    cnt = train["label"].value_counts()
    classes = list(cnt.index)
    n, k = len(train), len(classes)
    raw = np.array([(n / (k * cnt[c])) ** power for c in classes])
    w = raw / raw.mean()  # chuẩn hoá mean=1
    assert abs(w.mean() - 1.0) < 1e-9 and (w > 0).all() and not np.isnan(w).any()
    return dict(zip(classes, (round(float(x), 4) for x in w), strict=True))


# %% [markdown]
# ## Build panel dataset (ghép 30 mã)
# Mỗi mã → `build_features` (đã chống leakage) → thêm `symbol`, `stock_id`, `time_idx`.
# `time_idx` = số phiên TĂNG DẦN per mã (yêu cầu của TimeSeriesDataSet).


# %%
async def _load_symbol_frame(symbol: str) -> pd.DataFrame:
    """Đọc 1 mã từ DB qua backend builder. Async vì builder dùng SQLAlchemy async."""
    from features.builder import load_training_frame  # import trễ (sau khi set sys.path)

    df = await load_training_frame(symbol)
    if df.empty:
        return df
    df = df.copy()
    df["symbol"] = symbol
    return df


async def build_panel(symbols: list[str]) -> pd.DataFrame:
    """Ghép nhiều mã → panel dataset cho global model. Dropna warm-up + loại label None."""
    frames = []
    for sym in symbols:
        f = await _load_symbol_frame(sym)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)

    # Loại hàng feature warm-up (ma20 cần 20 phiên) + hàng label None (hàng cuối mỗi mã).
    panel = panel.dropna(subset=FEATURES).copy()
    panel = panel[panel["label"].notna()].copy()

    # Sort TRƯỚC khi encode (cat.codes gán theo index hàng → phải sort xong mới gán, nếu không
    # stock_id lệch khỏi symbol). categories cố định theo danh sách mã → codes ổn định dù
    # thiếu mã (đồng bộ với CSV path trong main()).
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    codes = pd.Categorical(panel["symbol"], categories=sorted(symbols)).codes
    panel["stock_id"] = codes.astype(str)
    panel["time_idx"] = panel.groupby("symbol").cumcount()  # tăng dần per mã (đã sort theo date)
    return panel


def time_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split theo date (mốc chung mọi mã). Assert không chồng lấn thời gian."""
    d = pd.to_datetime(panel["date"])
    train = panel[d < TRAIN_END]
    val = panel[(d >= TRAIN_END) & (d < VAL_END)]
    test = panel[d >= VAL_END]
    # Walk-forward assert: train < val < test theo thời gian.
    if len(train) and len(val):
        assert pd.to_datetime(train["date"]).max() < pd.to_datetime(val["date"]).min()
    if len(val) and len(test):
        assert pd.to_datetime(val["date"]).max() < pd.to_datetime(test["date"]).min()
    return train, val, test


# %% [markdown]
# ## Cell PREFLIGHT (LOCAL — chỉ pandas/numpy, KHÔNG cần torch/GPU)
# Chạy local để tách RỦI RO LOGIC (split/weight/oversample của mình) khỏi RỦI RO API (version lib
# trên Kaggle). Cell này pass local → lên Kaggle lỗi gì thì chắc chắn do version, không phải code.
# Yêu cầu PANEL_CSV trỏ tới training_panel.csv (bỏ qua an toàn nếu chạy thẳng Kaggle với DB).


# %%
def preflight(panel: pd.DataFrame) -> None:
    """Kiểm split + công thức weight + logic oversample (Cách C) KHÔNG phá walk-forward."""
    train, val, test = time_split(panel)
    print(f"[preflight] split: train={len(train)} val={len(val)} test={len(test)}")

    w = class_weights_from_train(train)
    print(f"[preflight] class weight (train-only, power={CLASS_WEIGHT_POWER}, mean=1):", w)

    # Cách C: oversample lớp thiểu số trong TRAIN tới ~cân bằng — assert không lọt val/test.
    cnt = train["label"].value_counts()
    factor = {c: max(1, round(cnt.max() / cnt[c])) for c in cnt.index}
    over = pd.concat(
        [pd.concat([train[train.label == c]] * factor[c]) for c in cnt.index], ignore_index=True
    )
    assert over["date"].max() < pd.Timestamp(TRAIN_END), "LEAK: oversample lọt ngày >= TRAIN_END!"
    dist = over["label"].value_counts(normalize=True).round(3).to_dict()
    print(f"[preflight] oversample factor={factor} → phân bố {dist} | max(date)<TRAIN_END ✓")
    print("[preflight] ✓ logic OK — còn lại (CrossEntropy weight, predict fields) verify Kaggle.")


# Chạy preflight nếu có CSV local (an toàn bỏ qua khi chạy thẳng Kaggle/DB).
if PANEL_CSV and Path(PANEL_CSV).exists():
    _pf = pd.read_csv(PANEL_CSV, parse_dates=["date"])
    if "symbol" not in _pf.columns:
        _pf["symbol"] = _pf["stock_id"]
    _pf = _pf.dropna(subset=FEATURES + ["label"])
    preflight(_pf)


# %% [markdown]
# ## Baseline (cận dưới để so sánh)
# Nếu TFT không vượt được "luôn đoán đi ngang", model CHƯA học được gì.


# %%
def baseline_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Accuracy của majority-class (học từ train) + 'luôn di_ngang' trên TEST."""
    majority = train["label"].value_counts().idxmax()
    acc_majority = (test["label"] == majority).mean()
    acc_flat = (test["label"] == "di_ngang").mean()
    dist = test["label"].value_counts(normalize=True).round(4).to_dict()
    return {
        "majority_class": majority,
        "acc_majority_on_test": round(float(acc_majority), 4),
        "acc_always_flat_on_test": round(float(acc_flat), 4),
        "test_label_dist": dist,
    }


# %% [markdown]
# ## TFT (chạy trên Kaggle GPU)
#
# **⚠️ XÁC MINH API trên Kaggle:** pytorch-forecasting `TemporalFusionTransformer` vốn thiên về
# regression/quantile. Cho phân loại 3 lớp, cách chuẩn:
# - target là cột categorical (`label`), `TimeSeriesDataSet(..., target="label")`.
# - dùng loss phân loại: `from pytorch_forecasting.metrics import CrossEntropy` và
#   `TemporalFusionTransformer.from_dataset(..., loss=CrossEntropy(), output_size=3)`.
#
# CrossEntropy của pytorch-forecasting hỗ trợ classification, NHƯNG API (`output_size`, cách
# encode target categorical) thay đổi theo version. **PHẢI xác minh trên Kaggle với version
# thực tế.** Nếu version không hỗ trợ gọn → FALLBACK: train một GRU/LSTM classifier nhỏ trên cùng
# panel (cùng split thời gian) — kiến trúc đơn giản hơn nhưng đủ cho baseline có học.

# %%
RUN_TRAINING = True  # ĐẶT True trên Kaggle GPU. Local/smoke-test giữ False.


def build_tft_dataset(train: pd.DataFrame, max_encoder_length: int = 40):
    """Tạo TimeSeriesDataSet cho TFT. Import nặng nằm trong hàm để smoke-test không cần torch."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder

    return TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="label",
        group_ids=["stock_id"],
        static_categoricals=["stock_id"],
        time_varying_unknown_reals=FEATURES,
        max_encoder_length=max_encoder_length,
        max_prediction_length=1,  # dự đoán T+1
        target_normalizer=NaNLabelEncoder(),  # target phân loại
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
    )


def _weight_tensor(training, train: pd.DataFrame):
    """Tensor weight theo ĐÚNG thứ tự classes_ của target_normalizer (KHÔNG hardcode LABELS).

    NaNLabelEncoder fit theo data → thứ tự index có thể khác ["tang","giam",...]. Map dict
    class_weights_from_train (train-only) sang tensor đúng index model dùng.
    """
    import torch

    wmap = class_weights_from_train(train)  # {nhãn: weight} train-only, power, mean=1
    classes = [str(c) for c in training.target_normalizer.classes_]
    return torch.tensor([wmap[c] for c in classes], dtype=torch.float32), wmap, classes


def train_tft(train: pd.DataFrame, val: pd.DataFrame):
    """Train TFT phân loại 3 lớp + class weight (chống collapse) + EarlyStopping + tripwire.

    CÁCH A (chính): CrossEntropy(weight=tensor) — trọng số nghịch đảo tần suất train-only.
    TRIPWIRE: sau train, nếu 1 lớp chiếm > COLLAPSE_PRED_FRAC trong dự đoán val → weight bị
    version NUỐT IM LẶNG (vẫn collapse) → raise, chuyển CÁCH C (xem train_tft_oversample).
    CHỈ chạy khi RUN_TRAINING (Kaggle GPU).
    """
    import lightning.pytorch as pl
    import numpy as np
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    training = build_tft_dataset(train)
    validation = TimeSeriesDataSet.from_dataset(training, val, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=2)
    val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=2)

    weight, wmap, classes = _weight_tensor(training, train)
    print(f"class order (model học): {classes} | weight (train-only): {weight.tolist()}")

    # CÁCH A: CrossEntropy(weight=...) — ⚠️ sử dụng WeightedCrossEntropy tự định nghĩa.
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.2,
        loss=WeightedCrossEntropy(weight=weight),
        output_size=3,
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

    # TRIPWIRE chống nuốt-im-lặng: kiểm phân bố dự đoán trên VAL.
    vpred = np.asarray(
        tft.predict(val_loader, mode="prediction", return_x=True).output.cpu()
    ).ravel()
    frac = np.bincount(vpred.astype(int)).max() / len(vpred)
    print(
        f"[tripwire] val pred — lớp đông nhất {frac:.1%} (ngưỡng collapse {COLLAPSE_PRED_FRAC:.0%})"
    )
    if frac > COLLAPSE_PRED_FRAC:
        raise RuntimeError(
            f"VẪN COLLAPSE ({frac:.1%} 1 lớp) — weight có thể bị version nuốt im lặng. "
            "Chuyển CÁCH C: train_tft_oversample(train, val). Xem cell oversample bên dưới."
        )
    return tft, training, trainer


def train_tft_oversample(train: pd.DataFrame, val: pd.DataFrame):
    """CÁCH C (dự phòng khi A bị tripwire): oversample lớp thiểu số trong TRAIN tới ~cân bằng,
    rồi train CrossEntropy KHÔNG weight. Miễn nhiễm rủi ro version (không đụng API loss).

    An toàn split: chỉ nhân bản hàng TRAIN (date < TRAIN_END) → không thể lọt val/test. time_idx
    per mã giữ nguyên (nhân bản hàng không đổi time_idx của mã đó). Sau oversample phải sort lại.
    """
    cnt = train["label"].value_counts()
    factor = {c: max(1, round(cnt.max() / cnt[c])) for c in cnt.index}
    over = pd.concat(
        [pd.concat([train[train.label == c]] * factor[c]) for c in cnt.index], ignore_index=True
    )
    assert over["date"].max() < pd.Timestamp(TRAIN_END), "LEAK: oversample lọt ngày >= TRAIN_END!"
    over = over.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"[cách C] oversample factor={factor} → {len(train)} → {len(over)} hàng")

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import CrossEntropy

    training = build_tft_dataset(over)
    validation = TimeSeriesDataSet.from_dataset(training, val, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=2)
    val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=2)
    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=0.01, hidden_size=32, attention_head_size=2,
        dropout=0.2, loss=CrossEntropy(), output_size=3, log_interval=0,
    )
    trainer = pl.Trainer(
        max_epochs=30, accelerator="gpu", devices=1, gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return tft, training, trainer


def evaluate(tft, training, test: pd.DataFrame) -> dict:
    """Predict TEST out-of-sample → accuracy + confusion matrix + per-class F1.

    ALIGNMENT (quan trọng): dùng predict(return_x=True) để lấy y_true (decoder_target) ĐÚNG
    thứ tự với y_pred — KHÔNG slice test["label"][:len(y_pred)] (DataLoader có thể drop/reorder
    sample không đủ max_encoder_length → slice sai → accuracy vô nghĩa).

    LABEL ORDER: index lớp lấy từ training.target_normalizer.classes_ (thứ tự THẬT model học),
    KHÔNG hardcode LABELS — NaNLabelEncoder fit theo data, có thể khác thứ tự ["tang","giam",...].
    """
    import numpy as np
    from pytorch_forecasting import TimeSeriesDataSet
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    test_ds = TimeSeriesDataSet.from_dataset(training, test, stop_randomization=True)
    test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=2)

    # mode="prediction": với CrossEntropy, to_prediction() đã trả CLASS INDEX (không argmax kép).
    # ⚠️ TODO Kaggle: xác minh tên field của Prediction namedtuple (out.output / out.x) đúng version.
    out = tft.predict(test_loader, mode="prediction", return_x=True)
    y_pred = np.asarray(out.output.cpu()).ravel()
    y_true = np.asarray(out.x["decoder_target"].cpu()).ravel()

    # Bug-guard: nếu version trả [N,3] probabilities (ndim==2) thay vì class index → báo rõ.
    assert y_pred.ndim == 1, (
        f"predict() trả shape {out.output.shape} — version này KHÔNG argmax sẵn; "
        "thêm .argmax(-1) trước ravel(). Xác minh trên Kaggle."
    )

    classes = [str(c) for c in training.target_normalizer.classes_]  # thứ tự thật model học
    idx = list(range(len(classes)))
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=idx).tolist()
    report = classification_report(
        y_true, y_pred, labels=idx, target_names=classes, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "labels_order": classes,  # backend map index→nhãn theo đúng thứ tự này
        "confusion_matrix": cm,
        "per_class": report,
    }


# %% [markdown]
# ## Chạy chính


# %%
async def main(symbols: list[str] | None = None) -> dict:
    symbols = symbols or VN30

    if PANEL_CSV and Path(PANEL_CSV).exists():
        # Chạy Kaggle: đọc CSV đã export, không cần kết nối Supabase.
        print(f"Đọc panel từ CSV: {PANEL_CSV}")
        raw = pd.read_csv(PANEL_CSV)
        raw["date"] = pd.to_datetime(raw["date"])
        # Fallback CSV cũ (chỉ có stock_id = symbol-string, chưa có cột symbol).
        if "symbol" not in raw.columns:
            raw["symbol"] = raw["stock_id"]
        # Lọc trên symbol (KHÔNG trên stock_id — sau encode stock_id là số, isin(symbols) sẽ rỗng).
        raw = raw[raw["symbol"].isin(symbols)]
        raw = raw.dropna(subset=FEATURES).copy()
        raw = raw[raw["label"].notna()].copy()
        # Encode SAU sort + categories cố định — đồng bộ logic với build_panel().
        raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)
        codes = pd.Categorical(raw["symbol"], categories=sorted(symbols)).codes
        raw["stock_id"] = codes.astype(str)
        raw["time_idx"] = raw.groupby("symbol").cumcount()
        panel = raw
    else:
        panel = await build_panel(symbols)

    if panel.empty:
        raise RuntimeError("Panel rỗng — kiểm DB/.env backend hoặc đường dẫn PANEL_CSV.")

    train, val, test = time_split(panel)
    print(f"Panel: {len(panel)} hàng / {panel['symbol'].nunique()} mã")
    print(
        f"Split — train={len(train)} (<{TRAIN_END}), "
        f"val={len(val)} ({TRAIN_END}–{VAL_END}), test={len(test)} (>={VAL_END})"
    )

    base = baseline_report(train, test) if len(train) and len(test) else {}
    print("Baseline:", json.dumps(base, ensure_ascii=False))

    metrics = {
        "model_version": MODEL_VERSION,
        "baseline": base,
        # Tiêu chí v2 (xem grill 2026-06-10): accuracy KHÔNG phải mốc pass (trần feature giá).
        "pass_criteria": {
            "macro_f1_min": PASS_MACRO_F1_MIN,
            "all_classes_nonzero": True,
            "accuracy_is_reference_only": True,
            "note": "feature giá thuần có trần — chấp nhận acc < baseline; mục tiêu hết collapse.",
        },
        "changes_vs_v1": [
            f"CrossEntropy class weight nghịch đảo tần suất train-only, power={CLASS_WEIGHT_POWER}",
            "tripwire pred-dist > %.0f%% → fallback CÁCH C oversample" % (COLLAPSE_PRED_FRAC * 100),
        ],
    }

    if RUN_TRAINING:
        # CÁCH A; nếu tripwire bắt collapse (version nuốt weight) → CÁCH C oversample.
        try:
            tft, training, trainer = train_tft(train, val)
            metrics["train_method"] = "A:class_weight"
        except RuntimeError as exc:
            print(f"⚠️ {exc}\n→ Fallback CÁCH C (oversample).")
            tft, training, trainer = train_tft_oversample(train, val)
            metrics["train_method"] = "C:oversample"
        metrics["tft"] = evaluate(tft, training, test)
        # Đánh giá pass/fail theo tiêu chí v2 (macro-F1 + cả 3 lớp khác 0).
        per = metrics["tft"]["per_class"]
        cls = metrics["tft"]["labels_order"]
        macro_f1 = per["macro avg"]["f1-score"]
        all_nonzero = all(per[c]["f1-score"] > 0 for c in cls)
        metrics["tft"]["macro_f1"] = round(macro_f1, 4)
        metrics["passed"] = bool(macro_f1 >= PASS_MACRO_F1_MIN and all_nonzero)
        print(
            f"PASS={metrics['passed']} (macro-F1={macro_f1:.4f} >= {PASS_MACRO_F1_MIN}? "
            f"& cả 3 lớp>0? {all_nonzero}) | accuracy(tham chiếu)={metrics['tft']['accuracy']}"
        )
        # Checkpoint qua Lightning: lưu ĐẦY ĐỦ hparams + weights + dataset params trong 1 file.
        # Backend load lại bằng TemporalFusionTransformer.load_from_checkpoint(ckpt).
        # labels_order (thứ tự index→nhãn) nằm trong metrics_{version}.json để backend map đúng.
        ckpt = ARTIFACT_DIR / f"{MODEL_VERSION}.ckpt"
        trainer.save_checkpoint(ckpt)
        print(f"✓ checkpoint → {ckpt}")

    # Lưu metrics theo version (governance: accuracy theo version).
    (ARTIFACT_DIR / f"metrics_{MODEL_VERSION}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    return metrics


# %%
if __name__ == "__main__":
    import asyncio

    # Cho phép chạy asyncio trong môi trường Jupyter/Kaggle notebook đã có sẵn event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

    asyncio.run(main())
