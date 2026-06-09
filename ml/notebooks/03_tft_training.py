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
import sys
from pathlib import Path

import pandas as pd

# Cho phép import feature builder của backend (backend = nguồn sự thật cho feature).
_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

MODEL_VERSION = "tft_v1"  # bump mỗi lần retrain (lưu accuracy theo version)
ARTIFACT_DIR = _REPO / "ml" / "artifacts" / "tft_model"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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

    # stock_id (categorical) + time_idx tăng dần per mã (sau khi đã sort theo date trong builder).
    panel["stock_id"] = panel["symbol"].astype("category").cat.codes
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["time_idx"] = panel.groupby("symbol").cumcount()
    panel["stock_id"] = panel["stock_id"].astype(str)  # TimeSeriesDataSet cần group_id dạng str/cat
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
RUN_TRAINING = False  # ĐẶT True trên Kaggle GPU. Local/smoke-test giữ False.


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


def train_tft(train: pd.DataFrame, val: pd.DataFrame):
    """Train TFT phân loại 3 lớp + EarlyStopping. CHỈ chạy khi RUN_TRAINING (Kaggle GPU)."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import CrossEntropy

    training = build_tft_dataset(train)
    validation = TimeSeriesDataSet.from_dataset(training, val, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=2)
    val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=2)

    # ⚠️ output_size=3 (3 lớp) + CrossEntropy — xác minh đúng version trên Kaggle.
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.2,
        loss=CrossEntropy(),
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
    return tft, training


def evaluate(tft, training, test: pd.DataFrame) -> dict:
    """Predict TEST out-of-sample → accuracy + confusion matrix + per-class F1."""
    from pytorch_forecasting import TimeSeriesDataSet
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    test_ds = TimeSeriesDataSet.from_dataset(training, test, stop_randomization=True)
    test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=2)
    preds = tft.predict(test_loader)  # nhãn dự đoán (index lớp)
    y_pred = preds.cpu().numpy().ravel()

    # y_true căn theo decoded index của test_ds (xác minh thứ tự trên Kaggle).
    y_true = test_ds.x_to_index(test_loader)  # placeholder — xem TODO bên dưới
    # ⚠️ TODO Kaggle: lấy y_true đúng thứ tự từ test_ds. Tạm dùng cột label nếu thứ tự khớp.
    y_true = test["label"].map({lbl: i for i, lbl in enumerate(LABELS)}).to_numpy()[: len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    return {"accuracy": round(float(acc), 4), "confusion_matrix": cm, "per_class": report}


# %% [markdown]
# ## Chạy chính


# %%
async def main(symbols: list[str] | None = None) -> dict:
    symbols = symbols or VN30
    panel = await build_panel(symbols)
    if panel.empty:
        raise RuntimeError("Panel rỗng — kiểm DB/.env backend.")

    train, val, test = time_split(panel)
    print(f"Panel: {len(panel)} hàng / {panel['symbol'].nunique()} mã")
    print(
        f"Split — train={len(train)} (<{TRAIN_END}), "
        f"val={len(val)} ({TRAIN_END}–{VAL_END}), test={len(test)} (>={VAL_END})"
    )

    base = baseline_report(train, test) if len(train) and len(test) else {}
    print("Baseline:", json.dumps(base, ensure_ascii=False))

    metrics = {"model_version": MODEL_VERSION, "baseline": base}

    if RUN_TRAINING:
        tft, training = train_tft(train, val)
        metrics["tft"] = evaluate(tft, training, test)
        print("TFT out-of-sample:", json.dumps(metrics["tft"]["accuracy"], ensure_ascii=False))
        # Export checkpoint + model_version.
        ckpt = ARTIFACT_DIR / f"{MODEL_VERSION}.ckpt"
        import torch

        torch.save(tft.state_dict(), ckpt)
        print(f"✓ checkpoint → {ckpt}")

    # Lưu metrics theo version (governance: accuracy theo version).
    (ARTIFACT_DIR / f"metrics_{MODEL_VERSION}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    return metrics


# %%
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
