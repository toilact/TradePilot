# %% [markdown]
# # 04 — LightGBM global training (lgbm_v3, model thật thay TFT)
#
# Vì sao LightGBM (xem CONTEXT "Sự thật cần nhớ" + TICKLIST M1.1): TFT v1 VÀ v2 đều collapse
# (loss đứng im — optimization kẹt, không học được), trong khi LightGBM proxy lúc chẩn đoán
# với `class_weight=balanced` đoán cả 3 lớp, macro-F1 0.398 trên CÙNG feature + CÙNG split.
# → Model thật = LightGBM. Script proxy gốc chưa từng commit — file này là bản chính thức.
#
# Chạy LOCAL (vài giây, không cần Kaggle/GPU):
#   uv run --with lightgbm --with scikit-learn --with pandas \
#       python ml/notebooks/04_lgbm_training.py
#   (macOS cần libomp:
#    export DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH")
#
# Bất biến giữ nguyên từ notebook 03 (so sánh táo-với-táo với tft_v2 và proxy 0.398):
# - Split THỜI GIAN tuyệt đối: train < 2024, val 2024, test >= 2025. KHÔNG random split.
# - FEATURES y hệt (KHÔNG thêm feature mới ở đây — feature mới phải vào backend/features/builder.py
#   kèm test no-leakage, đó là việc của M2; KHÔNG dùng symbol làm categorical — proxy không dùng).
# - Tiêu chí PASS: macro-F1 >= 0.36 VÀ cả 3 lớp F1 > 0. Accuracy chỉ THAM CHIẾU (trần feature giá).
#
# Calibration (quyết định grill 2026-06-10, chạy ngay trong script):
# - Temperature scaling fit trên VAL (1 tham số, không đổi argmax).
# - Frontier coverage–precision threshold 0.40→0.80 bước 0.01 trên VAL → chọn threshold nhỏ nhất
#   đạt precision >= 0.55 & coverage >= 0.20; fallback precision >= 0.50; không có → null
#   (báo cáo frontier đầy đủ — kết quả âm tính cũng là dữ kiện cho M2).

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

MODEL_VERSION = "lgbm_v3"  # đếm GLOBAL: v0 stub → v1, v2 TFT (collapse) → v3 LightGBM
PANEL_CSV = _REPO / "ml" / "data" / "training_panel.csv"
ARTIFACT_DIR = _REPO / "ml" / "artifacts" / "lgbm_model"

# Mốc split + features + tiêu chí: GIỮ Y HỆT notebook 03 (so sánh được giữa các version).
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
FEATURES = ["ma7", "ma20", "rsi14", "macd", "macd_signal", "sentiment_agg", "news_count"]
PASS_MACRO_F1_MIN = 0.36

# Frontier confidence-gating (quyết định đã chốt trong PLAN).
THRESHOLD_GRID = np.round(np.arange(0.40, 0.801, 0.01), 2)
PRECISION_TARGET = 0.55
PRECISION_FALLBACK = 0.50
COVERAGE_MIN = 0.20

SEED = 42


# %%
def load_panel(csv_path: Path) -> pd.DataFrame:
    """Đọc panel CSV (export từ backend/scripts/export_training_data.py).

    Cùng quy tắc lọc với notebook 03: bỏ hàng warm-up (NaN feature) + hàng label None
    (hàng cuối mỗi mã — builder cố ý để None chống leakage).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} chưa có — chạy: "
            "cd backend && uv run python -m scripts.export_training_data"
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=FEATURES).copy()
    df = df[df["label"].notna()].copy()
    return df.sort_values(["symbol" if "symbol" in df.columns else "stock_id", "date"]).reset_index(
        drop=True
    )


def time_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split theo date (mốc chung mọi mã) — walk-forward, assert không chồng lấn thời gian."""
    d = panel["date"]
    train = panel[d < TRAIN_END]
    val = panel[(d >= TRAIN_END) & (d < VAL_END)]
    test = panel[d >= VAL_END]
    assert train["date"].max() < val["date"].min(), "LEAK: train chồng lấn val!"
    assert val["date"].max() < test["date"].min(), "LEAK: val chồng lấn test!"
    return train, val, test


def baseline_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Majority-class (học từ train) + 'luôn di_ngang' trên TEST — cận dưới để so sánh."""
    majority = train["label"].value_counts().idxmax()
    return {
        "majority_class": majority,
        "acc_majority_on_test": round(float((test["label"] == majority).mean()), 4),
        "acc_always_flat_on_test": round(float((test["label"] == "di_ngang").mean()), 4),
        "test_label_dist": test["label"].value_counts(normalize=True).round(4).to_dict(),
    }


# %% [markdown]
# ## Calibration: temperature scaling + frontier coverage–precision
# Temperature scaling trên log-prob: softmax(log(p)/T). Shift-invariance của softmax →
# tương đương chuẩn trên logits. T fit bằng minimize NLL trên VAL; T không đổi argmax
# nên macro-F1/accuracy giữ nguyên — chỉ "độ tự tin" được nắn lại cho threshold có nghĩa.


# %%
def fit_temperature(probs_val: np.ndarray, y_val_idx: np.ndarray) -> float:
    """Fit T > 0 minimize NLL trên val. probs_val: [N,3] từ predict_proba."""
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(probs_val, 1e-12, None))

    def nll(t: float) -> float:
        scaled = logp / t
        scaled -= scaled.max(axis=1, keepdims=True)  # ổn định số học
        log_softmax = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_softmax[np.arange(len(y_val_idx)), y_val_idx].mean()

    res = minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded")
    return round(float(res.x), 4)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-12, None)) / t
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def frontier(probs: np.ndarray, y_idx: np.ndarray, grid: np.ndarray) -> list[dict]:
    """Bảng coverage–precision theo threshold. precision = acc trên tập dám đoán (actionable)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    rows = []
    for thr in grid:
        mask = conf >= thr
        cov = float(mask.mean())
        prec = float((pred[mask] == y_idx[mask]).mean()) if mask.any() else None
        rows.append(
            {
                "threshold": float(thr),
                "coverage": round(cov, 4),
                "precision": round(prec, 4) if prec is not None else None,
            }
        )
    return rows


def choose_threshold(rows: list[dict]) -> tuple[float | None, str]:
    """Threshold NHỎ NHẤT đạt precision >= target & coverage >= min; fallback >= 0.50; else None."""
    for target, rule in (
        (PRECISION_TARGET, f"primary>={PRECISION_TARGET}"),
        (PRECISION_FALLBACK, f"fallback>={PRECISION_FALLBACK}"),
    ):
        for r in rows:  # grid tăng dần → threshold nhỏ nhất trước
            if (
                r["precision"] is not None
                and r["precision"] >= target
                and r["coverage"] >= COVERAGE_MIN
            ):
                return r["threshold"], rule
    return (
        None,
        "none: không threshold nào đạt (precision target + coverage >= 20%) — trần feature giá",
    )


# %% [markdown]
# ## Train + evaluate
# `class_weight="balanced"` là yếu tố chống collapse đã chứng minh bằng proxy (macro-F1 0.398).
# Hyperparams để DEFAULT (không tune) — mục tiêu phiên này là TÁI LẬP proxy, không phải vắt số.


# %%
def main() -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    panel = load_panel(PANEL_CSV)
    train, val, test = time_split(panel)
    n_syms = panel["symbol"].nunique() if "symbol" in panel.columns else panel["stock_id"].nunique()
    print(f"Panel: {len(panel)} hàng / {n_syms} mã")
    print(
        f"Split — train={len(train)} (<{TRAIN_END}), val={len(val)}, test={len(test)} (>={VAL_END})"
    )

    base = baseline_report(train, test)
    print("Baseline:", json.dumps(base, ensure_ascii=False))

    model = LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
    model.fit(train[FEATURES], train["label"])
    classes = [str(c) for c in model.classes_]
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    def eval_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        probs = model.predict_proba(df[FEATURES])
        y_idx = df["label"].map(cls_to_idx)
        assert not y_idx.isna().any(), (
            f"Label lạ ngoài {classes} trong CSV — kiểm tra panel/export."
        )
        return probs, y_idx.to_numpy()

    probs_test, y_test = eval_split(test)
    y_pred = probs_test.argmax(axis=1)
    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    all_nonzero = all(report[c]["f1-score"] > 0 for c in classes)
    pred_dist = {c: round(float((y_pred == i).mean()), 4) for i, c in enumerate(classes)}
    passed = bool(macro_f1 >= PASS_MACRO_F1_MIN and all_nonzero)
    print(
        f"PASS={passed} (macro-F1={macro_f1:.4f} >= {PASS_MACRO_F1_MIN}? "
        f"& cả 3 lớp>0? {all_nonzero})"
        f" | acc(tham chiếu)={report['accuracy']:.4f} | pred-dist={pred_dist}"
    )

    # Calibration trên VAL → áp threshold đã chọn lên TEST để báo cáo trung thực out-of-sample.
    probs_val, y_val = eval_split(val)
    temperature = fit_temperature(probs_val, y_val)
    frontier_val = frontier(apply_temperature(probs_val, temperature), y_val, THRESHOLD_GRID)
    threshold, threshold_rule = choose_threshold(frontier_val)
    frontier_test = frontier(apply_temperature(probs_test, temperature), y_test, THRESHOLD_GRID)
    test_at_thr = (
        next((r for r in frontier_test if r["threshold"] == threshold), None)
        if threshold is not None
        else None
    )
    print(f"Calibration: T={temperature} | threshold={threshold} ({threshold_rule})")
    print(f"Test tại threshold: {test_at_thr}")

    metrics = {
        "model_version": MODEL_VERSION,
        "model_family": "lightgbm",
        "why_not_tft": (
            "tft_v1 + tft_v2 đều collapse (loss đứng im); LightGBM proxy 0.398 → model thật"
        ),
        "baseline": base,
        "pass_criteria": {
            "macro_f1_min": PASS_MACRO_F1_MIN,
            "all_classes_nonzero": True,
            "accuracy_is_reference_only": True,
            "note": "feature giá thuần có trần — chấp nhận acc < baseline; mục tiêu hết collapse.",
        },
        "train_params": {
            "class_weight": "balanced",
            "random_state": SEED,
            "defaults": "lightgbm 4.x",
        },
        "feature_names": FEATURES,
        "classes_order": classes,  # backend map index→nhãn theo đúng thứ tự này (model.classes_)
        "feature_importance_gain": dict(
            zip(
                FEATURES,
                (round(float(x), 2) for x in model.booster_.feature_importance("gain")),
                strict=True,
            )
        ),
        "test": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(macro_f1), 4),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=range(len(classes))
            ).tolist(),
            "per_class": report,
            "pred_dist": pred_dist,
        },
        "calibration": {
            "temperature": temperature,
            "threshold": threshold,
            "threshold_rule": threshold_rule,
            "coverage_min": COVERAGE_MIN,
            "test_at_threshold": test_at_thr,
            "frontier_val": frontier_val,
            "frontier_test": frontier_test,
        },
        "passed": passed,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / f"{MODEL_VERSION}.txt"
    model.booster_.save_model(model_path)  # native format: load không cần sklearn, không pickle
    (ARTIFACT_DIR / f"metrics_{MODEL_VERSION}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    print(f"✓ model → {model_path}\n✓ metrics → metrics_{MODEL_VERSION}.json")
    return metrics


# %%
if __name__ == "__main__":
    main()
