# %% [markdown]
# # 04 — LightGBM global training (model thật thay TFT)
#
# Vì sao LightGBM (xem CONTEXT "Sự thật cần nhớ" + TICKLIST M1.1): TFT v1 VÀ v2 đều collapse
# (loss đứng im — optimization kẹt, không học được), trong khi LightGBM proxy lúc chẩn đoán
# với `class_weight=balanced` đoán cả 3 lớp, macro-F1 0.398 trên CÙNG feature + CÙNG split.
# → Model thật = LightGBM.
#
# Script tham số hoá theo MODEL_VERSION (đặt qua biến môi trường hoặc sửa default):
#   - lgbm_v3: feature giá cũ (ma7/ma20/rsi/macd + sentiment stub), split train<2024/val2024/test2025.
#   - lgbm_v4: feature volatility/momentum + sector (LGBM_V4_FEATURES từ backend). Split GIỐNG v3
#     (train<2024 / val 2024 / test 2025) để so GATE táo-táo với v3 — feature mới thắng rõ
#     (0.4234 > 0.4018 trên cùng test 2025). 2026 báo cáo riêng làm HOLDOUT vô nhiễm.
#     2008-2009 cắt (chế độ dị thường 4.4% data, train_start=2010). Xem notebook 05 + TICKLIST M2.
#     (Split dịch tiến test 2026 đã thử: cả v3+v4 ~0.40 vì 2026 mỏng+lệch chế độ — chôn cải thiện
#      thật của feature; quyết quay lại test 2025 cho công bằng, grill 2026-06-11.)
#
# Chạy LOCAL (vài giây, không cần Kaggle/GPU):
#   uv run --with lightgbm --with scikit-learn --with pandas --with scipy \
#       python ml/notebooks/04_lgbm_training.py
#   (macOS cần libomp:
#    export DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH")
#
# Bất biến:
# - Split THỜI GIAN tuyệt đối (walk-forward). KHÔNG random split.
# - Feature list = nguồn sự thật ở backend/features/builder.py (LGBM_V4_FEATURES) — KHÔNG tự chế
#   ở đây (chống train/serve skew). sector là categorical native của LightGBM.
# - Tiêu chí PASS: macro-F1 >= 0.36 VÀ cả 3 lớp F1 > 0. Accuracy chỉ THAM CHIẾU (trần feature giá).
#   v4 thêm GATE so v3: macro-F1 phải > 0.4018 (mốc v3) để đáng wire frontend (M2 DoD).
#
# Calibration (quyết định grill 2026-06-10, chạy ngay trong script):
# - Temperature scaling fit trên VAL (1 tham số, không đổi argmax).
# - Frontier coverage–precision threshold 0.40→0.80 bước 0.01 trên VAL → chọn threshold nhỏ nhất
#   đạt precision >= 0.55 & coverage >= 0.20; fallback precision >= 0.50; không có → null.
#   ⚠️ Bài học regime-shift v3 (val 2024 → test 2025 không transfer). v4: val 2025 vẫn KHÁC chế độ
#   test 2026 (di_ngang 53.7% vs 42.5%) → threshold val có thể không transfer; báo cáo frontier test
#   đầy đủ để thấy trung thực.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
# Feature list = nguồn sự thật ở backend (chống train/serve skew).
sys.path.insert(0, str(_REPO / "backend"))
from features.builder import LGBM_V4_FEATURES  # noqa: E402

MODEL_VERSION = os.environ.get("MODEL_VERSION", "lgbm_v4")
PANEL_CSV = _REPO / "ml" / "data" / "training_panel.csv"
ARTIFACT_DIR = _REPO / "ml" / "artifacts" / "lgbm_model"

# --- Cấu hình theo version (split + feature + categorical khác nhau) ---
# v3: tái lập lịch sử (so sánh táo-táo với tft_v2 + proxy). v4: split dịch tiến + feature mới.
_V3_FEATURES = [
    "ma7",
    "ma20",
    "rsi14",
    "macd",
    "macd_signal",
    "sentiment_agg",
    "news_count",
]
CONFIGS = {
    "lgbm_v3": {
        "features": _V3_FEATURES,
        "categorical": [],
        "train_start": None,
        "train_end": "2024-01-01",
        "val_end": "2025-01-01",
        "test_end": None,  # test = phần còn lại >= val_end
    },
    "lgbm_v4": {
        "features": list(LGBM_V4_FEATURES),
        "categorical": ["sector"],
        "train_start": "2010-01-01",  # cắt 2008-2009 (chế độ dị thường)
        "train_end": "2024-01-01",
        "val_end": "2025-01-01",
        "test_end": "2026-01-01",  # test = 2025 (so táo-táo v3); 2026 báo cáo riêng làm holdout
        "holdout_start": "2026-01-01",  # holdout vô nhiễm — báo cáo, KHÔNG chọn threshold
    },
    # v5 = v4 + sentiment THẬT (PhoBERT v3, daily_sentiment re-score 3960 tin). CÙNG split v4 để so
    # táo-táo: chênh lệch macro-F1 = đóng góp ròng của sentiment. Sentiment thưa (đa số ngày 0 tin)
    # → kỳ vọng cải thiện nhỏ; frontier v5 vs v4 quyết định có wire production hay không (bước E).
    "lgbm_v5": {
        "features": [f for f in LGBM_V4_FEATURES if f != "sector"]
        + ["sentiment_agg", "news_count", "sector"],
        "categorical": ["sector"],
        "train_start": "2010-01-01",
        "train_end": "2024-01-01",
        "val_end": "2025-01-01",
        "test_end": "2026-01-01",
        "holdout_start": "2026-01-01",
    },
}
CFG = CONFIGS[MODEL_VERSION]
FEATURES = CFG["features"]
CATEGORICAL = CFG["categorical"]

PASS_MACRO_F1_MIN = 0.36
V3_MACRO_F1 = 0.4018  # mốc v3 — v4 phải vượt để đáng wire (M2 gate)

# Frontier confidence-gating (quyết định đã chốt trong PLAN).
THRESHOLD_GRID = np.round(np.arange(0.40, 0.801, 0.01), 2)
PRECISION_TARGET = 0.55
PRECISION_FALLBACK = 0.50
COVERAGE_MIN = 0.20

SEED = 42


# %%
def load_panel(csv_path: Path) -> pd.DataFrame:
    """Đọc panel CSV (export từ backend/scripts/export_training_data.py).

    Bỏ hàng warm-up (NaN feature SỐ) + hàng label None (hàng cuối mỗi mã — builder cố ý
    để None chống leakage). sector là categorical (không dropna theo nó — None hợp lệ là 1 mức).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} chưa có — chạy: "
            "cd backend && uv run python -m scripts.export_training_data"
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    numeric_feats = [f for f in FEATURES if f not in CATEGORICAL]
    df = df.dropna(subset=numeric_feats).copy()
    df = df[df["label"].notna()].copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")  # LightGBM categorical native
    sort_key = "symbol" if "symbol" in df.columns else "stock_id"
    return df.sort_values([sort_key, "date"]).reset_index(drop=True)


def time_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split theo date (mốc chung mọi mã) — walk-forward, assert không chồng lấn thời gian."""
    d = panel["date"]
    train_mask = d < CFG["train_end"]
    if CFG["train_start"] is not None:
        train_mask &= d >= CFG["train_start"]
    train = panel[train_mask]
    val = panel[(d >= CFG["train_end"]) & (d < CFG["val_end"])]
    test_mask = d >= CFG["val_end"]
    if CFG["test_end"] is not None:
        test_mask &= d < CFG["test_end"]
    test = panel[test_mask]
    assert train["date"].max() < val["date"].min(), "LEAK: train chồng lấn val!"
    assert val["date"].max() < test["date"].min(), "LEAK: val chồng lấn test!"
    return train, val, test


def baseline_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Majority-class (học từ train) + 'luôn di_ngang' trên TEST — cận dưới để so sánh."""
    majority = train["label"].value_counts().idxmax()
    return {
        "majority_class": majority,
        "acc_majority_on_test": round(float((test["label"] == majority).mean()), 4),
        "acc_always_flat_on_test": round(
            float((test["label"] == "di_ngang").mean()), 4
        ),
        "test_label_dist": test["label"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
    }


# %% [markdown]
# ## Calibration: temperature scaling + frontier coverage–precision
# Temperature scaling trên log-prob: softmax(log(p)/T). T fit bằng minimize NLL trên VAL; T không
# đổi argmax nên macro-F1/accuracy giữ nguyên — chỉ "độ tự tin" được nắn lại cho threshold có nghĩa.


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
# Hyperparams để DEFAULT (không tune) — mục tiêu là tín hiệu feature mới, không vắt số.


# %%
def main() -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    panel = load_panel(PANEL_CSV)
    train, val, test = time_split(panel)
    n_syms = (
        panel["symbol"].nunique()
        if "symbol" in panel.columns
        else panel["stock_id"].nunique()
    )
    print(
        f"=== {MODEL_VERSION} === Panel: {len(panel)} hàng / {n_syms} mã | features={FEATURES}"
    )
    print(
        f"Split — train={len(train)} "
        f"([{CFG['train_start'] or 'đầu'}, {CFG['train_end']})), "
        f"val={len(val)} (<{CFG['val_end']}), test={len(test)} (>={CFG['val_end']})"
    )

    base = baseline_report(train, test)
    print("Baseline:", json.dumps(base, ensure_ascii=False))

    fit_kwargs = {}
    if CATEGORICAL:
        fit_kwargs["categorical_feature"] = CATEGORICAL
    model = LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
    model.fit(train[FEATURES], train["label"], **fit_kwargs)
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
    pred_dist = {
        c: round(float((y_pred == i).mean()), 4) for i, c in enumerate(classes)
    }
    passed = bool(macro_f1 >= PASS_MACRO_F1_MIN and all_nonzero)
    beats_v3 = bool(macro_f1 > V3_MACRO_F1)
    print(
        f"PASS={passed} (macro-F1={macro_f1:.4f} >= {PASS_MACRO_F1_MIN}? "
        f"& cả 3 lớp>0? {all_nonzero})"
        f" | acc(tham chiếu)={report['accuracy']:.4f} | pred-dist={pred_dist}"
    )
    print(f"GATE so v3: macro-F1 {macro_f1:.4f} > {V3_MACRO_F1} (v3)? → {beats_v3}")

    # Calibration trên VAL → áp threshold đã chọn lên TEST để báo cáo trung thực out-of-sample.
    probs_val, y_val = eval_split(val)
    temperature = fit_temperature(probs_val, y_val)
    frontier_val = frontier(
        apply_temperature(probs_val, temperature), y_val, THRESHOLD_GRID
    )
    threshold, threshold_rule = choose_threshold(frontier_val)
    frontier_test = frontier(
        apply_temperature(probs_test, temperature), y_test, THRESHOLD_GRID
    )
    test_at_thr = (
        next((r for r in frontier_test if r["threshold"] == threshold), None)
        if threshold is not None
        else None
    )
    print(f"Calibration: T={temperature} | threshold={threshold} ({threshold_rule})")
    print(f"Test tại threshold: {test_at_thr}")

    # Holdout vô nhiễm (v4: 2026 H1) — KHÔNG fit/chọn gì, chỉ áp model+T+threshold đã chốt.
    # Đo độ bền out-of-sample thật khi gặp chế độ mới (di_ngang ~42% vs train ~53%).
    holdout_report = None
    holdout_start = CFG.get("holdout_start")
    if holdout_start is not None:
        hold = panel[panel["date"] >= holdout_start]
        if not hold.empty:
            from sklearn.metrics import accuracy_score as _acc

            probs_hold, y_hold = eval_split(hold)
            pred_hold = probs_hold.argmax(axis=1)
            rep_h = classification_report(
                y_hold,
                pred_hold,
                labels=range(len(classes)),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            )
            front_h = frontier(
                apply_temperature(probs_hold, temperature), y_hold, THRESHOLD_GRID
            )
            hold_at_thr = (
                next((r for r in front_h if r["threshold"] == threshold), None)
                if threshold is not None
                else None
            )
            holdout_report = {
                "period": f">={holdout_start}",
                "n": int(len(hold)),
                "accuracy": round(float(_acc(y_hold, pred_hold)), 4),
                "macro_f1": round(float(rep_h["macro avg"]["f1-score"]), 4),
                "pred_dist": {
                    c: round(float((pred_hold == i).mean()), 4)
                    for i, c in enumerate(classes)
                },
                "at_threshold": hold_at_thr,
                "note": "vô nhiễm — không tune; chế độ 2026 khác train (regime shift đã biết).",
            }
            print(
                f"Holdout 2026 (vô nhiễm): {json.dumps(holdout_report, ensure_ascii=False)}"
            )

    metrics = {
        "model_version": MODEL_VERSION,
        "model_family": "lightgbm",
        "why_not_tft": (
            "tft_v1 + tft_v2 đều collapse (loss đứng im); LightGBM proxy 0.398 → model thật"
        ),
        "split": {
            "train_start": CFG["train_start"],
            "train_end": CFG["train_end"],
            "val_end": CFG["val_end"],
            "test_end": CFG["test_end"],
            "note": (
                "v4: split GIỐNG v3 (train<2024 [từ 2010] / val 2024 / test 2025) để so gate "
                "táo-táo. 2026 báo cáo riêng (holdout) vì mỏng+lệch chế độ làm test chính bị nhiễu."
            ),
        },
        "baseline": base,
        "pass_criteria": {
            "macro_f1_min": PASS_MACRO_F1_MIN,
            "all_classes_nonzero": True,
            "accuracy_is_reference_only": True,
            "v4_gate_beats_v3_macro_f1": V3_MACRO_F1,
            "beats_v3": beats_v3,
            "note": "feature giá thuần có trần — chấp nhận acc < baseline; mục tiêu hết collapse.",
        },
        "train_params": {
            "class_weight": "balanced",
            "random_state": SEED,
            "defaults": "lightgbm 4.x",
            "categorical_feature": CATEGORICAL,
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
        "holdout": holdout_report,
        "passed": passed,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / f"{MODEL_VERSION}.txt"
    model.booster_.save_model(
        model_path
    )  # native format: load không cần sklearn, không pickle
    (ARTIFACT_DIR / f"metrics_{MODEL_VERSION}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    print(f"✓ model → {model_path}\n✓ metrics → metrics_{MODEL_VERSION}.json")
    return metrics


# %%
if __name__ == "__main__":
    main()
