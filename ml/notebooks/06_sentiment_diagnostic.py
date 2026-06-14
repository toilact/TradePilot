# %% [markdown]
# # 06 — Sentiment diagnostic (M8 Pha 1): sentiment có ích cho dự đoán giá KHÔNG?
#
# **Diagnostic-first** (xem plan M8): lgbm_v5 (sentiment làm feature) đã FAIL trước đây (noise)
# vì coverage ~2-3%. Sau Pha 1a (sector-mapping) coverage vùng gần đây lên 20-47%. TRƯỚC khi đốt
# công cào (Pha 2), chứng minh trần tín hiệu trên data HIỆN CÓ bằng 3 tầng rẻ→đắt. Mục tiêu cuối
# = (B) độ chính xác LightGBM (precision-trên-actionable), KHÔNG phải macro-F1 PhoBERT.
#
# - **Tầng 0** (không train): trên news-rows, sentiment_agg có liên hệ với nhãn T+1 không?
#   boxplot/MI + **null-test shuffle trong ngày** + **kiểm chứng 1-mã** (bắt artifact "nhớ ngày"
#   do sector-mapping tạo 1 tin→N mã cùng điểm). Fail → DỪNG, không cào.
# - **Tầng 1a** (sanity): LightGBM news-rows CÓ vs KHÔNG sentiment-feature; delta macro-F1 ≥ +0.01
#   out-of-time = tín hiệu khai thác được (cơ chế Option B — chỉ tham khảo).
# - **Tầng 1b** (GATE quyết định): mô phỏng đúng **Option C** — lgbm_v4 THẬT đông cứng + override
#   is_actionable theo sentiment_extreme cực đoan; đo precision-trên-actionable vs coverage theo
#   ngưỡng τ × 2 chế độ. Pass → mở Pha 2/3; fail → giữ v4 thuần, ghi âm tính.
#
# Chạy LOCAL:
#   export DYLD_LIBRARY_PATH="$HOME/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
#   cd backend && uv run python -m scripts.export_diagnostic_panel   # tạo diagnostic_panel.csv
#   uv run --with lightgbm --with scikit-learn --with pandas --with scipy \
#       python ml/notebooks/06_sentiment_diagnostic.py

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "backend"))
from features.builder import LGBM_V4_FEATURES  # noqa: E402

PANEL_CSV = _REPO / "ml" / "data" / "diagnostic_panel.csv"
LGBM_ARTIFACT = _REPO / "ml" / "artifacts" / "lgbm_model"

# Vùng "recent-dense" cho diagnostic (coverage 2022+ đủ dày sau sector-mapping).
RECENT_START = "2022-01-01"
# Split walk-forward cho Tầng 1a (news-rows): train < 2025 / val 2025 / test 2026.
T1A_TRAIN_END = "2025-01-01"
T1A_VAL_END = "2026-01-01"
# Tầng 1b gating-sim: đánh giá trên giai đoạn lgbm_v4 CHƯA thấy (test 2025 + holdout 2026).
GATING_EVAL_START = "2025-01-01"

TAU_GRID = [0.5, 0.6, 0.7, 0.8]
PRODUCTION_THRESHOLD = 0.60  # ngưỡng confidence gating hiện hành (ADR 0002)
T1A_DELTA_GATE = 0.01  # delta macro-F1 tối thiểu coi là "tín hiệu khai thác được"
SEED = 42


# %%
def load_panel() -> pd.DataFrame:
    if not PANEL_CSV.exists():
        raise FileNotFoundError(
            f"{PANEL_CSV} chưa có — chạy: "
            "cd backend && uv run python -m scripts.export_diagnostic_panel"
        )
    df = pd.read_csv(PANEL_CSV, parse_dates=["date"])
    df = df[df["label"].notna()].copy()
    if "sentiment_extreme" not in df.columns:
        df["sentiment_extreme"] = 0.0
    df["sentiment_extreme"] = df["sentiment_extreme"].fillna(0.0)
    df["sentiment_agg"] = df["sentiment_agg"].fillna(0.0)
    df["news_count"] = df["news_count"].fillna(0).astype(int)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# %% [markdown]
# ## Tầng 0 — Conditional signal test (không train)
# Chỉ news-rows (news_count>0) vùng recent-dense. sentiment_agg có dự báo nhãn T+1 không?
# Phải sống qua: boxplot tách lớp / MI > null + kiểm chứng 1-mã (loại artifact nhớ-ngày).


# %%
def _mi(x: np.ndarray, y: np.ndarray) -> float:
    """MI(sentiment liên tục → nhãn rời rạc). 1 con số gọn để so với null-shuffle."""
    from sklearn.feature_selection import mutual_info_classif

    return float(
        mutual_info_classif(
            x.reshape(-1, 1), y, discrete_features=False, random_state=SEED
        )[0]
    )


def _quintile_monotonic(df: pd.DataFrame) -> dict:
    """Chia sentiment_agg thành 5 nhóm → tỉ lệ 'tang' tăng dần? (đơn điệu = tín hiệu hướng)."""
    try:
        q = pd.qcut(df["sentiment_agg"], 5, duplicates="drop")
    except ValueError:
        return {"note": "không đủ phân vị (sentiment_agg ít giá trị)"}
    rate = df.assign(q=q).groupby("q", observed=True).apply(
        lambda g: float((g["label"] == "tang").mean()), include_groups=False
    )
    vals = [round(v, 4) for v in rate.tolist()]
    return {"p_tang_by_quintile": vals, "monotonic_up": vals == sorted(vals)}


def tier0(panel: pd.DataFrame) -> dict:
    news = panel[(panel["news_count"] > 0) & (panel["date"] >= RECENT_START)].copy()
    y = news["label"].to_numpy()
    x = news["sentiment_agg"].to_numpy()

    # Boxplot dạng số: mean sentiment_agg theo lớp (tách rõ = tín hiệu hướng).
    by_class = (
        news.groupby("label")["sentiment_agg"].agg(["mean", "std", "count"]).round(4)
    )
    mi_real = _mi(x, y)

    # Null-test: shuffle sentiment_agg TRONG TỪNG NGÀY → phá liên kết sentiment↔nhãn nhưng GIỮ
    # cấu trúc "ngày". MI vẫn cao sau shuffle = artifact nhớ-ngày (sector-mapping), không phải tín hiệu.
    rng = np.random.default_rng(SEED)
    mi_null = []
    for _ in range(5):
        shuffled = news.copy()
        shuffled["sentiment_agg"] = shuffled.groupby("date")["sentiment_agg"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        mi_null.append(_mi(shuffled["sentiment_agg"].to_numpy(), y))
    mi_null_mean = float(np.mean(mi_null))

    # Kiểm chứng 1-mã: mã nhiều news-rows nhất (panel gộp dễ "nhớ ngày"; 1 mã thì không).
    top_sym = news["symbol"].value_counts().idxmax()
    one = news[news["symbol"] == top_sym]
    mi_one = _mi(one["sentiment_agg"].to_numpy(), one["label"].to_numpy()) if len(one) > 30 else None

    quint = _quintile_monotonic(news)
    # Pass: MI thật > null rõ rệt (tín hiệu không phải nhớ-ngày) VÀ 1-mã không sụp về ~0.
    signal_above_null = mi_real > 1.5 * mi_null_mean and mi_real > 0.002
    one_stock_survives = mi_one is None or mi_one > 0.002
    passed = bool(signal_above_null and one_stock_survives)

    out = {
        "n_news_rows": int(len(news)),
        "mean_sentiment_by_class": by_class.to_dict("index"),
        "mi_real": round(mi_real, 5),
        "mi_null_shuffled_mean": round(mi_null_mean, 5),
        "mi_ratio_real_over_null": round(mi_real / mi_null_mean, 2) if mi_null_mean else None,
        "one_stock": {"symbol": top_sym, "n": int(len(one)), "mi": round(mi_one, 5) if mi_one else None},
        "quintile": quint,
        "signal_above_null": signal_above_null,
        "one_stock_survives": one_stock_survives,
        "passed": passed,
    }
    print("\n=== TẦNG 0 — conditional signal test ===")
    print(f"news-rows (recent {RECENT_START}+): {out['n_news_rows']}")
    print("mean sentiment_agg theo lớp:", json.dumps(out["mean_sentiment_by_class"], ensure_ascii=False))
    print(f"MI thật={out['mi_real']} | MI null(shuffle trong ngày)={out['mi_null_shuffled_mean']} "
          f"| tỉ lệ={out['mi_ratio_real_over_null']}")
    print(f"1-mã {top_sym} (n={out['one_stock']['n']}): MI={out['one_stock']['mi']}")
    print(f"quintile p(tang): {quint}")
    print(f"→ TẦNG 0 {'PASS' if passed else 'FAIL'}")
    return out


# %% [markdown]
# ## Tầng 1a — Conditional model test (sanity, cơ chế Option B)
# LightGBM CHỈ trên news-rows, CÓ vs KHÔNG sentiment-feature, cùng split walk-forward.


# %%
def tier1a(panel: pd.DataFrame) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score

    news = panel[(panel["news_count"] > 0) & (panel["date"] >= RECENT_START)].copy()
    news["sector"] = news["sector"].astype("category")
    price_feats = [f for f in LGBM_V4_FEATURES]  # gồm sector
    with_sent = price_feats + ["sentiment_agg", "news_count"]

    d = news["date"]
    tr = news[d < T1A_TRAIN_END]
    te = news[d >= T1A_VAL_END]  # test out-of-time = 2026
    if len(tr) < 100 or len(te) < 50:
        print("\n=== TẦNG 1a === SKIP (news-rows quá mỏng cho split)")
        return {"skipped": True, "n_train": int(len(tr)), "n_test": int(len(te))}

    def run(feats: list[str]) -> float:
        m = LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
        m.fit(tr[feats], tr["label"], categorical_feature=["sector"])
        pred = m.predict(te[feats])
        return float(f1_score(te["label"], pred, average="macro"))

    f1_base = run(price_feats)
    f1_sent = run(with_sent)
    delta = f1_sent - f1_base
    passed = delta >= T1A_DELTA_GATE
    out = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "macro_f1_price_only": round(f1_base, 4),
        "macro_f1_with_sentiment": round(f1_sent, 4),
        "delta": round(delta, 4), "gate": T1A_DELTA_GATE, "passed": bool(passed),
    }
    print("\n=== TẦNG 1a — feature model test (news-rows) ===")
    print(f"train={out['n_train']} test={out['n_test']} (out-of-time {T1A_VAL_END}+)")
    print(f"macro-F1: giá-thuần={out['macro_f1_price_only']} | +sentiment={out['macro_f1_with_sentiment']} "
          f"| delta={out['delta']} (gate +{T1A_DELTA_GATE}) → {'PASS' if passed else 'FAIL'}")
    return out


# %% [markdown]
# ## Tầng 1b — GATING SIMULATION (gate quyết định — đúng cơ chế Option C)
# lgbm_v4 THẬT (đông cứng) + temperature từ metrics. Override is_actionable theo sentiment_extreme
# cực đoan. Đo precision-trên-actionable vs coverage theo τ × 2 chế độ. So baseline (không override).


# %%
def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-12, None)) / t
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def _sentiment_dir(extreme: float) -> str | None:
    """Hướng tin: dương→'tang', âm→'giam'. (di_ngang không có hướng tin tương ứng.)"""
    if extreme > 0:
        return "tang"
    if extreme < 0:
        return "giam"
    return None


def _gating_metrics(actionable: np.ndarray, correct: np.ndarray) -> dict:
    n = int(actionable.sum())
    prec = float(correct[actionable].mean()) if n else None
    return {
        "coverage": round(float(actionable.mean()), 4),
        "n_actionable": n,
        "precision": round(prec, 4) if prec is not None else None,
    }


def tier1b(panel: pd.DataFrame) -> dict:
    import lightgbm as lgb

    meta = json.loads((LGBM_ARTIFACT / "metrics_lgbm_v4.json").read_text())
    classes = list(meta["classes_order"])
    temperature = float(meta["calibration"]["temperature"])
    booster = lgb.Booster(model_file=str(LGBM_ARTIFACT / "lgbm_v4.txt"))

    ev = panel[panel["date"] >= GATING_EVAL_START].copy()
    ev["sector"] = ev["sector"].astype("category")
    probs = apply_temperature(np.asarray(booster.predict(ev[list(LGBM_V4_FEATURES)])), temperature)
    conf = probs.max(axis=1)
    pred = np.array([classes[i] for i in probs.argmax(axis=1)])
    actual = ev["label"].to_numpy()
    correct = pred == actual
    base_actionable = conf >= PRODUCTION_THRESHOLD
    baseline = _gating_metrics(base_actionable, correct)

    news_mask = ev["news_count"].to_numpy() > 0
    extreme = ev["sentiment_extreme"].to_numpy()
    sent_dir = np.array([_sentiment_dir(e) for e in extreme], dtype=object)

    print("\n=== TẦNG 1b — GATING SIMULATION (lgbm_v4 thật, đông cứng) ===")
    print(f"eval rows (>={GATING_EVAL_START}): {len(ev)} | news-rows: {int(news_mask.sum())}")
    print(f"BASELINE (không override) @thr {PRODUCTION_THRESHOLD}: {json.dumps(baseline, ensure_ascii=False)}")

    results = []
    best = None
    for mode in ("suppress_only", "suppress_and_promote"):
        for tau in TAU_GRID:
            act = base_actionable.copy()
            strong = news_mask & (np.abs(extreme) >= tau)
            # Trái chiều (tin mạnh ngược hướng model, hoặc model nói di_ngang mà tin mạnh) → TẮT.
            disagree = strong & (
                ((pred == "tang") & (sent_dir == "giam"))
                | ((pred == "giam") & (sent_dir == "tang"))
                | (pred == "di_ngang")
            )
            act[disagree] = False
            if mode == "suppress_and_promote":
                # Đồng chiều + conf<thr → BẬT (rủi ro hơn).
                agree = strong & (
                    ((pred == "tang") & (sent_dir == "tang"))
                    | ((pred == "giam") & (sent_dir == "giam"))
                ) & (conf < PRODUCTION_THRESHOLD)
                act[agree] = True
            m = _gating_metrics(act, correct)
            row = {"mode": mode, "tau": tau, **m,
                   "d_precision": round((m["precision"] or 0) - (baseline["precision"] or 0), 4),
                   "d_coverage": round(m["coverage"] - baseline["coverage"], 4)}
            results.append(row)
            # "Thắng": precision tăng có nghĩa (>= +0.01) và coverage không sụp quá nửa.
            if (m["precision"] is not None and row["d_precision"] >= 0.01
                    and m["coverage"] >= 0.5 * baseline["coverage"] and m["n_actionable"] >= 20):
                if best is None or row["d_precision"] > best["d_precision"]:
                    best = row

    for r in results:
        print(f"  {r['mode']:>20} τ={r['tau']}: prec={r['precision']} (Δ{r['d_precision']:+}) "
              f"cov={r['coverage']} (Δ{r['d_coverage']:+}) n={r['n_actionable']}")
    passed = best is not None
    print(f"→ TẦNG 1b {'PASS' if passed else 'FAIL'}"
          + (f" | best: {best['mode']} τ={best['tau']} "
             f"prec {baseline['precision']}→{best['precision']} cov {baseline['coverage']}→{best['coverage']}"
             if passed else " (không τ/chế độ nào nâng precision-actionable đủ)"))
    return {"baseline": baseline, "grid": results, "best": best, "passed": passed}


# %%
def main() -> dict:
    panel = load_panel()
    print(f"Panel: {len(panel)} hàng / {panel['symbol'].nunique()} mã | "
          f"news-rows tổng: {int((panel['news_count'] > 0).sum())}")
    # Chạy CẢ 3 tầng vô điều kiện: Tầng 0 đo sentiment_agg, nhưng Option C ship bằng
    # sentiment_extreme + cơ chế gating (khác) → chạy 1b trực tiếp cho bằng chứng đúng cơ chế,
    # không suy diễn từ Tầng 0. Gate mở = Tầng 0 PASS VÀ Tầng 1b PASS (1a chỉ sanity).
    t0 = tier0(panel)
    t1a = tier1a(panel)
    t1b = tier1b(panel)

    gate_open = bool(t0["passed"] and t1b.get("passed"))
    print("\n" + "=" * 60)
    print(f"KẾT LUẬN DIAGNOSTIC: {'MỞ Pha 2/3 (cào dày + Option C)' if gate_open else 'DỪNG — giữ v4 thuần, ghi âm tính'}")
    print(f"  Tầng 0 (signal): {'PASS' if t0['passed'] else 'FAIL'}")
    print(f"  Tầng 1a (feature sanity): {t1a.get('passed', t1a.get('skipped'))}")
    print(f"  Tầng 1b (GATING GATE): {t1b.get('passed', t1b.get('skipped'))}")
    print("=" * 60)

    result = {"tier0": t0, "tier1a": t1a, "tier1b": t1b, "gate_open": gate_open}
    out_path = _REPO / "ml" / "artifacts" / "lgbm_model" / "diagnostic_sentiment.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"✓ kết quả → {out_path}")
    return result


# %%
if __name__ == "__main__":
    main()
