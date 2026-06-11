"""Export training panel CSV cho Kaggle.

Chạy từ repo root:
    cd backend && uv run python -m scripts.export_training_data

Output: ml/data/training_panel.csv
Columns: date, symbol, stock_id, close, *FEATURE_COLS (ma7, ma20, rsi14, macd, macd_signal,
         ret_1d, ret_5d, abs_ret_1d, vol_5, vol_20, dist_ma20, ma_ratio, sentiment_agg,
         news_count, sector), label
(symbol = stock_id = mã chứng khoán string; notebook encode stock_id → code số khi train.
 Feature list MODEL v4 = features.builder.LGBM_V4_FEATURES — train import từ đó, không tự chế.)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

# backend là nguồn sự thật cho feature builder
_REPO = Path(__file__).resolve().parents[2]
_BACKEND = _REPO / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]  # fmt: skip

OUTPUT = _REPO / "ml" / "data" / "training_panel.csv"


async def main() -> None:
    from features.builder import load_training_frame

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for symbol in VN30:
        print(f"  {symbol}...", end=" ", flush=True)
        df = await load_training_frame(symbol)
        if df.empty:
            print("SKIP (no data)")
            continue
        # Lưu cả symbol (cột rõ ràng) + stock_id (= symbol-string). Notebook filter trên symbol;
        # stock_id được encode lại thành code số trong notebook (categories cố định theo VN30).
        df.insert(1, "symbol", symbol)
        df.insert(2, "stock_id", symbol)
        frames.append(df)
        print(f"{len(df)} rows")

    if not frames:
        print("Không có data — kiểm tra DATABASE_URL trong .env")
        sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    panel.to_csv(OUTPUT, index=False)
    print(f"\nDone: {len(panel)} rows ({len(frames)} mã) → {OUTPUT}")
    print(f"Label distribution:\n{panel['label'].value_counts(dropna=False)}")


if __name__ == "__main__":
    asyncio.run(main())
