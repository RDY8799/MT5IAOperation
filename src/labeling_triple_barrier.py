from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CONFIG


def triple_barrier_labels(
    df: pd.DataFrame,
    timeframe: str,
    pt_mult: float | None = None,
    sl_mult: float | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    n = horizon or CONFIG.triple_barrier.horizon_by_tf[timeframe]
    k_pt = pt_mult if pt_mult is not None else CONFIG.triple_barrier.pt_atr_mult
    k_sl = sl_mult if sl_mult is not None else CONFIG.triple_barrier.sl_atr_mult

    labels = []
    t1 = []
    pt_used = []
    sl_used = []

    for i in range(len(out)):
        if i + 1 >= len(out):
            labels.append(0)
            t1.append(out.iloc[i]["time"])
            pt_used.append(float("nan"))
            sl_used.append(float("nan"))
            continue

        price0 = out.iloc[i]["close"]
        atr0 = out.iloc[i]["ATR_14"]
        up = price0 + (k_pt * atr0)
        dn = price0 - (k_sl * atr0)
        end_idx = min(i + n, len(out) - 1)
        future = out.iloc[i + 1 : end_idx + 1]

        label = 0
        end_time = out.iloc[end_idx]["time"]
        for _, row in future.iterrows():
            if row["high"] >= up:
                label = 1
                end_time = row["time"]
                break
            if row["low"] <= dn:
                label = -1
                end_time = row["time"]
                break

        labels.append(label)
        t1.append(end_time)
        pt_used.append(k_pt * atr0)
        sl_used.append(k_sl * atr0)

    out["y"] = labels
    out["t1"] = pd.to_datetime(t1, utc=True)
    out["pt"] = pt_used
    out["sl"] = sl_used
    return out.dropna().reset_index(drop=True)


def build_dataset(symbol: str, timeframe: str) -> Path:
    CONFIG.ensure_dirs()
    feats_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(feats_path)
    df = pd.read_parquet(feats_path)
    dataset = triple_barrier_labels(df, timeframe=timeframe)
    out_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_dataset.parquet"
    dataset.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate triple barrier labels.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_dataset(symbol=args.symbol, timeframe=args.tf)
    print(f"saved={out}")


if __name__ == "__main__":
    main()

