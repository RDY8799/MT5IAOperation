from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pandas as pd

from .backtest_engine import backtest_on_dataset
from .config import CONFIG
from .labeling_triple_barrier import triple_barrier_labels


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_grid(
    symbol: str,
    timeframe: str,
    thresholds: list[float],
    pt_mults: list[float],
    sl_mults: list[float],
    horizons: list[int],
    regime_quantiles: list[float | None],
) -> Path:
    CONFIG.ensure_dirs()
    feat_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    features_df = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True)

    rows: list[dict] = []
    total = len(thresholds) * len(pt_mults) * len(sl_mults) * len(horizons) * len(regime_quantiles)
    n = 0
    for thr, pt, sl, horizon, regime_q in product(
        thresholds, pt_mults, sl_mults, horizons, regime_quantiles
    ):
        n += 1
        print(
            f"[{n}/{total}] threshold={thr:.2f} pt={pt:.2f} sl={sl:.2f} "
            f"horizon={horizon} regime_q={regime_q}"
        )
        dataset_df = triple_barrier_labels(
            features_df,
            timeframe=timeframe,
            pt_mult=pt,
            sl_mult=sl,
            horizon=horizon,
        )
        report = backtest_on_dataset(
            df=dataset_df,
            symbol=symbol,
            timeframe=timeframe,
            threshold=thr,
            regime_vol_quantile=regime_q,
        )
        rows.append(
            {
                "threshold": thr,
                "pt_mult": pt,
                "sl_mult": sl,
                "horizon": horizon,
                "regime_q": regime_q if regime_q is not None else -1.0,
                "trades": report["trades"],
                "sharpe": report["sharpe"],
                "sortino": report["sortino"],
                "profit_factor": report["profit_factor"],
                "expectancy": report["expectancy"],
                "max_drawdown": report["max_drawdown"],
            }
        )

    result_df = pd.DataFrame(rows).sort_values(
        ["profit_factor", "expectancy", "sharpe"], ascending=[False, False, False]
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = CONFIG.reports_dir / f"grid_{symbol}_{timeframe}_{ts}.csv"
    json_path = CONFIG.reports_dir / f"grid_{symbol}_{timeframe}_{ts}.json"
    result_df.to_csv(csv_path, index=False)
    json_path.write_text(result_df.to_json(orient="records", indent=2), encoding="utf-8")

    top_path = CONFIG.reports_dir / f"grid_{symbol}_{timeframe}_{ts}_top10.json"
    top10 = result_df.head(10).to_dict(orient="records")
    top_path.write_text(json.dumps(top10, indent=2), encoding="utf-8")
    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_top10={top_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled strategy improvement grid runner.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    parser.add_argument("--thresholds", default="0.60,0.65,0.70")
    parser.add_argument("--pt-mults", default="1.5,2.0")
    parser.add_argument("--sl-mults", default="1.0")
    parser.add_argument("--horizons", default="4,6,8")
    parser.add_argument("--regime-quantiles", default="-1,0.5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regime_raw = _parse_float_list(args.regime_quantiles)
    regime_quantiles = [None if q < 0 else q for q in regime_raw]
    run_grid(
        symbol=args.symbol,
        timeframe=args.tf,
        thresholds=_parse_float_list(args.thresholds),
        pt_mults=_parse_float_list(args.pt_mults),
        sl_mults=_parse_float_list(args.sl_mults),
        horizons=_parse_int_list(args.horizons),
        regime_quantiles=regime_quantiles,
    )


if __name__ == "__main__":
    main()

