from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .backtest_engine import (
    evaluate_external_signals,
    generate_model_oof_probas,
    model_signals_from_probas,
    walk_forward_summary,
)
from .config import CONFIG
from .features import process_features
from .labeling_triple_barrier import triple_barrier_labels
from .metrics import max_drawdown, profit_factor


def _ensure_h4_features(symbol: str) -> Path:
    path = CONFIG.data_processed_dir / f"{symbol}_H4_features.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=["time"])
        _ = df  # keep quick probe to ensure file readable
        full = pd.read_parquet(path)
        required = {"ADX_14", "ma50_slope", "rolling_vol_20", "is_trend", "is_high_vol", "is_sideways"}
        if required.issubset(set(full.columns)):
            return path
    return process_features(symbol=symbol, timeframe="H4")


def _bootstrap(details: pd.DataFrame, sims: int = 2000, seed: int = 42) -> dict:
    trades = details[details["signal"] != 0]["ret"].values
    if len(trades) == 0:
        return {
            "simulations": sims,
            "n_trades_sampled": 0,
            "profit_factor": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
            "max_drawdown": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
            "cum_return": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
        }
    rng = np.random.default_rng(seed)
    pf_vals, dd_vals, cum_vals = [], [], []
    n = len(trades)
    for _ in range(sims):
        sample = rng.choice(trades, size=n, replace=True)
        s = pd.Series(sample)
        pf_vals.append(profit_factor(s))
        dd_vals.append(max_drawdown(s))
        cum_vals.append(float(np.sum(sample)))
    return {
        "simulations": sims,
        "n_trades_sampled": n,
        "profit_factor": {
            "p05": float(np.percentile(pf_vals, 5)),
            "p50": float(np.percentile(pf_vals, 50)),
            "p95": float(np.percentile(pf_vals, 95)),
        },
        "max_drawdown": {
            "p05": float(np.percentile(dd_vals, 5)),
            "p50": float(np.percentile(dd_vals, 50)),
            "p95": float(np.percentile(dd_vals, 95)),
        },
        "cum_return": {
            "p05": float(np.percentile(cum_vals, 5)),
            "p50": float(np.percentile(cum_vals, 50)),
            "p95": float(np.percentile(cum_vals, 95)),
        },
    }


def _cost_stress(
    df: pd.DataFrame,
    signals: np.ndarray,
    symbol: str,
    scenario: str,
) -> list[dict]:
    out = []
    for mult, name in ((1.0, "base"), (1.25, "+25%"), (1.5, "+50%"), (2.0, "+100%")):
        rep, _ = evaluate_external_signals(
            df=df,
            signals=signals,
            symbol=symbol,
            timeframe="H4",
            experiment_type=f"phase6_{scenario}",
            params={"cost_scenario": name, "cost_multiplier": mult},
            sanity_extra={"oof_no_in_sample_backtest": True},
            cost_multiplier=mult,
        )
        out.append(
            {
                "scenario": name,
                "cost_multiplier": mult,
                "trades": rep["trades"],
                "profit_factor": rep["profit_factor"],
                "expectancy": rep["expectancy"],
                "sharpe": rep["sharpe"],
                "max_drawdown": rep["max_drawdown"],
            }
        )
    return out


def run_phase6(
    symbol: str = "EURUSD",
    threshold: float = 0.6,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 8,
    seed: int = 42,
) -> Path:
    CONFIG.ensure_dirs()
    feat_path = _ensure_h4_features(symbol)
    feat_df = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True)
    df = triple_barrier_labels(
        feat_df, timeframe="H4", pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon
    )

    probas, class_to_idx = generate_model_oof_probas(df=df, n_splits=5, seed=seed)
    base_signals = model_signals_from_probas(
        df=df,
        probas=probas,
        class_to_idx=class_to_idx,
        threshold=threshold,
        regime_vol_quantile=None,
    )

    scenarios: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "trend_only": lambda x: x["is_trend"].astype(bool),
        "high_vol_only": lambda x: x["is_high_vol"].astype(bool),
        "trend_and_high_vol": lambda x: (x["is_trend"] & x["is_high_vol"]).astype(bool),
        "except_sideways": lambda x: (~x["is_sideways"]).astype(bool),
    }

    results = []
    for name, mask_fn in scenarios.items():
        mask = mask_fn(df).values
        signals = base_signals.copy()
        signals[~mask] = 0
        report, details = evaluate_external_signals(
            df=df,
            signals=signals,
            symbol=symbol,
            timeframe="H4",
            experiment_type=f"phase6_{name}",
            params={
                "threshold": threshold,
                "pt_mult": pt_mult,
                "sl_mult": sl_mult,
                "horizon": horizon,
                "seed": seed,
                "regime_filter": name,
            },
            sanity_extra={"oof_no_in_sample_backtest": True},
            cost_multiplier=1.0,
        )

        wf_rows = walk_forward_summary(details, windows=8)
        wf_df = pd.DataFrame(wf_rows)
        wf_pf_ratio = float((wf_df["profit_factor"] > 1.0).mean()) if not wf_df.empty else 0.0
        wf_pass = wf_pf_ratio >= 0.60

        bootstrap = _bootstrap(details, sims=2000, seed=seed)
        stress = _cost_stress(df=df, signals=signals, symbol=symbol, scenario=name)

        results.append(
            {
                "scenario": name,
                "report": report,
                "walkforward": {
                    "windows": wf_rows,
                    "pf_gt_1_ratio": wf_pf_ratio,
                    "pass_60pct": wf_pass,
                },
                "bootstrap": bootstrap,
                "cost_stress": stress,
            }
        )

    ranked = sorted(
        results,
        key=lambda x: (
            x["walkforward"]["pf_gt_1_ratio"],
            x["report"]["profit_factor"],
            x["report"]["expectancy"],
            x["report"]["sharpe"],
        ),
        reverse=True,
    )
    best = ranked[0]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": "H4",
        "params": {
            "threshold": threshold,
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "horizon": horizon,
            "seed": seed,
        },
        "criteria": {
            "walkforward_required_pf_gt_1_ratio": 0.60,
            "best_pf_ratio": best["walkforward"]["pf_gt_1_ratio"],
            "walkforward_passed": best["walkforward"]["pass_60pct"],
        },
        "best_scenario": best["scenario"],
        "results": results,
        "recommendation": "APTO PARA MICRO-LOTE LIVE"
        if best["walkforward"]["pass_60pct"]
        else "NECESSITA REVISÃO",
    }

    out_path = CONFIG.reports_dir / "phase6_summary_H4.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 regime activation filter runner for H4.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase6(
        symbol=args.symbol,
        threshold=args.threshold,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        horizon=args.horizon,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
