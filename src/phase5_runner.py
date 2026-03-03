from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest_engine import (
    evaluate_external_signals,
    generate_model_oof_probas,
    model_signals_from_probas,
    walk_forward_summary,
)
from .config import CONFIG
from .data_feed import run_collection
from .features import process_features
from .labeling_triple_barrier import triple_barrier_labels
from .metrics import expectancy, max_drawdown, profit_factor, sharpe
from .train_lgbm import train as train_lgbm


def _ensure_features(symbol: str, timeframe: str, months: int = 24) -> Path:
    feat = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
    if feat.exists():
        return feat
    raw = CONFIG.data_raw_dir / f"{symbol}_{timeframe}.parquet"
    if not raw.exists():
        run_collection(symbol=symbol, timeframes=[timeframe], months=months, credentials=None)
    return process_features(symbol=symbol, timeframe=timeframe)


def _build_h4_dataset(symbol: str, pt_mult: float, sl_mult: float, horizon: int) -> pd.DataFrame:
    feat_path = _ensure_features(symbol=symbol, timeframe="H4", months=24)
    feat_df = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True)
    return triple_barrier_labels(
        feat_df, timeframe="H4", pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon
    )


def _ensure_h4_model(symbol: str, dataset_df: pd.DataFrame) -> None:
    CONFIG.ensure_dirs()
    dataset_path = CONFIG.data_processed_dir / f"{symbol}_H4_dataset.parquet"
    dataset_df.to_parquet(dataset_path, index=False)
    has_model = any(CONFIG.models_dir.glob(f"{symbol}_H4_*.pkl"))
    if not has_model:
        train_lgbm(symbol=symbol, timeframe="H4", n_splits=5)


def _get_model_details(
    df: pd.DataFrame,
    symbol: str,
    threshold: float,
    seed: int,
    cost_multiplier: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    probas, class_to_idx = generate_model_oof_probas(df=df, n_splits=5, seed=seed)
    signals = model_signals_from_probas(
        df=df,
        probas=probas,
        class_to_idx=class_to_idx,
        threshold=threshold,
        regime_vol_quantile=None,
    )
    report, details = evaluate_external_signals(
        df=df,
        signals=signals,
        symbol=symbol,
        timeframe="H4",
        experiment_type="model_phase5",
        params={"threshold": threshold, "seed": seed},
        sanity_extra={"oof_no_in_sample_backtest": True},
        cost_multiplier=cost_multiplier,
    )
    return report, details, signals


def _phase5a_walkforward(details: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    wf_rows = walk_forward_summary(details, windows=8)
    wf_df = pd.DataFrame(wf_rows)
    pf_ok = float((wf_df["profit_factor"] > 1.0).mean()) if not wf_df.empty else 0.0
    dd_ok = bool((wf_df["max_drawdown"] >= -0.05).all()) if not wf_df.empty else False
    sharpe_avg = float(wf_df["sharpe"].mean()) if not wf_df.empty else 0.0
    result = {
        "windows": int(len(wf_df)),
        "pf_gt_1_ratio": pf_ok,
        "all_dd_above_minus_5pct": dd_ok,
        "avg_sharpe": sharpe_avg,
        "approved": bool((pf_ok >= 0.60) and dd_ok and (sharpe_avg > 0.5)),
    }
    return wf_df, result


def _phase5b_bootstrap(
    details: pd.DataFrame,
    out_dir: Path,
    simulations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    trades = details[details["signal"] != 0]["ret"].values
    if len(trades) == 0:
        raise RuntimeError("No trades for bootstrap")
    rng = np.random.default_rng(seed)
    pf_vals, dd_vals, cum_vals = [], [], []
    n = len(trades)
    for _ in range(simulations):
        sample = rng.choice(trades, size=n, replace=True)
        s = pd.Series(sample)
        pf_vals.append(profit_factor(s))
        dd_vals.append(max_drawdown(s))
        cum_vals.append(float(np.sum(sample)))
    out = {
        "simulations": simulations,
        "seed": seed,
        "n_trades_sampled": n,
        "profit_factor": {
            "p05": float(np.percentile(pf_vals, 5)),
            "p50": float(np.percentile(pf_vals, 50)),
            "p95": float(np.percentile(pf_vals, 95)),
            "mean": float(np.mean(pf_vals)),
        },
        "max_drawdown": {
            "p05": float(np.percentile(dd_vals, 5)),
            "p50": float(np.percentile(dd_vals, 50)),
            "p95": float(np.percentile(dd_vals, 95)),
            "mean": float(np.mean(dd_vals)),
        },
        "cum_return": {
            "p05": float(np.percentile(cum_vals, 5)),
            "p50": float(np.percentile(cum_vals, 50)),
            "p95": float(np.percentile(cum_vals, 95)),
            "mean": float(np.mean(cum_vals)),
        },
    }
    out_path = out_dir / "phase5_bootstrap_H4.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Histogram artifacts (PNG if matplotlib exists; fallback CSV bins).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        for name, values in (
            ("pf", pf_vals),
            ("max_drawdown", dd_vals),
            ("cum_return", cum_vals),
        ):
            plt.figure(figsize=(8, 4))
            plt.hist(values, bins=50)
            plt.title(f"Bootstrap Histogram: {name}")
            plt.tight_layout()
            plt.savefig(out_dir / f"phase5_bootstrap_hist_{name}_H4.png")
            plt.close()
    except Exception:
        for name, values in (
            ("pf", pf_vals),
            ("max_drawdown", dd_vals),
            ("cum_return", cum_vals),
        ):
            hist, edges = np.histogram(values, bins=50)
            csv = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})
            csv.to_csv(out_dir / f"phase5_bootstrap_hist_{name}_H4.csv", index=False)
    return out


def _phase5c_cost_stress(
    df: pd.DataFrame,
    symbol: str,
    threshold: float,
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for mult, label in ((1.0, "base"), (1.25, "+25%"), (1.5, "+50%"), (2.0, "+100%")):
        report, _, _ = _get_model_details(
            df=df, symbol=symbol, threshold=threshold, seed=seed, cost_multiplier=mult
        )
        rows.append(
            {
                "scenario": label,
                "cost_multiplier": mult,
                "trades": report["trades"],
                "profit_factor": report["profit_factor"],
                "expectancy": report["expectancy"],
                "sharpe": report["sharpe"],
                "max_drawdown": report["max_drawdown"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "phase5_cost_stress_H4.csv", index=False)
    return out_df


def _phase5d_paper_mode(symbol: str, out_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "src.bot_live",
        "--symbol",
        symbol,
        "--paper",
        "--timeframe",
        "H4",
        "--paper-bars",
        "1200",
    ]
    subprocess.run(cmd, check=True, cwd=str(CONFIG.root_dir))
    path = out_dir / "phase5_paper_log.json"
    if not path.exists():
        # fallback if bot_live saved in default reports path with same name
        path = CONFIG.reports_dir / "phase5_paper_log.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase5(symbol: str, threshold: float, pt_mult: float, sl_mult: float, horizon: int, seed: int) -> Path:
    CONFIG.ensure_dirs()
    out_dir = CONFIG.reports_dir
    df = _build_h4_dataset(symbol=symbol, pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon)
    _ensure_h4_model(symbol=symbol, dataset_df=df)

    # Base report/details
    base_report, base_details, _ = _get_model_details(
        df=df,
        symbol=symbol,
        threshold=threshold,
        seed=seed,
        cost_multiplier=1.0,
    )

    # 5A
    wf_df, wf_result = _phase5a_walkforward(base_details)
    wf_df.to_csv(out_dir / "phase5_walkforward_H4.csv", index=False)
    (out_dir / "phase5_walkforward_H4.json").write_text(
        json.dumps({"windows": wf_df.to_dict(orient="records"), "summary": wf_result}, indent=2),
        encoding="utf-8",
    )

    # 5B
    bootstrap_result = _phase5b_bootstrap(base_details, out_dir=out_dir, simulations=2000, seed=seed)

    # 5C
    cost_df = _phase5c_cost_stress(df=df, symbol=symbol, threshold=threshold, seed=seed, out_dir=out_dir)
    cost_25_ok = bool(float(cost_df.loc[cost_df["scenario"] == "+25%", "profit_factor"].iloc[0]) >= 1.0)

    # 5D
    paper = _phase5d_paper_mode(symbol=symbol, out_dir=out_dir)

    # Summary
    recommendation = "APTO PARA MICRO-LOTE LIVE"
    if not (wf_result["approved"] and cost_25_ok and bootstrap_result["profit_factor"]["p50"] > 1.0):
        recommendation = "NECESSITA REVISÃO"
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
        "base_report": base_report,
        "walkforward": wf_result,
        "bootstrap": bootstrap_result,
        "cost_stress": cost_df.to_dict(orient="records"),
        "paper": {
            "paper_trades": paper.get("paper_trades"),
            "avg_latency_ms": paper.get("avg_latency_ms"),
            "avg_sim_slippage": paper.get("avg_sim_slippage"),
            "paper_expectancy": paper.get("paper_expectancy"),
            "paper_cum_return": paper.get("paper_cum_return"),
            "divergence_vs_backtest_expectancy": float(
                (paper.get("paper_expectancy") or 0.0) - base_report["expectancy"]
            ),
        },
        "criteria": {
            "walkforward_min_60pct_pf_gt_1": wf_result["pf_gt_1_ratio"] >= 0.60,
            "walkforward_no_dd_below_minus_5pct": wf_result["all_dd_above_minus_5pct"],
            "walkforward_avg_sharpe_gt_0_5": wf_result["avg_sharpe"] > 0.5,
            "cost_plus_25_pf_ge_1": cost_25_ok,
            "bootstrap_pf_p50_gt_1": bootstrap_result["profit_factor"]["p50"] > 1.0,
        },
        "recommendation": recommendation,
    }
    out_path = out_dir / "phase5_summary_H4.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 robustness validation for fixed H4 setup.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase5(
        symbol=args.symbol,
        threshold=args.threshold,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        horizon=args.horizon,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
