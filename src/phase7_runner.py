from __future__ import annotations

import argparse
import json
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
from .labeling_triple_barrier import build_dataset, triple_barrier_labels
from .multitf import BUY, SELL, WAIT, align_h4_to_h1, apply_policy
from .train_lgbm import train as train_lgbm


def _ensure_features(symbol: str, tf: str, months: int = 24) -> Path:
    path = CONFIG.data_processed_dir / f"{symbol}_{tf}_features.parquet"
    if path.exists():
        return path
    raw = CONFIG.data_raw_dir / f"{symbol}_{tf}.parquet"
    if not raw.exists():
        run_collection(symbol=symbol, timeframes=[tf], months=months, credentials=None)
    return process_features(symbol=symbol, timeframe=tf)


def _ensure_dataset(symbol: str, tf: str) -> Path:
    path = CONFIG.data_processed_dir / f"{symbol}_{tf}_dataset.parquet"
    if path.exists():
        return path
    _ensure_features(symbol, tf)
    if tf == "H4":
        # H4 fixed setup from Fase 5/6
        feat = pd.read_parquet(CONFIG.data_processed_dir / f"{symbol}_{tf}_features.parquet")
        ds = triple_barrier_labels(feat, timeframe=tf, pt_mult=2.0, sl_mult=1.0, horizon=8)
        ds.to_parquet(path, index=False)
    else:
        build_dataset(symbol=symbol, timeframe=tf)
    return path


def _ensure_model(symbol: str, tf: str) -> None:
    if any(CONFIG.models_dir.glob(f"{symbol}_{tf}_*.pkl")):
        return
    _ensure_dataset(symbol, tf)
    train_lgbm(symbol=symbol, timeframe=tf, n_splits=5)


def _probas_to_buy_sell(
    probas: np.ndarray,
    class_to_idx: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    idx_buy = class_to_idx[BUY] if BUY in class_to_idx else class_to_idx[max(class_to_idx.keys())]
    idx_sell = class_to_idx[SELL] if SELL in class_to_idx else class_to_idx[min(class_to_idx.keys())]
    return probas[:, idx_buy], probas[:, idx_sell]


def _bootstrap(details: pd.DataFrame, sims: int = 2000, seed: int = 42) -> dict[str, Any]:
    trades = details[details["signal"] != 0]["ret"].values
    if len(trades) == 0:
        return {"simulations": sims, "n_trades_sampled": 0}
    rng = np.random.default_rng(seed)
    pf_vals = []
    for _ in range(sims):
        sample = rng.choice(trades, size=len(trades), replace=True)
        gross_profit = sample[sample > 0].sum()
        gross_loss = -sample[sample < 0].sum()
        pf_vals.append(float(gross_profit / gross_loss) if gross_loss > 0 else 0.0)
    return {
        "simulations": sims,
        "n_trades_sampled": int(len(trades)),
        "pf_p05": float(np.percentile(pf_vals, 5)),
        "pf_p50": float(np.percentile(pf_vals, 50)),
        "pf_p95": float(np.percentile(pf_vals, 95)),
    }


def run_phase7(symbol: str = "EURUSD", seed: int = 42) -> Path:
    CONFIG.ensure_dirs()
    out_dir = CONFIG.reports_dir
    h1_ds = pd.read_parquet(_ensure_dataset(symbol, "H1")).sort_values("time").reset_index(drop=True)
    h4_ds = pd.read_parquet(_ensure_dataset(symbol, "H4")).sort_values("time").reset_index(drop=True)
    _ensure_model(symbol, "H1")
    _ensure_model(symbol, "H4")

    h1_proba, h1_map = generate_model_oof_probas(h1_ds, n_splits=5, seed=seed)
    h4_proba, h4_map = generate_model_oof_probas(h4_ds, n_splits=5, seed=seed)
    h1_signals = model_signals_from_probas(h1_ds, h1_proba, h1_map, threshold=0.60)
    h4_signals = model_signals_from_probas(h4_ds, h4_proba, h4_map, threshold=0.60)
    h1_buy, h1_sell = _probas_to_buy_sell(h1_proba, h1_map)
    h4_buy, h4_sell = _probas_to_buy_sell(h4_proba, h4_map)

    h1_timeline = h1_ds[["time", "y", "spread"]].copy()
    h1_timeline["signal_h1"] = h1_signals
    h1_timeline["prob_h1_buy"] = h1_buy
    h1_timeline["prob_h1_sell"] = h1_sell

    h4_timeline = h4_ds[["time"]].copy()
    h4_timeline["signal_h4_aligned"] = h4_signals
    h4_timeline["prob_h4_buy_aligned"] = h4_buy
    h4_timeline["prob_h4_sell_aligned"] = h4_sell

    aligned = align_h4_to_h1(h1_timeline, h4_timeline, "time", "time")
    aligned["signal_h4_aligned"] = aligned["signal_h4_aligned"].fillna(0).astype(int)
    aligned["prob_h4_buy_aligned"] = aligned["prob_h4_buy_aligned"].fillna(0.0)
    aligned["prob_h4_sell_aligned"] = aligned["prob_h4_sell_aligned"].fillna(0.0)

    policies = ["H4_GATE_DIRECTION", "DOUBLE_CONFIRMATION", "ENSEMBLE_SCORE"]
    grid_rows = []
    audits = []

    for policy in policies:
        decisions, reasons, enriched = apply_policy(aligned, policy=policy, threshold_final=0.60)
        report, details = evaluate_external_signals(
            df=aligned,
            signals=decisions,
            symbol=symbol,
            timeframe="H1",
            experiment_type=f"phase7_{policy}",
            params={"policy": policy, "threshold_final": 0.60, "seed": seed},
            sanity_extra={"oof_no_in_sample_backtest": True, "multitf_no_lookahead_alignment": True},
            cost_multiplier=1.0,
        )
        wf = walk_forward_summary(details, windows=8)
        wf_df = pd.DataFrame(wf)
        pf_ratio = float((wf_df["profit_factor"] > 1.0).mean()) if not wf_df.empty else 0.0
        dd_ok = bool((wf_df["max_drawdown"] >= -0.05).all()) if not wf_df.empty else False
        boot = _bootstrap(details, sims=2000, seed=seed)

        report_25, _ = evaluate_external_signals(
            df=aligned,
            signals=decisions,
            symbol=symbol,
            timeframe="H1",
            experiment_type=f"phase7_{policy}_stress25",
            params={"policy": policy, "cost_plus": 25},
            sanity_extra={"oof_no_in_sample_backtest": True, "multitf_no_lookahead_alignment": True},
            cost_multiplier=1.25,
        )

        audit = {
            "policy": policy,
            "report": report,
            "walk_forward": {
                "windows": wf,
                "pf_gt_1_ratio": pf_ratio,
                "dd_windows_ok": dd_ok,
            },
            "bootstrap": boot,
            "stress_25": {
                "profit_factor": report_25["profit_factor"],
                "expectancy": report_25["expectancy"],
                "max_drawdown": report_25["max_drawdown"],
            },
            "blocked_reason_counts": pd.Series(reasons).value_counts().to_dict(),
            "sample_alignment": enriched[
                [
                    "time",
                    "signal_h1",
                    "signal_h4_aligned",
                    "decision_final",
                    "blocked_reason",
                    "prob_h1_buy",
                    "prob_h1_sell",
                    "prob_h4_buy_aligned",
                    "prob_h4_sell_aligned",
                    "score_buy",
                    "score_sell",
                ]
            ]
            .head(50)
            .to_dict(orient="records"),
        }
        audit_path = out_dir / f"phase7_policy_{policy}.json"
        audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        audits.append({"policy": policy, "audit_path": str(audit_path), **audit})

        grid_rows.append(
            {
                "policy": policy,
                "trades": report["trades"],
                "profit_factor": report["profit_factor"],
                "expectancy": report["expectancy"],
                "sharpe": report["sharpe"],
                "max_drawdown": report["max_drawdown"],
                "pf_gt_1_ratio": pf_ratio,
                "stress25_pf": report_25["profit_factor"],
                "dd_windows_ok": dd_ok,
                "sanity_ok": report["sanity_ok"],
                "class_distribution": json.dumps(report["class_distribution"], separators=(",", ":")),
                "audit_json_path": str(audit_path),
            }
        )

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["pf_gt_1_ratio", "profit_factor", "expectancy", "sharpe", "max_drawdown"],
        ascending=[False, False, False, False, False],
    )
    grid_path = out_dir / "phase7_multitf_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best = grid_df.iloc[0].to_dict()
    approved = (
        best["pf_gt_1_ratio"] >= 0.60
        and best["stress25_pf"] > 1.0
        and bool(best["dd_windows_ok"])
    )
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "setup": {
            "tf_entry": "H1",
            "tf_gate": "H4",
            "threshold_h1": 0.60,
            "threshold_h4": 0.60,
            "ensemble_threshold_final": 0.60,
            "seed": seed,
        },
        "criteria": {
            "pf_gt_1_ratio_min": 0.60,
            "stress25_pf_gt_1": True,
            "dd_window_floor": -0.05,
        },
        "best_policy": best["policy"],
        "best_metrics": best,
        "approved": approved,
        "recommendation": "APTO PARA MICRO-LOTE LIVE" if approved else "NECESSITA REVISÃO",
        "grid_csv": str(grid_path),
    }
    summary_path = out_dir / "phase7_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"saved_grid={grid_path}")
    print(f"saved_summary={summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 multi-timeframe validation runner.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase7(symbol=args.symbol, seed=args.seed)


if __name__ == "__main__":
    main()

