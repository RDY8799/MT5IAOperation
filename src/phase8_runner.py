from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .backtest_engine import DROP_COLS, evaluate_external_signals
from .config import CONFIG
from .data_feed import run_collection
from .features import process_features
from .labeling_triple_barrier import build_dataset, triple_barrier_labels
from .metrics import expectancy, max_drawdown, profit_factor, sharpe, sortino
from .multitf import BUY, SELL, WAIT, align_h4_to_h1, apply_policy

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


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
        feat = pd.read_parquet(CONFIG.data_processed_dir / f"{symbol}_{tf}_features.parquet")
        ds = triple_barrier_labels(feat, timeframe=tf, pt_mult=2.0, sl_mult=1.0, horizon=8)
        ds.to_parquet(path, index=False)
    else:
        build_dataset(symbol=symbol, timeframe=tf)
    return path


def _bootstrap(details: pd.DataFrame, sims: int = 2000, seed: int = 42) -> dict[str, Any]:
    trades = details[details["signal"] != 0]["ret"].values
    if len(trades) == 0:
        return {"simulations": sims, "n_trades_sampled": 0}
    rng = np.random.default_rng(seed)
    pf_vals = []
    dd_vals = []
    cum_vals = []
    for _ in range(sims):
        sample = rng.choice(trades, size=len(trades), replace=True)
        gross_profit = sample[sample > 0].sum()
        gross_loss = -sample[sample < 0].sum()
        pf_vals.append(float(gross_profit / gross_loss) if gross_loss > 0 else 0.0)
        cum = np.cumsum(sample)
        dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
        dd_vals.append(dd)
        cum_vals.append(float(sample.sum()))
    return {
        "simulations": sims,
        "n_trades_sampled": int(len(trades)),
        "pf_p05": float(np.percentile(pf_vals, 5)),
        "pf_p50": float(np.percentile(pf_vals, 50)),
        "pf_p95": float(np.percentile(pf_vals, 95)),
        "max_drawdown_p05": float(np.percentile(dd_vals, 5)),
        "max_drawdown_p50": float(np.percentile(dd_vals, 50)),
        "cum_return_p05": float(np.percentile(cum_vals, 5)),
        "cum_return_p50": float(np.percentile(cum_vals, 50)),
        "cum_return_p95": float(np.percentile(cum_vals, 95)),
    }


def _window_bounds(
    df_h1: pd.DataFrame,
    windows: int,
    start_time: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    scoped = df_h1
    if start_time is not None:
        scoped = scoped[scoped["time"] >= start_time].copy()
    chunks = np.array_split(np.arange(len(scoped)), windows)
    bounds: list[dict[str, Any]] = []
    for i, idx in enumerate(chunks, start=1):
        if len(idx) == 0:
            continue
        bounds.append(
            {
                "window": i,
                "start_time": pd.Timestamp(scoped.iloc[idx[0]]["time"]),
                "end_time": pd.Timestamp(scoped.iloc[idx[-1]]["time"]),
                "n_rows": int(len(idx)),
            }
        )
    return bounds


def _fit_predict_window(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    threshold: float,
    seed: int,
) -> tuple[pd.DataFrame, str | None]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    work = df.sort_values("time").reset_index(drop=True)
    test = work[(work["time"] >= test_start) & (work["time"] <= test_end)].copy()
    train = work[work["time"] < test_start].copy()
    if test.empty:
        return pd.DataFrame(columns=["time", "signal", "prob_buy", "prob_sell", "has_pred"]), "empty_test_window"
    if len(train) < 100:
        out = test[["time"]].copy()
        out["signal"] = WAIT
        out["prob_buy"] = np.nan
        out["prob_sell"] = np.nan
        out["has_pred"] = False
        return out, "insufficient_train_rows"

    feature_cols = [c for c in work.columns if c not in DROP_COLS]
    classes = sorted(train["y"].astype(int).unique().tolist())
    if len(classes) < 2:
        out = test[["time"]].copy()
        out["signal"] = WAIT
        out["prob_buy"] = np.nan
        out["prob_sell"] = np.nan
        out["has_pred"] = False
        return out, "insufficient_train_classes"

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = train["y"].astype(int).map(class_to_idx).values
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(w) for i, w in zip(np.unique(y_train), cw)}
    objective = "binary" if len(classes) == 2 else "multiclass"
    model = lgb.LGBMClassifier(
        objective=objective,
        num_class=len(classes) if len(classes) > 2 else None,
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        class_weight=class_weights,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    if isinstance(proba, list):  # pragma: no cover
        proba = np.asarray(proba)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = np.vstack([1.0 - proba, proba]).T

    idx_buy = class_to_idx.get(BUY)
    idx_sell = class_to_idx.get(SELL)
    p_buy = proba[:, idx_buy] if idx_buy is not None else np.full(len(test), np.nan, dtype=float)
    p_sell = proba[:, idx_sell] if idx_sell is not None else np.full(len(test), np.nan, dtype=float)

    signal = np.zeros(len(test), dtype=int)
    has_pred = ~(np.isnan(p_buy) | np.isnan(p_sell))
    valid_idx = np.where(has_pred)[0]
    for i in valid_idx:
        if p_buy[i] >= threshold and p_buy[i] > p_sell[i]:
            signal[i] = BUY
        elif p_sell[i] >= threshold and p_sell[i] > p_buy[i]:
            signal[i] = SELL

    out = test[["time"]].copy()
    out["signal"] = signal
    out["prob_buy"] = p_buy
    out["prob_sell"] = p_sell
    out["has_pred"] = has_pred
    return out, None


def _aggregate_report(
    details: pd.DataFrame,
    symbol: str,
    policy: str,
    sanity_extra: dict[str, bool] | None = None,
) -> dict[str, Any]:
    if details.empty:
        sanity = {
            "spread_included": True,
            "slippage_included": True,
            "commission_included": True,
            "signal_on_close_execute_next_bar": True,
            "no_lookahead_in_execution": True,
        }
        if sanity_extra:
            sanity.update(sanity_extra)
        return {
            "symbol": symbol,
            "timeframe": "H1",
            "experiment_type": f"phase8_{policy}",
            "params": {"policy": policy},
            "trades": 0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "class_distribution": {"buy_pct": 0.0, "sell_pct": 0.0, "wait_pct": 1.0},
            "conditional_returns": {},
            "sanity": sanity,
            "sanity_ok": bool(all(sanity.values())),
            "cost_multiplier": 1.0,
        }

    returns = details["ret"].astype(float)
    trades = details[details["signal"] != WAIT]
    wins = trades["ret"][trades["ret"] > 0]
    losses = trades["ret"][trades["ret"] < 0]
    class_dist = {
        "buy_pct": float((details["signal"] == BUY).mean()),
        "sell_pct": float((details["signal"] == SELL).mean()),
        "wait_pct": float((details["signal"] == WAIT).mean()),
    }
    conditional = {}
    for cls_name, cls_val in (("BUY", BUY), ("SELL", SELL), ("WAIT", WAIT)):
        cls_ret = details.loc[details["signal"] == cls_val, "ret"]
        cls_win = cls_ret[cls_ret > 0]
        cls_loss = cls_ret[cls_ret < 0]
        conditional[cls_name] = {
            "count": int(len(cls_ret)),
            "E_r": float(cls_ret.mean()) if len(cls_ret) else 0.0,
            "hit_rate": float((cls_ret > 0).mean()) if len(cls_ret) else 0.0,
            "avg_win": float(cls_win.mean()) if len(cls_win) else 0.0,
            "avg_loss": float(cls_loss.mean()) if len(cls_loss) else 0.0,
        }
    sanity = {
        "spread_included": True,
        "slippage_included": True,
        "commission_included": True,
        "signal_on_close_execute_next_bar": True,
        "no_lookahead_in_execution": True,
    }
    if sanity_extra:
        sanity.update(sanity_extra)
    return {
        "symbol": symbol,
        "timeframe": "H1",
        "experiment_type": f"phase8_{policy}",
        "params": {"policy": policy},
        "trades": int(len(trades)),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": float((trades["ret"] > 0).mean()) if len(trades) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "class_distribution": class_dist,
        "conditional_returns": conditional,
        "sanity": sanity,
        "sanity_ok": bool(all(sanity.values())),
        "cost_multiplier": 1.0,
    }


def run_phase8(symbol: str = "EURUSD", seed: int = 42, windows: int = 8) -> Path:
    CONFIG.ensure_dirs()
    out_dir = CONFIG.reports_dir
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    h1_ds = pd.read_parquet(_ensure_dataset(symbol, "H1")).sort_values("time").reset_index(drop=True)
    h4_ds = pd.read_parquet(_ensure_dataset(symbol, "H4")).sort_values("time").reset_index(drop=True)
    h1_ds["time"] = pd.to_datetime(h1_ds["time"], utc=True)
    h4_ds["time"] = pd.to_datetime(h4_ds["time"], utc=True)

    policy = "ENSEMBLE_SCORE"
    threshold = 0.60
    min_train_rows = 100
    h1_min_time = pd.Timestamp(h1_ds.iloc[min(min_train_rows, len(h1_ds) - 1)]["time"])
    h4_min_time = pd.Timestamp(h4_ds.iloc[min(min_train_rows, len(h4_ds) - 1)]["time"])
    eval_start = max(h1_min_time, h4_min_time, pd.Timestamp(h4_ds["time"].min()))
    bounds = _window_bounds(h1_ds, windows=windows, start_time=eval_start)

    coverage_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    all_details: list[pd.DataFrame] = []
    all_details_stress: list[pd.DataFrame] = []
    invalid_causes: list[dict[str, Any]] = []

    for b in bounds:
        w = int(b["window"])
        start_time = pd.Timestamp(b["start_time"])
        end_time = pd.Timestamp(b["end_time"])

        preds_h1, err_h1 = _fit_predict_window(h1_ds, start_time, end_time, threshold=threshold, seed=seed)
        preds_h4, err_h4 = _fit_predict_window(h4_ds, start_time, end_time, threshold=threshold, seed=seed)

        p1 = preds_dir / f"preds_h1_window_{w}.csv"
        p4 = preds_dir / f"preds_h4_window_{w}.csv"
        preds_h1.to_csv(p1, index=False)
        preds_h4.to_csv(p4, index=False)

        h1_test = h1_ds[(h1_ds["time"] >= start_time) & (h1_ds["time"] <= end_time)].copy()
        h1_test = h1_test[["time", "y", "spread"]].merge(
            preds_h1.rename(
                columns={
                    "signal": "signal_h1",
                    "prob_buy": "prob_h1_buy",
                    "prob_sell": "prob_h1_sell",
                    "has_pred": "has_h1_pred",
                }
            ),
            on="time",
            how="left",
        )
        h1_test["has_h1_pred"] = h1_test["has_h1_pred"].fillna(False)
        h1_test["signal_h1"] = h1_test["signal_h1"].fillna(WAIT).astype(int)

        h4_pred_df = preds_h4.rename(
            columns={
                "signal": "signal_h4_aligned",
                "prob_buy": "prob_h4_buy_aligned",
                "prob_sell": "prob_h4_sell_aligned",
                "has_pred": "has_h4_pred",
            }
        )
        aligned = align_h4_to_h1(h1_test, h4_pred_df, "time", "time")
        decisions, reasons, enriched = apply_policy(aligned, policy=policy, threshold_final=threshold)

        report, details = evaluate_external_signals(
            df=enriched,
            signals=decisions,
            symbol=symbol,
            timeframe="H1",
            experiment_type=f"phase8_{policy}_window_{w}",
            params={"policy": policy, "window": w, "threshold_final": threshold, "seed": seed},
            sanity_extra={
                "oof_no_in_sample_backtest": True,
                "multitf_no_lookahead_alignment": True,
            },
            cost_multiplier=1.0,
        )
        details["window"] = w
        all_details.append(details)

        _, details_stress = evaluate_external_signals(
            df=enriched,
            signals=decisions,
            symbol=symbol,
            timeframe="H1",
            experiment_type=f"phase8_{policy}_window_{w}_stress25",
            params={"policy": policy, "window": w, "cost_plus": 25},
            sanity_extra={
                "oof_no_in_sample_backtest": True,
                "multitf_no_lookahead_alignment": True,
            },
            cost_multiplier=1.25,
        )
        details_stress["window"] = w
        all_details_stress.append(details_stress)

        n_bars_window = int(len(enriched))
        n_preds_h1_available = int(enriched["has_h1_pred"].fillna(False).sum())
        n_preds_h4_available = int(preds_h4["has_pred"].fillna(False).sum())
        n_h4_aligned = int(enriched["has_h4_pred"].fillna(False).sum())
        nonzero_h4 = (
            enriched["has_h4_pred"].fillna(False)
            & (
                enriched["prob_h4_buy_aligned"].fillna(0.0).astype(float).abs() > 0.0
            )
            | (
                enriched["has_h4_pred"].fillna(False)
                & (enriched["prob_h4_sell_aligned"].fillna(0.0).astype(float).abs() > 0.0)
            )
        )
        n_h4_nonzero_prob = int(nonzero_h4.sum())
        n_decisions_ok = int((pd.Series(reasons) == "ok").sum())
        n_trades = int((details["signal"] != WAIT).sum())
        blocked = pd.Series(reasons).value_counts().to_dict()
        coverage_ratio = (float(n_h4_aligned) / float(n_bars_window)) if n_bars_window else 0.0

        window_invalid = n_preds_h4_available == 0
        invalid_cause = None
        if window_invalid:
            invalid_cause = err_h4 or "h4_predictions_missing"
            invalid_causes.append(
                {
                    "window": w,
                    "window_invalid": True,
                    "cause": invalid_cause,
                    "preds_h4_path": str(p4),
                }
            )

        coverage_rows.append(
            {
                "window": w,
                "start": str(start_time),
                "end": str(end_time),
                "n_bars_window": n_bars_window,
                "n_preds_h1_available": n_preds_h1_available,
                "n_preds_h4_available": n_preds_h4_available,
                "n_times_h4_aligned_nonzero_prob": n_h4_nonzero_prob,
                "n_decisions_ok": n_decisions_ok,
                "n_trades": n_trades,
                "coverage_ratio": coverage_ratio,
                "window_invalid": window_invalid,
                "invalid_cause": invalid_cause,
                "blocked_reason_counts": json.dumps(blocked, separators=(",", ":")),
                "preds_h1_path": str(p1),
                "preds_h4_path": str(p4),
            }
        )
        window_rows.append(
            {
                "window": w,
                "start": str(start_time),
                "end": str(end_time),
                "trades": report["trades"],
                "profit_factor": report["profit_factor"],
                "expectancy": report["expectancy"],
                "sharpe": report["sharpe"],
                "max_drawdown": report["max_drawdown"],
                "window_invalid": window_invalid,
                "no_signal_window": report["trades"] == 0,
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = out_dir / f"phase8_coverage_{policy}.csv"
    coverage_df.to_csv(coverage_path, index=False)

    wf_df = pd.DataFrame(window_rows)
    all_details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    all_details_stress_df = pd.concat(all_details_stress, ignore_index=True) if all_details_stress else pd.DataFrame()

    valid_mask = ~wf_df["window_invalid"]
    valid_windows = int(valid_mask.sum())
    invalid_windows = int((~valid_mask).sum())
    no_signal_windows = int((valid_mask & wf_df["no_signal_window"]).sum())
    wf_valid_trade = wf_df[valid_mask & (~wf_df["no_signal_window"])].copy()
    pf_gt_1_ratio_valid_only = float((wf_valid_trade["profit_factor"] > 1.0).mean()) if not wf_valid_trade.empty else 0.0

    coverage_pass_ratio = float((coverage_df["coverage_ratio"] >= 0.90).mean()) if not coverage_df.empty else 0.0
    windows_with_10_trades = int((wf_df["trades"] >= 10).sum())
    trades_10_ratio = float((wf_df["trades"] >= 10).mean()) if not wf_df.empty else 0.0
    trades_total = int((all_details_df["signal"] != WAIT).sum()) if not all_details_df.empty else 0
    trade_windows_count = int((wf_df["trades"] > 0).sum()) if not wf_df.empty else 0
    min_trades_rule_ok = bool(
        (trades_10_ratio >= 0.60) or (trades_total >= 200 and trade_windows_count >= 4)
    )
    dd_windows_ok = bool((wf_df["max_drawdown"] >= -0.05).all()) if not wf_df.empty else False

    agg_report = _aggregate_report(
        all_details_df,
        symbol=symbol,
        policy=policy,
        sanity_extra={"oof_no_in_sample_backtest": True, "multitf_no_lookahead_alignment": True},
    )
    stress_report = _aggregate_report(
        all_details_stress_df,
        symbol=symbol,
        policy=f"{policy}_stress25",
        sanity_extra={"oof_no_in_sample_backtest": True, "multitf_no_lookahead_alignment": True},
    )
    bootstrap = _bootstrap(all_details_df, sims=2000, seed=seed)

    policy_audit = {
        "policy": policy,
        "report": agg_report,
        "walk_forward": {
            "windows": wf_df.to_dict(orient="records"),
            "valid_windows": valid_windows,
            "invalid_windows": invalid_windows,
            "no_signal_windows": no_signal_windows,
            "pf_gt_1_ratio_valid_only": pf_gt_1_ratio_valid_only,
        },
        "coverage": {
            "coverage_csv_path": str(coverage_path),
            "rows": coverage_df.to_dict(orient="records"),
            "coverage_pass_ratio_ge_0_90": coverage_pass_ratio,
        },
        "stress_25": {
            "profit_factor": stress_report["profit_factor"],
            "expectancy": stress_report["expectancy"],
            "max_drawdown": stress_report["max_drawdown"],
        },
        "bootstrap": bootstrap,
        "invalid_window_causes": invalid_causes,
        "criteria_eval": {
            "coverage_ge_0_90_majority": coverage_pass_ratio >= 0.60,
            "min_trades_per_window_rule_ok": min_trades_rule_ok,
            "pf_gt_1_ratio_valid_only_ge_0_60": pf_gt_1_ratio_valid_only >= 0.60,
            "stress25_pf_gt_1": stress_report["profit_factor"] > 1.0,
            "dd_windows_ok": dd_windows_ok,
        },
    }
    audit_path = out_dir / f"phase8_policy_{policy}.json"
    audit_path.write_text(json.dumps(policy_audit, indent=2, default=str), encoding="utf-8")

    grid_df = pd.DataFrame(
        [
            {
                "policy": policy,
                "trades": agg_report["trades"],
                "profit_factor": agg_report["profit_factor"],
                "expectancy": agg_report["expectancy"],
                "sharpe": agg_report["sharpe"],
                "max_drawdown": agg_report["max_drawdown"],
                "pf_gt_1_ratio_valid_only": pf_gt_1_ratio_valid_only,
                "valid_windows": valid_windows,
                "invalid_windows": invalid_windows,
                "no_signal_windows": no_signal_windows,
                "coverage_pass_ratio_ge_0_90": coverage_pass_ratio,
                "windows_with_10_trades": windows_with_10_trades,
                "stress25_pf": stress_report["profit_factor"],
                "dd_windows_ok": dd_windows_ok,
                "sanity_ok": agg_report["sanity_ok"],
                "audit_json_path": str(audit_path),
            }
        ]
    )
    grid_path = out_dir / "phase8_multitf_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    approved = bool(
        (pf_gt_1_ratio_valid_only >= 0.60)
        and (coverage_pass_ratio >= 0.60)
        and min_trades_rule_ok
        and (stress_report["profit_factor"] > 1.0)
        and dd_windows_ok
    )
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "setup": {
            "tf_entry": "H1",
            "tf_gate": "H4",
            "policy": policy,
            "threshold_final": threshold,
            "seed": seed,
            "windows": windows,
            "eval_start_time": str(eval_start),
            "min_train_rows_per_window": min_train_rows,
        },
        "criteria": {
            "coverage_ratio_min": 0.90,
            "coverage_majority_ratio_min": 0.60,
            "pf_gt_1_ratio_valid_only_min": 0.60,
            "min_trades_per_window": 10,
            "min_trades_windows_ratio": 0.60,
            "fallback_total_trades": 200,
            "fallback_trade_windows": 4,
            "stress25_pf_gt_1": True,
            "dd_window_floor": -0.05,
        },
        "result": {
            "valid_windows": valid_windows,
            "invalid_windows": invalid_windows,
            "no_signal_windows": no_signal_windows,
            "pf_gt_1_ratio_valid_only": pf_gt_1_ratio_valid_only,
            "coverage_pass_ratio_ge_0_90": coverage_pass_ratio,
            "min_trades_rule_ok": min_trades_rule_ok,
            "trades_total": trades_total,
            "trade_windows_count": trade_windows_count,
            "stress25_pf": stress_report["profit_factor"],
            "dd_windows_ok": dd_windows_ok,
        },
        "approved": approved,
        "recommendation": "APTO PARA MICRO-LOTE LIVE" if approved else "NECESSITA REVISÃO",
        "outputs": {
            "grid_csv": str(grid_path),
            "summary_json": str(out_dir / "phase8_summary.json"),
            "policy_audit_json": str(audit_path),
            "coverage_csv": str(coverage_path),
            "preds_dir": str(preds_dir),
        },
    }
    summary_path = out_dir / "phase8_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"saved_grid={grid_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_policy_audit={audit_path}")
    print(f"saved_coverage={coverage_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 multi-TF coverage/robustness runner.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase8(symbol=args.symbol, seed=args.seed, windows=args.windows)


if __name__ == "__main__":
    main()
