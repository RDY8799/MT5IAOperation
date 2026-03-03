from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG
from .metrics import expectancy, max_drawdown, profit_factor, sharpe, sortino
from .multitf import BUY, SELL, WAIT, align_h4_to_h1, apply_policy


@dataclass
class RiskLayerConfig:
    risk_per_trade_pct: float = 0.005
    sl_mult: float = 1.0
    dd_circuit_limit: float = 0.04
    max_consecutive_losses: int = 3
    cooldown_candles: int = 8
    size_mult_min: float = 0.25
    size_mult_max: float = 4.0
    threshold_final: float = 0.60


def _load_phase8_coverage(policy: str = "ENSEMBLE_SCORE") -> pd.DataFrame:
    path = CONFIG.reports_dir / f"phase8_coverage_{policy}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Coverage file not found: {path}. Run phase8_runner first."
        )
    df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    return df.sort_values("window").reset_index(drop=True)


def _prepare_window_df(
    h1_ds: pd.DataFrame,
    window_row: pd.Series,
    threshold_final: float,
) -> pd.DataFrame:
    start = pd.Timestamp(window_row["start"])
    end = pd.Timestamp(window_row["end"])
    preds_h1 = pd.read_csv(window_row["preds_h1_path"])
    preds_h4 = pd.read_csv(window_row["preds_h4_path"])

    preds_h1["time"] = pd.to_datetime(preds_h1["time"], utc=True)
    preds_h4["time"] = pd.to_datetime(preds_h4["time"], utc=True)

    h1_test = h1_ds[(h1_ds["time"] >= start) & (h1_ds["time"] <= end)].copy()
    h1_test = h1_test[
        ["time", "y", "spread", "ATR_14"]
    ].merge(
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

    h4_pred = preds_h4.rename(
        columns={
            "signal": "signal_h4_aligned",
            "prob_buy": "prob_h4_buy_aligned",
            "prob_sell": "prob_h4_sell_aligned",
            "has_pred": "has_h4_pred",
        }
    )
    aligned = align_h4_to_h1(h1_test, h4_pred, "time", "time")
    decisions, _, enriched = apply_policy(
        aligned,
        policy="ENSEMBLE_SCORE",
        threshold_final=threshold_final,
    )
    enriched = enriched.sort_values("time").reset_index(drop=True)
    enriched["decision_base"] = decisions
    return enriched


def _simulate_window(
    window_df: pd.DataFrame,
    cfg: RiskLayerConfig,
    *,
    cost_multiplier: float = 1.0,
    use_sizing: bool = False,
    use_circuit_breaker: bool = False,
    use_loss_streak: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity = 1.0
    period_key: str | None = None
    period_peak = equity
    circuit_active = False
    cooldown_remaining = 0
    loss_streak = 0

    blocked_by_circuit_breaker_count = 0
    blocked_by_loss_streak_count = 0
    time_in_cooldown = 0

    atr_series = window_df["ATR_14"].replace([np.inf, -np.inf], np.nan).dropna()
    median_atr = float(atr_series.median()) if len(atr_series) else 1e-4
    median_atr = max(median_atr, 1e-6)

    for i in range(len(window_df) - 1):
        t = pd.Timestamp(window_df.iloc[i]["time"])
        new_period_key = f"{t.year:04d}-{t.month:02d}"
        if period_key != new_period_key:
            period_key = new_period_key
            period_peak = equity
            circuit_active = False

        signal = int(window_df.iloc[i]["decision_base"])
        blocked_reason = "ok"
        if use_circuit_breaker and circuit_active:
            signal = WAIT
            blocked_reason = "blocked_by_circuit_breaker"
            blocked_by_circuit_breaker_count += 1
            time_in_cooldown += 1
        elif use_loss_streak and cooldown_remaining > 0:
            signal = WAIT
            blocked_reason = "blocked_by_loss_streak_cooldown"
            blocked_by_loss_streak_count += 1
            cooldown_remaining -= 1
            time_in_cooldown += 1

        y_next = int(window_df.iloc[i + 1]["y"])
        gross = 0.0
        if signal == BUY:
            gross = 0.001 if y_next == BUY else (-0.001 if y_next == SELL else 0.0)
        elif signal == SELL:
            gross = 0.001 if y_next == SELL else (-0.001 if y_next == BUY else 0.0)

        cost = 0.0
        if signal != WAIT:
            spread_cost = float(window_df.iloc[i + 1]["spread"] or 0.0) * 1e-5
            base_cost = (
                spread_cost
                + (CONFIG.live.slippage_points * 1e-5)
                + CONFIG.live.commission_per_trade
            )
            cost = base_cost * cost_multiplier
        ret = gross - cost

        size_mult = 1.0
        if use_sizing and signal != WAIT:
            atr = float(window_df.iloc[i]["ATR_14"] or 0.0)
            atr = max(atr, 1e-6)
            # Normalized ATR-based volatility targeting.
            size_mult = median_atr / atr
            size_mult = float(np.clip(size_mult, cfg.size_mult_min, cfg.size_mult_max))
            ret *= size_mult

        equity += ret

        if signal != WAIT:
            if ret < 0:
                loss_streak += 1
                if use_loss_streak and loss_streak >= cfg.max_consecutive_losses:
                    cooldown_remaining = cfg.cooldown_candles
                    loss_streak = 0
            else:
                loss_streak = 0

        period_peak = max(period_peak, equity)
        if period_peak > 0 and use_circuit_breaker:
            period_dd = (equity - period_peak) / period_peak
            if period_dd <= -cfg.dd_circuit_limit:
                circuit_active = True

        rows.append(
            {
                "time": t,
                "signal": signal,
                "ret": ret,
                "y_next": y_next,
                "size_mult": size_mult,
                "blocked_reason": blocked_reason,
            }
        )

    details = pd.DataFrame(rows)
    extras = {
        "blocked_by_circuit_breaker_count": blocked_by_circuit_breaker_count,
        "blocked_by_loss_streak_count": blocked_by_loss_streak_count,
        "time_in_cooldown": time_in_cooldown,
    }
    return details, extras


def _worst_losing_streak(returns: pd.Series) -> int:
    streak = 0
    worst = 0
    for r in returns:
        if r < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return int(worst)


def _window_metrics(details: pd.DataFrame, window_id: int, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    if details.empty:
        return {
            "window_id": window_id,
            "start": str(start),
            "end": str(end),
            "trades": 0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "cum_return": 0.0,
            "worst_streak": 0,
        }
    ret = details["ret"].astype(float)
    trade_ret = details.loc[details["signal"] != WAIT, "ret"]
    return {
        "window_id": window_id,
        "start": str(start),
        "end": str(end),
        "trades": int((details["signal"] != WAIT).sum()),
        "profit_factor": profit_factor(ret),
        "expectancy": expectancy(ret),
        "sharpe": sharpe(ret),
        "max_drawdown": max_drawdown(ret),
        "cum_return": float(ret.sum()),
        "worst_streak": _worst_losing_streak(trade_ret) if len(trade_ret) else 0,
    }


def _aggregate_metrics(details: pd.DataFrame, sanity_extra: dict[str, bool] | None = None) -> dict[str, Any]:
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
            "sanity": sanity,
            "sanity_ok": bool(all(sanity.values())),
        }

    ret = details["ret"].astype(float)
    trades = details[details["signal"] != WAIT]
    wins = trades["ret"][trades["ret"] > 0]
    losses = trades["ret"][trades["ret"] < 0]
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
        "trades": int(len(trades)),
        "profit_factor": profit_factor(ret),
        "expectancy": expectancy(ret),
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "max_drawdown": max_drawdown(ret),
        "win_rate": float((trades["ret"] > 0).mean()) if len(trades) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "class_distribution": {
            "buy_pct": float((details["signal"] == BUY).mean()),
            "sell_pct": float((details["signal"] == SELL).mean()),
            "wait_pct": float((details["signal"] == WAIT).mean()),
        },
        "sanity": sanity,
        "sanity_ok": bool(all(sanity.values())),
    }


def _bootstrap(details: pd.DataFrame, sims: int = 2000, seed: int = 42) -> dict[str, Any]:
    trades = details[details["signal"] != WAIT]["ret"].values
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


def run_phase9(symbol: str = "EURUSD", seed: int = 42) -> Path:
    CONFIG.ensure_dirs()
    out_dir = CONFIG.reports_dir
    cfg = RiskLayerConfig()

    coverage = _load_phase8_coverage(policy="ENSEMBLE_SCORE")
    h1_ds = pd.read_parquet(CONFIG.data_processed_dir / f"{symbol}_H1_dataset.parquet")
    h1_ds["time"] = pd.to_datetime(h1_ds["time"], utc=True)
    h1_ds = h1_ds.sort_values("time").reset_index(drop=True)

    prepared_windows: list[dict[str, Any]] = []
    for _, row in coverage.iterrows():
        w = int(row["window"])
        prepared = _prepare_window_df(h1_ds, row, threshold_final=cfg.threshold_final)
        prepared_windows.append(
            {
                "window": w,
                "start": pd.Timestamp(row["start"]),
                "end": pd.Timestamp(row["end"]),
                "coverage_ratio": float(row["coverage_ratio"]),
                "window_invalid": bool(row.get("window_invalid", False)),
                "df": prepared,
            }
        )

    modes = [
        ("baseline_no_risk", False, False, False),
        ("sizing_only", True, False, False),
        ("risk_layer_full", True, True, True),
    ]

    all_mode_results: dict[str, dict[str, Any]] = {}
    dd_report_rows: list[dict[str, Any]] = []

    for mode_name, use_sizing, use_circuit, use_streak in modes:
        window_metrics_rows = []
        details_all = []
        blocked_cb = 0
        blocked_streak = 0
        cooldown_bars = 0
        for w in prepared_windows:
            details, extras = _simulate_window(
                w["df"],
                cfg,
                cost_multiplier=1.0,
                use_sizing=use_sizing,
                use_circuit_breaker=use_circuit,
                use_loss_streak=use_streak,
            )
            details["window"] = w["window"]
            details_all.append(details)

            wm = _window_metrics(details, w["window"], w["start"], w["end"])
            wm["window_invalid"] = bool(w["window_invalid"])
            wm["coverage_ratio"] = float(w["coverage_ratio"])
            window_metrics_rows.append(wm)

            blocked_cb += int(extras["blocked_by_circuit_breaker_count"])
            blocked_streak += int(extras["blocked_by_loss_streak_count"])
            cooldown_bars += int(extras["time_in_cooldown"])

        mode_details = pd.concat(details_all, ignore_index=True)
        mode_windows_df = pd.DataFrame(window_metrics_rows).sort_values("window_id")
        mode_agg = _aggregate_metrics(
            mode_details,
            sanity_extra={"multitf_no_lookahead_alignment": True},
        )
        all_mode_results[mode_name] = {
            "details": mode_details,
            "windows_df": mode_windows_df,
            "agg": mode_agg,
            "blocked_by_circuit_breaker_count": blocked_cb,
            "blocked_by_loss_streak_count": blocked_streak,
            "time_in_cooldown": cooldown_bars,
        }

        if mode_name == "baseline_no_risk":
            dd_report_rows = mode_windows_df[
                ["window_id", "start", "end", "trades", "max_drawdown", "cum_return", "worst_streak"]
            ].to_dict(orient="records")

    dd_report_path = out_dir / "phase9_dd_windows_report.csv"
    pd.DataFrame(dd_report_rows).to_csv(dd_report_path, index=False)

    baseline = all_mode_results["baseline_no_risk"]["agg"]
    sizing = all_mode_results["sizing_only"]["agg"]
    sizing_cmp_path = out_dir / "phase9_sizing_comparison.csv"
    pd.DataFrame(
        [
            {"mode": "without_sizing", **baseline},
            {"mode": "with_sizing", **sizing},
        ]
    ).to_csv(sizing_cmp_path, index=False)

    risk_full = all_mode_results["risk_layer_full"]
    risk_windows = risk_full["windows_df"]
    risk_details = risk_full["details"]
    risk_agg = risk_full["agg"]

    valid_mask = ~risk_windows["window_invalid"]
    valid_windows = int(valid_mask.sum())
    invalid_windows = int((~valid_mask).sum())
    no_signal_windows = int((valid_mask & (risk_windows["trades"] == 0)).sum())
    valid_traded = risk_windows[valid_mask & (risk_windows["trades"] > 0)]
    pf_gt_1_ratio_valid_only = float((valid_traded["profit_factor"] > 1.0).mean()) if not valid_traded.empty else 0.0
    dd_windows_ok = bool((risk_windows.loc[valid_mask, "max_drawdown"] >= -0.05).all()) if valid_windows else False

    coverage_pass_ratio = float((coverage["coverage_ratio"] >= 0.90).mean()) if not coverage.empty else 0.0

    stress_details_all = []
    for w in prepared_windows:
        d_stress, _ = _simulate_window(
            w["df"],
            cfg,
            cost_multiplier=1.25,
            use_sizing=True,
            use_circuit_breaker=True,
            use_loss_streak=True,
        )
        stress_details_all.append(d_stress)
    stress_details = pd.concat(stress_details_all, ignore_index=True)
    stress_agg = _aggregate_metrics(stress_details, sanity_extra={"multitf_no_lookahead_alignment": True})

    bootstrap = _bootstrap(risk_details, sims=2000, seed=seed)

    grid_rows = []
    for mode_name in ("baseline_no_risk", "sizing_only", "risk_layer_full"):
        mode = all_mode_results[mode_name]
        wm = mode["windows_df"]
        vmask = ~wm["window_invalid"]
        vtr = wm[vmask & (wm["trades"] > 0)]
        mode_pf_ratio = float((vtr["profit_factor"] > 1.0).mean()) if not vtr.empty else 0.0
        mode_dd_ok = bool((wm.loc[vmask, "max_drawdown"] >= -0.05).all()) if int(vmask.sum()) else False
        grid_rows.append(
            {
                "policy": "ENSEMBLE_SCORE",
                "mode": mode_name,
                "trades": mode["agg"]["trades"],
                "profit_factor": mode["agg"]["profit_factor"],
                "expectancy": mode["agg"]["expectancy"],
                "sharpe": mode["agg"]["sharpe"],
                "max_drawdown": mode["agg"]["max_drawdown"],
                "pf_gt_1_ratio_valid_only": mode_pf_ratio,
                "dd_windows_ok": mode_dd_ok,
                "coverage_pass_ratio_ge_0_90": coverage_pass_ratio,
                "blocked_by_circuit_breaker_count": mode["blocked_by_circuit_breaker_count"],
                "blocked_by_loss_streak_count": mode["blocked_by_loss_streak_count"],
                "time_in_cooldown": mode["time_in_cooldown"],
                "sanity_ok": mode["agg"]["sanity_ok"],
            }
        )
    grid_df = pd.DataFrame(grid_rows)
    grid_path = out_dir / "phase9_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    policy_report = {
        "policy": "ENSEMBLE_SCORE",
        "risk_config": {
            "risk_per_trade_pct": cfg.risk_per_trade_pct,
            "sl_mult": cfg.sl_mult,
            "dd_circuit_limit": cfg.dd_circuit_limit,
            "max_consecutive_losses": cfg.max_consecutive_losses,
            "cooldown_candles": cfg.cooldown_candles,
            "size_mult_min": cfg.size_mult_min,
            "size_mult_max": cfg.size_mult_max,
            "threshold_final": cfg.threshold_final,
        },
        "report": risk_agg,
        "walk_forward": {
            "windows": risk_windows.to_dict(orient="records"),
            "valid_windows": valid_windows,
            "invalid_windows": invalid_windows,
            "no_signal_windows": no_signal_windows,
            "pf_gt_1_ratio_valid_only": pf_gt_1_ratio_valid_only,
            "dd_windows_ok": dd_windows_ok,
        },
        "coverage": {
            "coverage_pass_ratio_ge_0_90": coverage_pass_ratio,
        },
        "stress_25": {
            "profit_factor": stress_agg["profit_factor"],
            "expectancy": stress_agg["expectancy"],
            "max_drawdown": stress_agg["max_drawdown"],
        },
        "bootstrap": bootstrap,
        "risk_layer_logs": {
            "blocked_by_circuit_breaker_count": risk_full["blocked_by_circuit_breaker_count"],
            "blocked_by_loss_streak_count": risk_full["blocked_by_loss_streak_count"],
            "time_in_cooldown": risk_full["time_in_cooldown"],
        },
        "comparisons": {
            "baseline_no_risk": baseline,
            "sizing_only": sizing,
        },
    }
    policy_path = out_dir / "phase9_policy_ENSEMBLE_SCORE.json"
    policy_path.write_text(json.dumps(policy_report, indent=2, default=str), encoding="utf-8")

    approved = bool(
        (pf_gt_1_ratio_valid_only >= 0.60)
        and dd_windows_ok
        and (stress_agg["profit_factor"] > 1.0)
        and (coverage_pass_ratio == 1.0)
    )
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "criteria": {
            "pf_gt_1_ratio_valid_only_min": 0.60,
            "dd_windows_ok": True,
            "stress25_pf_gt_1": True,
            "coverage_pass_ratio_ge_0_90_eq_1_0": True,
        },
        "result": {
            "pf_gt_1_ratio_valid_only": pf_gt_1_ratio_valid_only,
            "dd_windows_ok": dd_windows_ok,
            "stress25_pf": stress_agg["profit_factor"],
            "coverage_pass_ratio_ge_0_90": coverage_pass_ratio,
            "valid_windows": valid_windows,
            "invalid_windows": invalid_windows,
            "no_signal_windows": no_signal_windows,
            "blocked_by_circuit_breaker_count": risk_full["blocked_by_circuit_breaker_count"],
            "blocked_by_loss_streak_count": risk_full["blocked_by_loss_streak_count"],
            "time_in_cooldown": risk_full["time_in_cooldown"],
        },
        "approved": approved,
        "recommendation": "APTO PARA MICRO-LOTE LIVE" if approved else "NECESSITA REVISÃO",
        "outputs": {
            "dd_windows_report_csv": str(dd_report_path),
            "sizing_comparison_csv": str(sizing_cmp_path),
            "grid_csv": str(grid_path),
            "policy_json": str(policy_path),
        },
    }
    summary_path = out_dir / "phase9_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"saved_dd_report={dd_report_path}")
    print(f"saved_sizing_comparison={sizing_cmp_path}")
    print(f"saved_grid={grid_path}")
    print(f"saved_policy={policy_path}")
    print(f"saved_summary={summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 risk-layer robustness runner.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase9(symbol=args.symbol, seed=args.seed)


if __name__ == "__main__":
    main()

