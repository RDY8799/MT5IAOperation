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
from .config import CONFIG, TimeframeConfig
from .data_feed import run_collection
from .features import process_features
from .labeling_triple_barrier import build_dataset, triple_barrier_labels
from .metrics import max_drawdown, profit_factor
from .multitf import BUY, SELL, WAIT, align_h4_to_h1

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


def _ensure_dataset(symbol: str, tf: str, horizon: int) -> Path:
    path = CONFIG.data_processed_dir / f"{symbol}_{tf}_dataset.parquet"
    if path.exists():
        return path
    _ensure_features(symbol, tf)
    if tf in {"H4", "M30", "M5", "M1"}:
        feat = pd.read_parquet(CONFIG.data_processed_dir / f"{symbol}_{tf}_features.parquet")
        ds = triple_barrier_labels(feat, timeframe=tf, pt_mult=2.0, sl_mult=1.0, horizon=horizon)
        ds.to_parquet(path, index=False)
    else:
        build_dataset(symbol=symbol, timeframe=tf)
    return path


def _window_bounds(df: pd.DataFrame, windows: int, min_train_rows: int = 120) -> list[dict[str, Any]]:
    start_idx = min(min_train_rows, len(df) - 1)
    scoped = df.iloc[start_idx:].copy()
    chunks = np.array_split(np.arange(len(scoped)), windows)
    out: list[dict[str, Any]] = []
    for i, idx in enumerate(chunks, start=1):
        if len(idx) == 0:
            continue
        out.append(
            {
                "window": i,
                "start": pd.Timestamp(scoped.iloc[idx[0]]["time"]),
                "end": pd.Timestamp(scoped.iloc[idx[-1]]["time"]),
            }
        )
    return out


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
    if len(train) < 120:
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
    proba = np.asarray(model.predict_proba(X_test))
    if proba.ndim == 1:
        proba = np.vstack([1.0 - proba, proba]).T
    idx_buy = class_to_idx.get(BUY)
    idx_sell = class_to_idx.get(SELL)
    p_buy = proba[:, idx_buy] if idx_buy is not None else np.full(len(test), np.nan)
    p_sell = proba[:, idx_sell] if idx_sell is not None else np.full(len(test), np.nan)
    has_pred = ~(np.isnan(p_buy) | np.isnan(p_sell))
    signal = np.zeros(len(test), dtype=int)
    for i in np.where(has_pred)[0]:
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


def _direction_policy(entry_signal: int, gate_signal: int) -> tuple[int, str]:
    if entry_signal == WAIT:
        return WAIT, "entry_wait"
    if gate_signal == WAIT:
        return WAIT, "blocked_by_tf_conflict"
    if gate_signal != entry_signal:
        return WAIT, "blocked_by_tf_conflict"
    return entry_signal, "ok"


def _apply_phase10_controls(
    df: pd.DataFrame,
    profile: TimeframeConfig,
    threshold: float,
    min_signal_margin: float,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    out = df.copy()
    entry_signal = np.zeros(len(out), dtype=int)
    for i in range(len(out)):
        p_buy = float(out.iloc[i]["prob_h1_buy"])
        p_sell = float(out.iloc[i]["prob_h1_sell"])
        if p_buy >= threshold and p_buy > p_sell:
            entry_signal[i] = BUY
        elif p_sell >= threshold and p_sell > p_buy:
            entry_signal[i] = SELL
    gate_signal = out["signal_h4_aligned"].fillna(WAIT).astype(int).values

    atr = out["ATR_14"].fillna(0.0).astype(float).values
    atr_valid = atr[np.isfinite(atr)]
    atr_p_min = float(np.percentile(atr_valid, profile.volatility_p_min)) if len(atr_valid) else 0.0
    atr_p_max = float(np.percentile(atr_valid, profile.volatility_p_max)) if len(atr_valid) else 1e9

    decisions = np.zeros(len(out), dtype=int)
    reasons: list[str] = []
    trade_times: list[pd.Timestamp] = []
    last_by_dir: dict[int, pd.Timestamp | None] = {BUY: None, SELL: None}
    last_close_by_dir: dict[int, pd.Timestamp | None] = {BUY: None, SELL: None}

    for i in range(len(out) - 1):
        t = pd.Timestamp(out.iloc[i]["time"])
        d, reason = _direction_policy(entry_signal[i], int(gate_signal[i]))
        if d != WAIT:
            p_buy = float(out.iloc[i]["prob_h1_buy"])
            p_sell = float(out.iloc[i]["prob_h1_sell"])
            if abs(p_buy - p_sell) < min_signal_margin:
                d, reason = WAIT, "blocked_by_low_signal_margin"

        if d != WAIT:
            atr_v = float(atr[i])
            if atr_v < atr_p_min:
                d, reason = WAIT, "blocked_by_volatility_low"
            elif atr_v > atr_p_max:
                d, reason = WAIT, "blocked_by_volatility_high"

        if d != WAIT:
            if profile.trades_window_mode == "fixed_hour":
                h0 = t.floor("h")
                n = sum(1 for x in trade_times if x.floor("h") == h0)
            else:
                n = sum(1 for x in trade_times if (t - x).total_seconds() <= 3600)
            if n >= profile.max_trades_per_hour:
                d, reason = WAIT, "blocked_by_frequency"

        if d != WAIT and profile.min_candles_between_same_direction_trades > 0:
            prev = last_by_dir[d]
            if prev is not None:
                elapsed = (t - prev).total_seconds() / 60.0
                candles = int(elapsed // max(1, 5))
                if candles < profile.min_candles_between_same_direction_trades:
                    d, reason = WAIT, "blocked_by_reentry_rule"

        if d != WAIT and profile.reentry_block_candles > 0:
            prev_close = last_close_by_dir[d]
            if prev_close is not None:
                elapsed = (t - prev_close).total_seconds() / 60.0
                candles = int(elapsed // max(1, 5))
                if candles < profile.reentry_block_candles:
                    d, reason = WAIT, "blocked_by_reentry_rule"

        decisions[i] = d
        reasons.append(reason)
        if d != WAIT:
            trade_times.append(t)
            last_by_dir[d] = t
            hold_idx = min(i + profile.horizon_candles, len(out) - 1)
            last_close_by_dir[d] = pd.Timestamp(out.iloc[hold_idx]["time"])

    reasons.append("tail")
    return decisions, reasons, {"atr_p_min_value": atr_p_min, "atr_p_max_value": atr_p_max}


def _bootstrap(details: pd.DataFrame, sims: int = 2000, seed: int = 42) -> dict[str, Any]:
    trades = details[details["signal"] != WAIT]["ret"].values
    if len(trades) == 0:
        return {"simulations": sims, "n_trades_sampled": 0}
    rng = np.random.default_rng(seed)
    pf_vals, dd_vals, cum_vals = [], [], []
    for _ in range(sims):
        sample = rng.choice(trades, size=len(trades), replace=True)
        gp = sample[sample > 0].sum()
        gl = -sample[sample < 0].sum()
        pf_vals.append(float(gp / gl) if gl > 0 else 0.0)
        cum = np.cumsum(sample)
        dd_vals.append(float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0)
        cum_vals.append(float(sample.sum()))
    return {
        "simulations": sims,
        "n_trades_sampled": int(len(trades)),
        "pf_p05": float(np.percentile(pf_vals, 5)),
        "pf_p50": float(np.percentile(pf_vals, 50)),
        "pf_p95": float(np.percentile(pf_vals, 95)),
        "dd_p05": float(np.percentile(dd_vals, 5)),
        "cum_return_p50": float(np.percentile(cum_vals, 50)),
    }


def run_phase10(
    symbol: str = "EURUSD",
    entry_tf: str = "M5",
    gate_tf: str = "M30",
    windows: int = 8,
    seed: int = 42,
    threshold: float | None = None,
    min_signal_margin: float | None = None,
    output_prefix: str = "phase10",
) -> Path:
    CONFIG.ensure_dirs()
    profile = CONFIG.timeframe_profiles.get(entry_tf)
    if not profile or not profile.enabled:
        raise RuntimeError(f"Timeframe profile disabled/not found for {entry_tf}")
    if profile.tf_gate != gate_tf:
        raise RuntimeError(f"Config mismatch: profile gate for {entry_tf} is {profile.tf_gate}, not {gate_tf}")

    _ensure_features(symbol, entry_tf)
    _ensure_features(symbol, gate_tf)
    entry_ds = pd.read_parquet(_ensure_dataset(symbol, entry_tf, profile.horizon_candles)).sort_values("time").reset_index(drop=True)
    gate_ds = pd.read_parquet(_ensure_dataset(symbol, gate_tf, 8)).sort_values("time").reset_index(drop=True)
    entry_ds["time"] = pd.to_datetime(entry_ds["time"], utc=True)
    gate_ds["time"] = pd.to_datetime(gate_ds["time"], utc=True)

    bounds = _window_bounds(entry_ds, windows=windows, min_train_rows=120)
    threshold = float(profile.signal_threshold if threshold is None else threshold)
    min_signal_margin = float(profile.min_signal_margin if min_signal_margin is None else min_signal_margin)
    preds_dir = CONFIG.reports_dir / "preds_phase10"
    preds_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows: list[dict[str, Any]] = []
    wf_rows: list[dict[str, Any]] = []
    all_details: list[pd.DataFrame] = []
    all_stress: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for b in bounds:
        w = int(b["window"])
        start = pd.Timestamp(b["start"])
        end = pd.Timestamp(b["end"])
        p_entry, err_e = _fit_predict_window(entry_ds, start, end, threshold=threshold, seed=seed)
        p_gate, err_g = _fit_predict_window(gate_ds, start, end, threshold=threshold, seed=seed)
        p1 = preds_dir / f"phase10_preds_{entry_tf}_window_{w}.csv"
        p2 = preds_dir / f"phase10_preds_{gate_tf}_window_{w}.csv"
        p_entry.to_csv(p1, index=False)
        p_gate.to_csv(p2, index=False)

        test = entry_ds[(entry_ds["time"] >= start) & (entry_ds["time"] <= end)].copy()
        test = test[["time", "y", "spread", "ATR_14"]].merge(
            p_entry.rename(
                columns={"signal": "signal_h1", "prob_buy": "prob_h1_buy", "prob_sell": "prob_h1_sell", "has_pred": "has_h1_pred"}
            ),
            on="time",
            how="left",
        )
        test["has_h1_pred"] = test["has_h1_pred"].fillna(False)
        test["signal_h1"] = test["signal_h1"].fillna(WAIT).astype(int)
        gate_pred = p_gate.rename(
            columns={
                "signal": "signal_h4_aligned",
                "prob_buy": "prob_h4_buy_aligned",
                "prob_sell": "prob_h4_sell_aligned",
                "has_pred": "has_h4_pred",
            }
        )
        aligned = align_h4_to_h1(test, gate_pred, "time", "time")
        decisions, reasons, atr_meta = _apply_phase10_controls(
            aligned,
            profile=profile,
            threshold=threshold,
            min_signal_margin=min_signal_margin,
        )

        rep, det = evaluate_external_signals(
            df=aligned,
            signals=decisions,
            symbol=symbol,
            timeframe=entry_tf,
            experiment_type=f"phase10_{entry_tf}_{gate_tf}_window_{w}",
            params={"entry_tf": entry_tf, "gate_tf": gate_tf, "window": w, "threshold": threshold},
            sanity_extra={"multitf_no_lookahead_alignment": True, "oof_window_training_only": True},
        )
        det["window"] = w
        all_details.append(det)

        rep_s, det_s = evaluate_external_signals(
            df=aligned,
            signals=decisions,
            symbol=symbol,
            timeframe=entry_tf,
            experiment_type=f"phase10_{entry_tf}_{gate_tf}_window_{w}_stress25",
            params={"entry_tf": entry_tf, "gate_tf": gate_tf, "window": w, "cost_plus": 25},
            sanity_extra={"multitf_no_lookahead_alignment": True, "oof_window_training_only": True},
            cost_multiplier=1.25,
        )
        det_s["window"] = w
        all_stress.append(det_s)

        blocked_counts = pd.Series(reasons).value_counts().to_dict()
        coverage_ratio = float(aligned["has_h4_pred"].fillna(False).mean()) if len(aligned) else 0.0
        window_invalid = bool(int(p_gate["has_pred"].fillna(False).sum()) == 0)
        coverage_rows.append(
            {
                "window": w,
                "start": str(start),
                "end": str(end),
                "n_bars_window": int(len(aligned)),
                "n_preds_entry_available": int(aligned["has_h1_pred"].fillna(False).sum()),
                "n_preds_gate_available": int(p_gate["has_pred"].fillna(False).sum()),
                "coverage_ratio": coverage_ratio,
                "n_trades": int((det["signal"] != WAIT).sum()),
                "window_invalid": window_invalid,
                "invalid_cause": (err_g or err_e) if window_invalid else "",
                "atr_p_min_value": atr_meta["atr_p_min_value"],
                "atr_p_max_value": atr_meta["atr_p_max_value"],
                "threshold_used": threshold,
                "min_signal_margin": min_signal_margin,
                "blocked_reason_counts": json.dumps(blocked_counts, separators=(",", ":")),
                "preds_entry_path": str(p1),
                "preds_gate_path": str(p2),
            }
        )
        wf_rows.append(
            {
                "window": w,
                "start": str(start),
                "end": str(end),
                "trades": rep["trades"],
                "profit_factor": rep["profit_factor"],
                "max_drawdown": rep["max_drawdown"],
                "window_invalid": window_invalid,
                "no_signal_window": rep["trades"] == 0,
            }
        )
        audit_rows.append(
            {
                "window": w,
                "params": {"entry_tf": entry_tf, "gate_tf": gate_tf, "threshold": threshold},
                "report": rep,
                "stress25": rep_s,
                "blocked_reason_counts": blocked_counts,
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    wf_df = pd.DataFrame(wf_rows)
    details_all = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    stress_all = pd.concat(all_stress, ignore_index=True) if all_stress else pd.DataFrame()

    valid_mask = ~wf_df["window_invalid"]
    valid_windows = int(valid_mask.sum())
    no_signal_windows = int((valid_mask & wf_df["no_signal_window"]).sum())
    wf_valid_traded = wf_df[valid_mask & (wf_df["trades"] > 0)]
    pf_gt_1_ratio = float((wf_valid_traded["profit_factor"] > 1.0).mean()) if not wf_valid_traded.empty else 0.0
    dd_windows_ok = bool((wf_df.loc[valid_mask, "max_drawdown"] >= -0.05).all()) if valid_windows else False
    coverage_ratio = float((coverage_df["coverage_ratio"] >= 0.90).mean()) if not coverage_df.empty else 0.0
    trade_windows_count = int((wf_df["trades"] > 0).sum()) if not wf_df.empty else 0
    trades_total = int((details_all["signal"] != WAIT).sum()) if not details_all.empty else 0

    if not stress_all.empty:
        stress_returns = stress_all["ret"].astype(float)
        gp = float(stress_returns[stress_returns > 0].sum())
        gl = float(-stress_returns[stress_returns < 0].sum())
        stress_pf = gp / gl if gl > 0 else 0.0
    else:
        stress_pf = 0.0

    grid_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "tf_entry": entry_tf,
        "tf_gate": gate_tf,
        "trades": trades_total,
        "profit_factor": profit_factor(details_all["ret"]) if not details_all.empty else 0.0,
        "max_drawdown": max_drawdown(details_all["ret"]) if not details_all.empty else 0.0,
        "pf_gt_1_ratio_valid_only": pf_gt_1_ratio,
        "dd_windows_ok": dd_windows_ok,
        "stress25_pf": stress_pf,
        "trade_windows_count": trade_windows_count,
        "coverage_ratio": coverage_ratio,
        "trades_total": trades_total,
        "no_signal_windows": no_signal_windows,
    }
    grid_df = pd.DataFrame([grid_row])

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "entry_tf": entry_tf,
        "gate_tf": gate_tf,
        "profile": profile.model_dump(),
        "signal_filters": {"threshold_used": threshold, "min_signal_margin": min_signal_margin},
        "criteria": {
            "pf_gt_1_ratio_valid_only_min": 0.60,
            "dd_windows_ok": True,
            "stress25_pf_gt_1": True,
            "trade_windows_count_min": 6,
            "coverage_ratio_min": 0.90,
            "trades_total_min": 120,
        },
        "result": {
            "pf_gt_1_ratio_valid_only": pf_gt_1_ratio,
            "dd_windows_ok": dd_windows_ok,
            "stress25_pf": stress_pf,
            "trade_windows_count": trade_windows_count,
            "coverage_ratio": coverage_ratio,
            "trades_total": trades_total,
            "valid_windows": valid_windows,
            "no_signal_windows": no_signal_windows,
        },
        "approved": bool(
            (pf_gt_1_ratio >= 0.60)
            and dd_windows_ok
            and (stress_pf > 1.0)
            and (trade_windows_count >= 6)
            and (coverage_ratio >= 0.90)
            and (trades_total >= 120)
        ),
        "bootstrap": _bootstrap(details_all, sims=2000, seed=seed),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    coverage_path = CONFIG.reports_dir / f"{output_prefix}_{entry_tf}_coverage.csv"
    grid_path = CONFIG.reports_dir / f"{output_prefix}_{entry_tf}_grid.csv"
    summary_path = CONFIG.reports_dir / f"{output_prefix}_{entry_tf}_summary.json"
    bootstrap_path = CONFIG.reports_dir / f"{output_prefix}_{entry_tf}_bootstrap.json"
    coverage_df.to_csv(coverage_path, index=False)
    grid_df.to_csv(grid_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    bootstrap_path.write_text(json.dumps(summary["bootstrap"], indent=2, default=str), encoding="utf-8")
    for item in audit_rows:
        p = CONFIG.reports_dir / f"{output_prefix}_{entry_tf}_audit_w{item['window']}_{stamp}.json"
        p.write_text(json.dumps(item, indent=2, default=str), encoding="utf-8")
    print(f"saved_coverage={coverage_path}")
    print(f"saved_grid={grid_path}")
    print(f"saved_bootstrap={bootstrap_path}")
    print(f"saved_summary={summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10 runner (M5 first).")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--tf_entry", default="M5")
    parser.add_argument("--tf_gate", default="M30")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase10(
        symbol=args.symbol,
        entry_tf=args.tf_entry,
        gate_tf=args.tf_gate,
        windows=args.windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
