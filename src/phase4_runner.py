from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest_engine import (
    evaluate_external_signals,
    generate_model_oof_probas,
    model_signals_from_probas,
    walk_forward_summary,
)
from .baselines import BUY, SELL, flip_signal, mean_reversion_signal, momentum_signal, random_signal
from .config import CONFIG
from .data_feed import run_collection
from .features import process_features
from .labeling_triple_barrier import triple_barrier_labels


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _ensure_features(symbol: str, timeframe: str, months: int = 24) -> Path:
    feat = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
    if feat.exists():
        return feat
    raw = CONFIG.data_raw_dir / f"{symbol}_{timeframe}.parquet"
    if not raw.exists():
        run_collection(symbol=symbol, timeframes=[timeframe], months=months, credentials=None)
    return process_features(symbol=symbol, timeframe=timeframe)


def _save_experiment_json(payload: dict[str, Any], out_dir: Path, name: str, symbol: str, tf: str, ts: str) -> Path:
    path = out_dir / f"phase4_{name}_{symbol}_{tf}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _record_row(
    report: dict[str, Any],
    symbol: str,
    timeframe: str,
    experiment_type: str,
    params: dict[str, Any],
    report_json_path: Path,
) -> dict[str, Any]:
    cond = report.get("conditional_returns", {})
    buy = cond.get("BUY", {})
    sell = cond.get("SELL", {})
    wait = cond.get("WAIT", {})
    cd = report.get("class_distribution", {})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "experiment_type": experiment_type,
        "params_json": json.dumps(params, separators=(",", ":"), ensure_ascii=True),
        "trades": report.get("trades", 0),
        "profit_factor": report.get("profit_factor", 0.0),
        "expectancy": report.get("expectancy", 0.0),
        "sharpe": report.get("sharpe", 0.0),
        "sortino": report.get("sortino", 0.0),
        "max_drawdown": report.get("max_drawdown", 0.0),
        "win_rate": report.get("win_rate", 0.0),
        "avg_win": report.get("avg_win", 0.0),
        "avg_loss": report.get("avg_loss", 0.0),
        "class_buy_pct": cd.get("buy_pct", 0.0),
        "class_sell_pct": cd.get("sell_pct", 0.0),
        "class_wait_pct": cd.get("wait_pct", 0.0),
        "E_r_buy": buy.get("E_r", 0.0),
        "E_r_sell": sell.get("E_r", 0.0),
        "E_r_wait": wait.get("E_r", 0.0),
        "sanity_ok": report.get("sanity_ok", False),
        "report_json_path": str(report_json_path),
    }


def _evaluate_and_save(
    df: pd.DataFrame,
    signals,
    symbol: str,
    tf: str,
    experiment_type: str,
    params: dict[str, Any],
    out_dir: Path,
    ts: str,
) -> tuple[dict[str, Any], Path]:
    report, details = evaluate_external_signals(
        df=df,
        signals=signals,
        symbol=symbol,
        timeframe=tf,
        experiment_type=experiment_type,
        params=params,
    )
    report["walk_forward"] = walk_forward_summary(details, windows=4)
    payload = {
        "meta": {"symbol": symbol, "timeframe": tf, "experiment_type": experiment_type, "params": params},
        "report": report,
    }
    path = _save_experiment_json(
        payload=payload,
        out_dir=out_dir,
        name=experiment_type,
        symbol=symbol,
        tf=tf,
        ts=ts,
    )
    return report, path


def run_phase4(
    symbol: str,
    tfs: list[str],
    out_dir: Path,
    model_threshold: float,
    model_pt: float,
    model_sl: float,
    model_horizon: int,
    model_regime_q: float | None,
    seed: int,
) -> tuple[Path, Path]:
    CONFIG.ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"by_timeframe": {}}

    horizons = [4, 6, 8]
    momentum_lookbacks = [3, 6, 12, 24]
    meanrev_windows = [20, 50]
    meanrev_ks = [1.0, 1.5, 2.0]
    random_seeds = [1, 2, 3]

    for tf in tfs:
        feat_path = _ensure_features(symbol=symbol, timeframe=tf, months=24)
        feat_df = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True)
        datasets = {
            h: triple_barrier_labels(
                feat_df, timeframe=tf, pt_mult=model_pt, sl_mult=model_sl, horizon=h
            )
            for h in horizons
        }

        # Model best setup + inverted (same dataset and costs)
        model_df = datasets[model_horizon]
        probas, class_to_idx = generate_model_oof_probas(model_df, n_splits=5, seed=seed)
        model_signals = model_signals_from_probas(
            df=model_df,
            probas=probas,
            class_to_idx=class_to_idx,
            threshold=model_threshold,
            regime_vol_quantile=model_regime_q,
        )

        model_params = {
            "threshold": model_threshold,
            "pt_mult": model_pt,
            "sl_mult": model_sl,
            "horizon": model_horizon,
            "regime_q": model_regime_q,
            "seed": seed,
        }
        report, path = _evaluate_and_save(
            df=model_df,
            signals=model_signals,
            symbol=symbol,
            tf=tf,
            experiment_type="model",
            params=model_params,
            out_dir=out_dir,
            ts=ts,
        )
        rows.append(_record_row(report, symbol, tf, "model", model_params, path))

        inv_params = {**model_params, "flip_signal": True}
        inv_report, inv_path = _evaluate_and_save(
            df=model_df,
            signals=flip_signal(model_signals),
            symbol=symbol,
            tf=tf,
            experiment_type="invert_model",
            params=inv_params,
            out_dir=out_dir,
            ts=ts,
        )
        rows.append(_record_row(inv_report, symbol, tf, "invert_model", inv_params, inv_path))
        trade_rate = float((model_signals[:-1] != 0).mean())

        # Momentum baselines
        for h in horizons:
            df_h = datasets[h]
            for n in momentum_lookbacks:
                params = {"lookback_n": n, "horizon": h, "pt_mult": model_pt, "sl_mult": model_sl}
                signal = momentum_signal(df_h["close"], lookback_n=n, threshold=0.0)
                rep, pth = _evaluate_and_save(
                    df=df_h,
                    signals=signal,
                    symbol=symbol,
                    tf=tf,
                    experiment_type="momentum",
                    params=params,
                    out_dir=out_dir,
                    ts=ts,
                )
                rows.append(_record_row(rep, symbol, tf, "momentum", params, pth))

        # Mean reversion baselines
        for h in horizons:
            df_h = datasets[h]
            for window in meanrev_windows:
                for k in meanrev_ks:
                    params = {"window": window, "k": k, "horizon": h, "pt_mult": model_pt, "sl_mult": model_sl}
                    signal = mean_reversion_signal(df_h["close"], window=window, k=k)
                    rep, pth = _evaluate_and_save(
                        df=df_h,
                        signals=signal,
                        symbol=symbol,
                        tf=tf,
                        experiment_type="meanrev",
                        params=params,
                        out_dir=out_dir,
                        ts=ts,
                    )
                    rows.append(_record_row(rep, symbol, tf, "meanrev", params, pth))

        # Random control baseline
        p_buy = trade_rate / 2.0
        p_sell = trade_rate / 2.0
        for h in horizons:
            df_h = datasets[h]
            for s in random_seeds:
                params = {"horizon": h, "seed": s, "p_buy": p_buy, "p_sell": p_sell}
                signal = random_signal(length=len(df_h), p_buy=p_buy, p_sell=p_sell, seed=s)
                rep, pth = _evaluate_and_save(
                    df=df_h,
                    signals=signal,
                    symbol=symbol,
                    tf=tf,
                    experiment_type="random",
                    params=params,
                    out_dir=out_dir,
                    ts=ts,
                )
                rows.append(_record_row(rep, symbol, tf, "random", params, pth))

        tf_df = pd.DataFrame([r for r in rows if r["timeframe"] == tf]).sort_values(
            ["profit_factor", "expectancy", "sharpe", "max_drawdown", "trades"],
            ascending=[False, False, False, False, False],
        )
        summary["by_timeframe"][tf] = {
            "best_experiment": tf_df.iloc[0].to_dict() if not tf_df.empty else {},
            "invert_model": {
                "normal_pf": float(report["profit_factor"]),
                "flip_pf": float(inv_report["profit_factor"]),
                "normal_expectancy": float(report["expectancy"]),
                "flip_expectancy": float(inv_report["expectancy"]),
            },
        }

    all_df = pd.DataFrame(rows).sort_values(
        ["profit_factor", "expectancy", "sharpe", "max_drawdown", "trades"],
        ascending=[False, False, False, False, False],
    )
    csv_path = out_dir / f"phase4_{symbol}_{ts}.csv"
    all_df.to_csv(csv_path, index=False)

    top10 = all_df.head(10).to_dict(orient="records")
    top10_path = out_dir / f"phase4_top10_{ts}.json"
    top10_path.write_text(json.dumps(top10, indent=2, default=str), encoding="utf-8")

    summary_path = out_dir / f"phase4_summary_{symbol}_{ts}.json"
    summary["top10_path"] = str(top10_path)
    summary["csv_path"] = str(csv_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"saved_csv={csv_path}")
    print(f"saved_top10={top10_path}")
    print(f"saved_summary={summary_path}")
    return csv_path, top10_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 strategy diagnostics runner.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tfs", default="H1")
    parser.add_argument("--out", default=str(CONFIG.reports_dir))
    parser.add_argument("--model-threshold", type=float, default=0.60)
    parser.add_argument("--model-pt", type=float, default=2.0)
    parser.add_argument("--model-sl", type=float, default=1.0)
    parser.add_argument("--model-horizon", type=int, default=8)
    parser.add_argument("--model-regime-q", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tfs = [tf.strip().upper() for tf in args.tfs.split(",") if tf.strip()]
    if "H1" not in tfs:
        tfs = ["H1"] + tfs
    run_phase4(
        symbol=args.symbol,
        tfs=tfs,
        out_dir=Path(args.out),
        model_threshold=args.model_threshold,
        model_pt=args.model_pt,
        model_sl=args.model_sl,
        model_horizon=args.model_horizon,
        model_regime_q=None if args.model_regime_q < 0 else args.model_regime_q,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

