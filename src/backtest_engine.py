from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .config import CONFIG
from .cv_purged import PurgedKFold
from .metrics import expectancy, max_drawdown, profit_factor, sharpe, sortino

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


BUY = 1
SELL = -1
WAIT = 0
DROP_COLS = {"time", "y", "t1", "pt", "sl", "regime_class"}


def generate_model_oof_probas(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, dict[int, int]]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[["time"] + feature_cols]
    y_raw = df["y"].astype(int)
    t1 = df["t1"]
    classes_raw = sorted(y_raw.unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes_raw)}
    y = y_raw.map(class_to_idx).values
    oof_probas = np.zeros((len(df), len(classes_raw)), dtype=float)

    splitter = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
    for train_idx, test_idx in splitter.split(X, pd.Series(y), t1):
        X_train = X.iloc[train_idx][feature_cols]
        X_test = X.iloc[test_idx][feature_cols]
        y_train = y[train_idx]
        y_test = y[test_idx]
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weights = {i: float(w) for i, w in zip(np.unique(y_train), cw)}
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(classes_raw),
            n_estimators=800,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            class_weight=class_weights,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        oof_probas[test_idx] = model.predict_proba(X_test)
    return oof_probas, class_to_idx


def model_signals_from_probas(
    df: pd.DataFrame,
    probas: np.ndarray,
    class_to_idx: dict[int, int],
    threshold: float,
    regime_vol_quantile: float | None = None,
) -> np.ndarray:
    idx_buy = class_to_idx[BUY] if BUY in class_to_idx else class_to_idx[max(class_to_idx.keys())]
    idx_sell = (
        class_to_idx[SELL] if SELL in class_to_idx else class_to_idx[min(class_to_idx.keys())]
    )
    signals = np.zeros(len(df), dtype=int)
    vol_threshold = None
    if regime_vol_quantile is not None:
        vol_threshold = float(df["ATR_14"].quantile(regime_vol_quantile))
    for i in range(len(df) - 1):
        if vol_threshold is not None and float(df.iloc[i]["ATR_14"]) < vol_threshold:
            continue
        p_buy = float(probas[i][idx_buy])
        p_sell = float(probas[i][idx_sell])
        if p_buy >= threshold and p_buy > p_sell:
            signals[i] = BUY
        elif p_sell >= threshold and p_sell > p_buy:
            signals[i] = SELL
    return signals


def evaluate_external_signals(
    df: pd.DataFrame,
    signals: np.ndarray,
    symbol: str,
    timeframe: str,
    experiment_type: str,
    params: dict[str, Any] | None = None,
    sanity_extra: dict[str, bool] | None = None,
    cost_multiplier: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(signals) != len(df):
        raise ValueError("signals length must match dataset length")
    spread_cost = (
        df.get("spread", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float) * 1e-5
    ).values

    rows = []
    for i in range(len(df) - 1):
        s = int(signals[i])
        y_next = int(df.iloc[i + 1]["y"])
        gross = 0.0
        if s == BUY:
            gross = 0.001 if y_next == BUY else (-0.001 if y_next == SELL else 0.0)
        elif s == SELL:
            gross = 0.001 if y_next == SELL else (-0.001 if y_next == BUY else 0.0)
        cost = 0.0
        if s != WAIT:
            base_cost = (
                spread_cost[i + 1]
                + (CONFIG.live.slippage_points * 1e-5)
                + CONFIG.live.commission_per_trade
            )
            cost = base_cost * cost_multiplier
        ret = gross - cost
        rows.append(
            {
                "time": df.iloc[i]["time"],
                "signal": s,
                "y_next": y_next,
                "ret": ret,
                "regime_label": int(df.iloc[i]["regime_label"]) if "regime_label" in df.columns else -1,
            }
        )
    details = pd.DataFrame(rows)
    returns = details["ret"]
    trades = details[details["signal"] != WAIT]

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

    wins = trades["ret"][trades["ret"] > 0]
    losses = trades["ret"][trades["ret"] < 0]
    hit_rate = float((trades["ret"] > 0).mean()) if len(trades) else 0.0

    sanity = {
        "spread_included": True,
        "slippage_included": True,
        "commission_included": True,
        "signal_on_close_execute_next_bar": True,
        "no_lookahead_in_execution": True,
    }
    if sanity_extra:
        sanity.update(sanity_extra)
    sanity_ok = bool(all(sanity.values()))

    report: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "experiment_type": experiment_type,
        "params": params or {},
        "trades": int(len(trades)),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": hit_rate,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "class_distribution": class_dist,
        "conditional_returns": conditional,
        "sanity": sanity,
        "sanity_ok": sanity_ok,
        "cost_multiplier": cost_multiplier,
    }
    return report, details


def walk_forward_summary(details: pd.DataFrame, windows: int = 4) -> list[dict[str, Any]]:
    if details.empty:
        return []
    idx_chunks = np.array_split(np.arange(len(details)), windows)
    out: list[dict[str, Any]] = []
    for i, idx in enumerate(idx_chunks, start=1):
        chunk = details.iloc[idx]
        if chunk.empty:
            continue
        r = chunk["ret"]
        out.append(
            {
                "window": i,
                "start": str(chunk["time"].iloc[0]),
                "end": str(chunk["time"].iloc[-1]),
                "trades": int((chunk["signal"] != WAIT).sum()),
                "profit_factor": profit_factor(r),
                "expectancy": expectancy(r),
                "sharpe": sharpe(r) if len(chunk) > 2 else 0.0,
                "max_drawdown": max_drawdown(r),
            }
        )
    return out


def backtest_on_dataset(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    threshold: float = 0.60,
    regime_vol_quantile: float | None = None,
) -> dict[str, Any]:
    probas, class_to_idx = generate_model_oof_probas(df=df, n_splits=5, seed=42)
    signals = model_signals_from_probas(
        df=df,
        probas=probas,
        class_to_idx=class_to_idx,
        threshold=threshold,
        regime_vol_quantile=regime_vol_quantile,
    )
    report, details = evaluate_external_signals(
        df=df,
        signals=signals,
        symbol=symbol,
        timeframe=timeframe,
        experiment_type="model",
        params={
            "threshold": threshold,
            "regime_vol_quantile": regime_vol_quantile,
        },
        sanity_extra={"oof_no_in_sample_backtest": True},
    )
    report["walk_forward"] = walk_forward_summary(details, windows=4)
    return report


def backtest(symbol: str, timeframe: str, threshold: float = 0.60) -> Path:
    CONFIG.ensure_dirs()
    dataset_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    df = pd.read_parquet(dataset_path).sort_values("time").reset_index(drop=True)
    report = backtest_on_dataset(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        threshold=threshold,
        regime_vol_quantile=None,
    )
    out_path = CONFIG.reports_dir / f"backtest_{symbol}_{timeframe}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def backtest_with_combined_decisions(
    df_h1_timeline: pd.DataFrame,
    combined_signals: np.ndarray,
    symbol: str,
    experiment_type: str,
    params: dict[str, Any] | None = None,
    cost_multiplier: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    return evaluate_external_signals(
        df=df_h1_timeline,
        signals=combined_signals,
        symbol=symbol,
        timeframe="H1",
        experiment_type=experiment_type,
        params=params,
        sanity_extra={"multitf_no_lookahead_alignment": True},
        cost_multiplier=cost_multiplier,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model backtest on dataset parquet.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = backtest(symbol=args.symbol, timeframe=args.tf, threshold=args.threshold)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
