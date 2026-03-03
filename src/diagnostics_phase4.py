from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .config import CONFIG
from .labeling_triple_barrier import triple_barrier_labels
from .metrics import expectancy, max_drawdown, profit_factor, sharpe, sortino

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


DROP_COLS = {"time", "y", "t1", "pt", "sl"}


def _simulate(
    df: pd.DataFrame,
    decisions: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    spread_cost = (
        df.get("spread", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float) * 1e-5
    ).values
    returns = []
    rows = []
    for i in range(len(df) - 1):
        d = int(decisions[i])
        y_next = int(df.iloc[i + 1]["y"])
        gross = 0.0
        if d == 1:
            gross = 0.001 if y_next == 1 else (-0.001 if y_next == -1 else 0.0)
        elif d == -1:
            gross = 0.001 if y_next == -1 else (-0.001 if y_next == 1 else 0.0)
        cost = 0.0
        if d != 0:
            cost = (
                spread_cost[i + 1]
                + (CONFIG.live.slippage_points * 1e-5)
                + CONFIG.live.commission_per_trade
            )
        r = gross - cost
        returns.append(r)
        rows.append(
            {
                "time": df.iloc[i]["time"],
                "decision": d,
                "y_next": y_next,
                "ret": r,
                "regime": int(df.iloc[i]["regime_label"]) if "regime_label" in df.columns else -1,
            }
        )
    return pd.Series(returns), pd.DataFrame(rows)


def _stats(name: str, returns: pd.Series, details: pd.DataFrame) -> dict:
    non_zero = details[details["decision"] != 0]
    buy = non_zero[non_zero["decision"] == 1]["ret"]
    sell = non_zero[non_zero["decision"] == -1]["ret"]
    out = {
        "name": name,
        "trades": int((details["decision"] != 0).sum()),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "max_drawdown": max_drawdown(returns),
        "E_r_buy": float(buy.mean()) if len(buy) else 0.0,
        "E_r_sell": float(sell.mean()) if len(sell) else 0.0,
    }
    by_regime = {}
    for regime, grp in non_zero.groupby("regime"):
        rg_ret = grp["ret"]
        by_regime[str(regime)] = {
            "trades": int(len(grp)),
            "profit_factor": profit_factor(rg_ret),
            "expectancy": expectancy(rg_ret),
            "sharpe": sharpe(rg_ret) if len(grp) > 2 else 0.0,
        }
    out["by_regime"] = by_regime
    return out


def _build_oof_probas(df: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    from .cv_purged import PurgedKFold

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[["time"] + feature_cols]
    y_raw = df["y"].astype(int)
    t1 = df["t1"]
    classes_raw = sorted(y_raw.unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes_raw)}
    y = y_raw.map(class_to_idx).values
    oof_probas = np.zeros((len(df), len(classes_raw)), dtype=float)
    splitter = PurgedKFold(n_splits=5, embargo_pct=0.01)

    for train_idx, test_idx in splitter.split(X, pd.Series(y), t1):
        X_train = X.iloc[train_idx][feature_cols]
        X_test = X.iloc[test_idx][feature_cols]
        y_train = y[train_idx]
        y_test = y[test_idx]
        cw = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {i: float(w) for i, w in zip(np.unique(y_train), cw)}
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(classes_raw),
            n_estimators=800,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
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


def _model_decisions(
    df: pd.DataFrame,
    threshold: float,
    regime_q: float | None,
    invert: bool = False,
) -> np.ndarray:
    probas, class_to_idx = _build_oof_probas(df)
    idx_buy = class_to_idx[1] if 1 in class_to_idx else class_to_idx[max(class_to_idx.keys())]
    idx_sell = class_to_idx[-1] if -1 in class_to_idx else class_to_idx[min(class_to_idx.keys())]
    decisions = np.zeros(len(df), dtype=int)
    vol_thr = float(df["ATR_14"].quantile(regime_q)) if regime_q is not None else None
    for i in range(len(df) - 1):
        if vol_thr is not None and float(df.iloc[i]["ATR_14"]) < vol_thr:
            continue
        p_buy = probas[i][idx_buy]
        p_sell = probas[i][idx_sell]
        if p_buy >= threshold and p_buy > p_sell:
            decisions[i] = 1
        elif p_sell >= threshold and p_sell > p_buy:
            decisions[i] = -1
    if invert:
        decisions = -decisions
    return decisions


def _momentum_decisions(df: pd.DataFrame, n: int) -> np.ndarray:
    mom = np.log(df["close"]).diff(n)
    d = np.sign(mom).fillna(0.0).astype(int).values
    return d


def _mean_reversion_decisions(df: pd.DataFrame, window: int, k: float) -> np.ndarray:
    mean = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std().replace(0.0, np.nan)
    z = (df["close"] - mean) / std
    d = np.zeros(len(df), dtype=int)
    d[z < -k] = 1
    d[z > k] = -1
    return d


def _random_decisions(size: int, trade_rate: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active = rng.random(size) < trade_rate
    side = rng.choice([-1, 1], size=size)
    return np.where(active, side, 0).astype(int)


def run_diagnostics(
    symbol: str,
    timeframe: str,
    threshold: float,
    pt_mult: float,
    sl_mult: float,
    horizon: int,
    regime_q: float | None,
) -> Path:
    CONFIG.ensure_dirs()
    feat_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    feat_df = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True)
    df = triple_barrier_labels(
        feat_df,
        timeframe=timeframe,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        horizon=horizon,
    )

    label_dist = df["y"].value_counts(normalize=True).sort_index().to_dict()

    model_dec = _model_decisions(df, threshold=threshold, regime_q=regime_q, invert=False)
    model_ret, model_det = _simulate(df, model_dec)
    model_stats = _stats("model", model_ret, model_det)

    inv_dec = _model_decisions(df, threshold=threshold, regime_q=regime_q, invert=True)
    inv_ret, inv_det = _simulate(df, inv_dec)
    inv_stats = _stats("model_inverted", inv_ret, inv_det)

    baselines = []
    for n in (3, 6, 12, 24):
        d = _momentum_decisions(df, n=n)
        r, det = _simulate(df, d)
        baselines.append(_stats(f"momentum_n{n}", r, det))

    for k in (1.0, 1.5, 2.0):
        d = _mean_reversion_decisions(df, window=20, k=k)
        r, det = _simulate(df, d)
        baselines.append(_stats(f"meanrev_k{k}", r, det))

    trade_rate = float((model_dec != 0).mean())
    rnd = _random_decisions(len(df), trade_rate=trade_rate, seed=42)
    rnd_ret, rnd_det = _simulate(df, rnd)
    random_stats = _stats("random_matched_rate", rnd_ret, rnd_det)

    all_stats = [model_stats, inv_stats, random_stats] + baselines
    rank_df = pd.DataFrame(all_stats).sort_values(
        ["profit_factor", "expectancy", "sharpe"], ascending=[False, False, False]
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = CONFIG.reports_dir / f"diagnostics_{symbol}_{timeframe}_{ts}.json"
    out_csv = CONFIG.reports_dir / f"diagnostics_{symbol}_{timeframe}_{ts}.csv"
    rank_df.to_csv(out_csv, index=False)

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "setup": {
            "threshold": threshold,
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "horizon": horizon,
            "regime_q": regime_q,
        },
        "label_distribution": label_dist,
        "results_ranked": rank_df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_json={out_json}")
    print(f"saved_csv={out_csv}")
    return out_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 strategy diagnostics.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--regime-q", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diagnostics(
        symbol=args.symbol,
        timeframe=args.tf,
        threshold=args.threshold,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        horizon=args.horizon,
        regime_q=None if args.regime_q < 0 else args.regime_q,
    )


if __name__ == "__main__":
    main()

