from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .config import CONFIG
from .cv_purged import PurgedKFold
from .metrics import brier_multiclass, class_accuracy, confusion
from .model_registry import save_model_bundle

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


DROP_COLS = {"time", "y", "t1", "pt", "sl", "regime_class"}


def _class_map(y: pd.Series) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    classes = sorted(y.unique().tolist())
    to_idx = {c: i for i, c in enumerate(classes)}
    from_idx = {i: c for c, i in to_idx.items()}
    y_idx = y.map(to_idx).values
    return y_idx, to_idx, from_idx


def train(symbol: str, timeframe: str, n_splits: int = 5, seed: int = 42) -> tuple[str, str]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    dataset_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    df = pd.read_parquet(dataset_path).sort_values("time").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[["time"] + feature_cols]
    y = df["y"].astype(int)
    t1 = df["t1"]
    y_idx, _, from_idx = _class_map(y)
    class_labels = [from_idx[i] for i in sorted(from_idx)]

    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_idx), y=y_idx)
    class_weights = {i: float(w) for i, w in zip(np.unique(y_idx), cw)}

    splitter = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
    oof_pred = np.zeros(len(df), dtype=int)
    oof_proba = np.zeros((len(df), len(class_labels)), dtype=float)
    last_model = None

    for train_idx, test_idx in splitter.split(X, y, t1):
        X_train = X.iloc[train_idx][feature_cols]
        X_test = X.iloc[test_idx][feature_cols]
        y_train = y_idx[train_idx]
        y_test = y_idx[test_idx]

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(class_labels),
            n_estimators=1200,
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
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)
        oof_pred[test_idx] = pred
        oof_proba[test_idx] = proba
        last_model = model

    if last_model is None:
        raise RuntimeError("Training failed: no folds produced.")

    y_true = y.values
    y_pred = np.array([class_labels[i] for i in oof_pred])
    acc = class_accuracy(y_true, y_pred)
    cm = confusion(y_true, y_pred)
    brier = brier_multiclass(y_true, oof_proba, class_labels)
    metrics = {"class_accuracy": acc, "confusion_matrix": cm, "brier_score": brier}

    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "features": feature_cols,
        "metrics": metrics,
        "class_labels": class_labels,
        "seed": seed,
        "params": {
            "n_splits": n_splits,
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        "dataset_range": {
            "start": str(pd.Timestamp(df["time"].min())),
            "end": str(pd.Timestamp(df["time"].max())),
            "rows": int(len(df)),
        },
    }
    model_path, meta_path = save_model_bundle(last_model, metadata, symbol, timeframe)
    print(json.dumps(metrics, indent=2))
    return str(model_path), str(meta_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM with PurgedKFold.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path, meta_path = train(symbol=args.symbol, timeframe=args.tf, n_splits=args.splits, seed=args.seed)
    print(f"model={model_path}")
    print(f"meta={meta_path}")


if __name__ == "__main__":
    main()
