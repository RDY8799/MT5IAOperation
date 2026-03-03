from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .fracdiff import fracdiff
from .regimes import adx, classify_regimes, detect_vol_regime


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return_1"] = np.log(out["close"]).diff(1)
    out["log_return_3"] = np.log(out["close"]).diff(3)
    out["log_return_12"] = np.log(out["close"]).diff(12)
    out["ATR_14"] = atr(out, CONFIG.feature.atr_window)

    out["body"] = (out["close"] - out["open"]) / out["ATR_14"]
    out["range"] = (out["high"] - out["low"]) / out["ATR_14"]
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["ATR_14"]
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["ATR_14"]

    ema9, ema21, ema50 = CONFIG.feature.ema_windows
    out["EMA_9"] = out["close"].ewm(span=ema9, adjust=False).mean()
    out["EMA_21"] = out["close"].ewm(span=ema21, adjust=False).mean()
    out["EMA_50"] = out["close"].ewm(span=ema50, adjust=False).mean()
    out["MA_50"] = out["close"].rolling(50).mean()
    out["ma50_slope"] = out["MA_50"].diff()
    out["slope_ema21"] = out["EMA_21"].diff()
    out["dist_ema21"] = (out["close"] - out["EMA_21"]) / out["ATR_14"]
    out["dist_ema50"] = (out["close"] - out["EMA_50"]) / out["ATR_14"]
    out["RSI_14"] = rsi(out["close"], CONFIG.feature.rsi_window)
    out["rolling_vol_20"] = np.log(out["close"]).diff().rolling(20).std()
    out["resistance_20"] = out["high"].rolling(20).max()
    out["support_20"] = out["low"].rolling(20).min()
    out["resistance_50"] = out["high"].rolling(50).max()
    out["support_50"] = out["low"].rolling(50).min()
    out["dist_resistance_20"] = (out["resistance_20"] - out["close"]) / out["ATR_14"]
    out["dist_support_20"] = (out["close"] - out["support_20"]) / out["ATR_14"]
    out["dist_resistance_50"] = (out["resistance_50"] - out["close"]) / out["ATR_14"]
    out["dist_support_50"] = (out["close"] - out["support_50"]) / out["ATR_14"]
    out["is_breakout_up_20"] = (out["close"] > out["resistance_20"].shift(1)).astype(int)
    out["is_breakout_down_20"] = (out["close"] < out["support_20"].shift(1)).astype(int)
    out["ADX_14"] = adx(out, window=14)
    out["regime_label"] = detect_vol_regime(out["close"])
    out["fracdiff_close"] = fracdiff(
        np.log(out["close"]),
        d=CONFIG.feature.fracdiff_d,
        thresh=CONFIG.fracdiff_threshold,
    )
    out = classify_regimes(out)
    out = out.dropna().reset_index(drop=True)
    return out


def process_features(symbol: str, timeframe: str) -> Path:
    CONFIG.ensure_dirs()
    in_path = CONFIG.data_raw_dir / f"{symbol}_{timeframe}.parquet"
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    df = pd.read_parquet(in_path).sort_values("time").reset_index(drop=True)
    feats = build_features(df)
    out_path = CONFIG.data_processed_dir / f"{symbol}_{timeframe}_features.parquet"
    feats.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leak-free features.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tf", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = process_features(symbol=args.symbol, timeframe=args.tf)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
