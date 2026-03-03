from __future__ import annotations

import numpy as np
import pandas as pd


def detect_vol_regime(close: pd.Series, window: int = 50) -> pd.Series:
    returns = np.log(close).diff()
    vol = returns.rolling(window).std()
    med = vol.median()
    regime = (vol > med).astype(int)
    regime.name = "regime_label"
    return regime


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(window).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(window).mean() / atr
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    return dx.rolling(window).mean()


def classify_regimes(
    df: pd.DataFrame,
    adx_col: str = "ADX_14",
    atr_col: str = "ATR_14",
    ma50_slope_col: str = "ma50_slope",
    atr_percentile: float = 0.60,
    slope_threshold_mult: float = 0.10,
) -> pd.DataFrame:
    out = df.copy()
    atr_thr = float(out[atr_col].quantile(atr_percentile))
    slope_thr = float((out[atr_col] * slope_threshold_mult).median())
    out["is_sideways"] = out[adx_col] < 15.0
    out["is_trend"] = (out[adx_col] > 20.0) & (out[ma50_slope_col].abs() > slope_thr)
    out["is_high_vol"] = out[atr_col] > atr_thr
    regime = np.full(len(out), "NEUTRAL", dtype=object)
    regime[out["is_sideways"].values] = "SIDEWAYS"
    regime[out["is_high_vol"].values] = "HIGH_VOL"
    regime[out["is_trend"].values] = "TREND"
    out["regime_class"] = regime
    return out
