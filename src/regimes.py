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


def bollinger_bandwidth(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    bw = (upper - lower) / ma.replace(0.0, np.nan)
    return bw.replace([np.inf, -np.inf], np.nan)


def hurst_exponent_series(close: pd.Series, window: int = 100) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=c.index, dtype=float)
    if len(c) < window:
        return out
    for i in range(window, len(c) + 1):
        w = c.iloc[i - window : i].dropna().values
        if len(w) < window // 2:
            continue
        x = np.diff(np.log(np.maximum(w, 1e-9)))
        if len(x) < 20:
            continue
        lags = np.arange(2, min(20, len(x) // 2))
        tau = []
        for lag in lags:
            s = np.std(x[lag:] - x[:-lag])
            tau.append(max(float(s), 1e-12))
        if len(tau) < 3:
            continue
        poly = np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)
        out.iloc[i - 1] = float(poly[0] * 2.0)
    return out


def classify_regimes(
    df: pd.DataFrame,
    adx_col: str = "ADX_14",
    atr_col: str = "ATR_14",
    ma50_slope_col: str = "ma50_slope",
    atr_ratio_col: str = "atr_ratio_14_50",
    bb_bw_col: str = "bb_bandwidth_20",
    hurst_col: str = "hurst_100",
) -> pd.DataFrame:
    out = df.copy()
    adx_s = pd.to_numeric(out.get(adx_col), errors="coerce").fillna(0.0)
    atr_s = pd.to_numeric(out.get(atr_col), errors="coerce").fillna(0.0)
    slope_s = pd.to_numeric(out.get(ma50_slope_col), errors="coerce").fillna(0.0)
    atr_ratio_s = pd.to_numeric(out.get(atr_ratio_col), errors="coerce").fillna(1.0)
    bb_bw_s = pd.to_numeric(out.get(bb_bw_col), errors="coerce").fillna(0.0)
    hurst_s = pd.to_numeric(out.get(hurst_col), errors="coerce").fillna(0.5)

    slope_thr = float((atr_s * 0.10).median()) if len(atr_s) else 0.0
    atr_hi = float(atr_ratio_s.quantile(0.70)) if len(atr_ratio_s) else 1.2
    atr_lo = float(atr_ratio_s.quantile(0.30)) if len(atr_ratio_s) else 0.8
    bb_hi = float(bb_bw_s.quantile(0.75)) if len(bb_bw_s) else 0.0
    bb_lo = float(bb_bw_s.quantile(0.30)) if len(bb_bw_s) else 0.0

    low_vol_sideways = (adx_s < 15.0) & (atr_ratio_s <= atr_lo) & (bb_bw_s <= bb_lo)
    trending_strong = (adx_s >= 25.0) & (slope_s.abs() > slope_thr) & (hurst_s >= 0.55)
    high_vol_breakout = (atr_ratio_s >= atr_hi) & (bb_bw_s >= bb_hi) & (adx_s >= 18.0)
    post_news_shock = (atr_ratio_s >= max(1.5, atr_hi)) & (adx_s < 18.0)
    ranging = (adx_s.between(15.0, 22.0, inclusive="both")) & (hurst_s.between(0.42, 0.58, inclusive="both"))

    regime = np.full(len(out), "RANGING", dtype=object)
    regime[low_vol_sideways.values] = "LOW_VOL_SIDEWAYS"
    regime[post_news_shock.values] = "POST_NEWS_SHOCK"
    regime[high_vol_breakout.values] = "HIGH_VOL_BREAKOUT"
    regime[trending_strong.values] = "TRENDING_STRONG"
    regime[ranging.values] = "RANGING"

    out["is_sideways"] = low_vol_sideways.astype(int)
    out["is_trend"] = trending_strong.astype(int)
    out["is_high_vol"] = high_vol_breakout.astype(int)
    out["regime_class"] = regime
    return out
