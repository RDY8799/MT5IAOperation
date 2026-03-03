from __future__ import annotations

import numpy as np
import pandas as pd

BUY = 1
SELL = -1
WAIT = 0


def momentum_signal(prices: pd.Series, lookback_n: int, threshold: float = 0.0) -> np.ndarray:
    ret = np.log(prices).diff(lookback_n)
    s = np.zeros(len(prices), dtype=int)
    s[ret > threshold] = BUY
    s[ret < -threshold] = SELL
    return s


def mean_reversion_signal(prices: pd.Series, window: int = 20, k: float = 1.5) -> np.ndarray:
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std().replace(0.0, np.nan)
    z = (prices - sma) / std
    s = np.zeros(len(prices), dtype=int)
    s[z < -k] = BUY
    s[z > k] = SELL
    return s


def random_signal(length: int, p_buy: float, p_sell: float, seed: int = 42) -> np.ndarray:
    if p_buy < 0 or p_sell < 0 or (p_buy + p_sell) > 1:
        raise ValueError("Invalid probabilities for random_signal")
    rng = np.random.default_rng(seed)
    draw = rng.random(length)
    s = np.zeros(length, dtype=int)
    s[draw < p_buy] = BUY
    s[(draw >= p_buy) & (draw < p_buy + p_sell)] = SELL
    return s


def flip_signal(signal: np.ndarray) -> np.ndarray:
    return -signal.astype(int)

