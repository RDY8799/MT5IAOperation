from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights(d: float, size: int, thresh: float = 1e-5) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w[::-1], dtype=float)


def fracdiff(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    x = series.astype(float).copy()
    weights = get_weights(d=d, size=len(x), thresh=thresh)
    width = len(weights)
    out = pd.Series(index=x.index, dtype=float)
    for i in range(width - 1, len(x)):
        window = x.iloc[i - width + 1 : i + 1].values
        out.iloc[i] = float(np.dot(weights, window))
    return out


def choose_d_by_adf(
    series: pd.Series,
    d_values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6),
    p_threshold: float = 0.05,
    thresh: float = 1e-5,
) -> float:
    best_d = d_values[-1]
    for d in d_values:
        fd = fracdiff(series, d=d, thresh=thresh).dropna()
        if len(fd) < 50:
            continue
        p_val = adfuller(fd, maxlag=1, autolag="AIC")[1]
        if p_val < p_threshold:
            best_d = d
            break
    return best_d

