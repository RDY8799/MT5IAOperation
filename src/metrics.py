from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix


def class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in sorted(np.unique(y_true)):
        mask = y_true == cls
        out[str(cls)] = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else 0.0
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    labels = sorted(np.unique(y_true))
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside)


def profit_factor(returns: pd.Series) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    if gross_loss == 0:
        return 0.0
    return float(gross_profit / gross_loss)


def expectancy(returns: pd.Series) -> float:
    return float(returns.mean())


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def brier_multiclass(y_true: np.ndarray, probas: np.ndarray, classes: list[int]) -> float:
    score = 0.0
    for i, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(int)
        score += brier_score_loss(binary_true, probas[:, i])
    return float(score / len(classes))

