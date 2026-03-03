from __future__ import annotations

import numpy as np
import pandas as pd

BUY = 1
SELL = -1
WAIT = 0


def align_h4_to_h1(
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    h1_time_col: str = "time",
    h4_time_col: str = "time",
) -> pd.DataFrame:
    left = h1_df.sort_values(h1_time_col).copy()
    right = h4_df.sort_values(h4_time_col).copy()
    left[h1_time_col] = pd.to_datetime(left[h1_time_col], utc=True, errors="coerce")
    right[h4_time_col] = pd.to_datetime(right[h4_time_col], utc=True, errors="coerce")
    left = left[left[h1_time_col].notna()].copy()
    right = right[right[h4_time_col].notna()].copy()
    left_ns = (
        left[h1_time_col]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    right_ns = (
        right[h4_time_col]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    left["_h1_ts_ns"] = left_ns.astype("int64")
    right["_h4_ts_ns"] = right_ns.astype("int64")
    right = right.rename(columns={h4_time_col: "h4_time"})
    aligned = pd.merge_asof(
        left,
        right,
        left_on="_h1_ts_ns",
        right_on="_h4_ts_ns",
        direction="backward",
    )
    aligned["has_h4_pred"] = aligned["h4_time"].notna()
    aligned = aligned.drop(columns=["_h1_ts_ns", "_h4_ts_ns"], errors="ignore")
    return aligned


def apply_policy(
    df: pd.DataFrame,
    policy: str,
    threshold_final: float = 0.60,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    policy = policy.upper()
    out = df.copy()
    h1_signal = out["signal_h1"].fillna(0).astype(int).values
    h4_signal = out["signal_h4_aligned"].fillna(0).astype(int).values
    h1_buy = out["prob_h1_buy"].astype(float).values
    h1_sell = out["prob_h1_sell"].astype(float).values
    h4_buy = out["prob_h4_buy_aligned"].astype(float).values
    h4_sell = out["prob_h4_sell_aligned"].astype(float).values
    has_h1_pred = (
        out["has_h1_pred"].fillna(False).astype(bool).values
        if "has_h1_pred" in out.columns
        else ~(np.isnan(h1_buy) | np.isnan(h1_sell))
    )
    has_h4_pred = (
        out["has_h4_pred"].fillna(False).astype(bool).values
        if "has_h4_pred" in out.columns
        else ~(np.isnan(h4_buy) | np.isnan(h4_sell))
    )

    decisions = np.zeros(len(out), dtype=int)
    reasons: list[str] = []
    score_buy = np.zeros(len(out), dtype=float)
    score_sell = np.zeros(len(out), dtype=float)

    for i in range(len(out)):
        reason = "ok"
        if not has_h1_pred[i]:
            decisions[i] = WAIT
            reasons.append("blocked_by_missing_h1_pred")
            continue
        if not has_h4_pred[i]:
            decisions[i] = WAIT
            reasons.append("blocked_by_missing_h4_pred")
            continue

        if policy == "H4_GATE_DIRECTION":
            if h4_signal[i] == WAIT:
                decisions[i] = WAIT
                reason = "blocked_by_gate_wait"
            elif h4_signal[i] == BUY:
                if h1_signal[i] == BUY:
                    decisions[i] = BUY
                else:
                    decisions[i] = WAIT
                    reason = "blocked_by_gate_direction"
            elif h4_signal[i] == SELL:
                if h1_signal[i] == SELL:
                    decisions[i] = SELL
                else:
                    decisions[i] = WAIT
                    reason = "blocked_by_gate_direction"

        elif policy == "DOUBLE_CONFIRMATION":
            if h4_signal[i] != WAIT and h4_signal[i] == h1_signal[i]:
                decisions[i] = h1_signal[i]
            else:
                decisions[i] = WAIT
                reason = "blocked_by_no_confirmation"

        elif policy == "ENSEMBLE_SCORE":
            score_sell[i] = (0.7 * h4_sell[i]) + (0.3 * h1_sell[i])
            score_buy[i] = (0.7 * h4_buy[i]) + (0.3 * h1_buy[i])
            best = max(score_buy[i], score_sell[i])
            if best > threshold_final:
                decisions[i] = BUY if score_buy[i] > score_sell[i] else SELL
            else:
                decisions[i] = WAIT
                reason = "blocked_by_low_ensemble_score"
        else:
            raise ValueError(f"Unknown policy: {policy}")
        reasons.append(reason)

    out["decision_final"] = decisions
    out["blocked_reason"] = reasons
    out["score_buy"] = score_buy
    out["score_sell"] = score_sell
    return decisions, reasons, out
