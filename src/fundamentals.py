from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


@dataclass
class FundamentalWindowStatus:
    blocked: bool
    reason: str
    high_impact_in_next_60min: bool
    hours_since_last_high_impact: float
    next_event_at: str
    next_event_name: str


_CAL_CACHE: dict[str, Any] = {"path": "", "mtime": 0.0, "df": pd.DataFrame()}


def _normalize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}
    ts_col = cols.get("timestamp_utc") or cols.get("time_utc") or cols.get("datetime_utc") or cols.get("time")
    if ts_col is None:
        raise ValueError("Calendar CSV must contain timestamp_utc/time_utc/datetime_utc/time")
    out["timestamp_utc"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    impact_col = cols.get("impact") or cols.get("priority") or cols.get("importance")
    if impact_col is None:
        out["impact"] = "high"
    else:
        out["impact"] = out[impact_col].astype(str).str.lower()
    evt_col = cols.get("event") or cols.get("title") or cols.get("name")
    out["event"] = out[evt_col].astype(str) if evt_col else "high_impact_event"
    cur_col = cols.get("currency") or cols.get("ccy")
    out["currency"] = out[cur_col].astype(str).str.upper() if cur_col else "ALL"
    out = out.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    return out


def load_calendar(path: Path | None = None, force: bool = False) -> pd.DataFrame:
    csv_path = Path(path or CONFIG.fundamentals.calendar_csv)
    if not csv_path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "impact", "event", "currency"])
    mtime = float(csv_path.stat().st_mtime)
    if (
        not force
        and _CAL_CACHE["path"] == str(csv_path)
        and float(_CAL_CACHE["mtime"]) == mtime
        and isinstance(_CAL_CACHE["df"], pd.DataFrame)
    ):
        return _CAL_CACHE["df"].copy()
    raw = pd.read_csv(csv_path)
    norm = _normalize_calendar(raw)
    _CAL_CACHE["path"] = str(csv_path)
    _CAL_CACHE["mtime"] = mtime
    _CAL_CACHE["df"] = norm
    return norm.copy()


def _high_impact(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["impact"].str.contains("high", case=False, na=False)].copy()


def get_fundamental_window_status(
    now_utc: datetime,
    *,
    pre_minutes: int | None = None,
    post_minutes: int | None = None,
    next_window_minutes: int | None = None,
    calendar_df: pd.DataFrame | None = None,
) -> FundamentalWindowStatus:
    cal = load_calendar() if calendar_df is None else calendar_df
    cal = _high_impact(cal)
    pre = int(CONFIG.live.news_blackout_minutes_pre if pre_minutes is None else pre_minutes)
    post = int(CONFIG.live.news_blackout_minutes_post if post_minutes is None else post_minutes)
    next_win = int(CONFIG.fundamentals.next_event_window_minutes if next_window_minutes is None else next_window_minutes)
    if cal.empty:
        return FundamentalWindowStatus(
            blocked=False,
            reason="",
            high_impact_in_next_60min=False,
            hours_since_last_high_impact=1e9,
            next_event_at="",
            next_event_name="",
        )
    times = pd.to_datetime(cal["timestamp_utc"], utc=True)
    now_ts = pd.Timestamp(now_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    delta_min = (times - now_ts).dt.total_seconds() / 60.0
    next_mask = delta_min.between(0.0, float(next_win), inclusive="both")
    in_pre = delta_min.between(0.0, float(pre), inclusive="both")
    in_post = delta_min.between(float(-post), 0.0, inclusive="both")
    blocked = bool(in_pre.any() or in_post.any())
    reason = "fundamental_blackout" if blocked else ""
    past = delta_min[delta_min <= 0.0]
    hours_since = float(abs(past.max()) / 60.0) if len(past) else 1e9
    next_rows = cal[next_mask]
    if not next_rows.empty:
        nr = next_rows.iloc[0]
        next_at = str(pd.Timestamp(nr["timestamp_utc"]).isoformat())
        next_name = str(nr.get("event", "high_impact_event"))
    else:
        future = cal[delta_min > 0.0]
        if not future.empty:
            fr = future.iloc[0]
            next_at = str(pd.Timestamp(fr["timestamp_utc"]).isoformat())
            next_name = str(fr.get("event", "high_impact_event"))
        else:
            next_at = ""
            next_name = ""
    return FundamentalWindowStatus(
        blocked=blocked,
        reason=reason,
        high_impact_in_next_60min=bool(next_mask.any()),
        hours_since_last_high_impact=hours_since,
        next_event_at=next_at,
        next_event_name=next_name,
    )


def add_fundamental_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()
    if time_col not in out.columns:
        out["high_impact_in_next_60min"] = 0
        out["hours_since_last_high_impact"] = 1e9
        return out
    cal = _high_impact(load_calendar())
    t = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if cal.empty or t.isna().all():
        out["high_impact_in_next_60min"] = 0
        out["hours_since_last_high_impact"] = 1e9
        return out
    evt = pd.to_datetime(cal["timestamp_utc"], utc=True).sort_values().reset_index(drop=True)
    vals = t.astype("int64").to_numpy()
    evt_vals = evt.astype("int64").to_numpy()
    idx_next = np.searchsorted(evt_vals, vals, side="left")
    idx_prev = np.searchsorted(evt_vals, vals, side="right") - 1

    next_delta_h = np.full(len(out), np.inf, dtype=float)
    prev_delta_h = np.full(len(out), np.inf, dtype=float)

    ok_next = (idx_next >= 0) & (idx_next < len(evt_vals))
    ok_prev = (idx_prev >= 0) & (idx_prev < len(evt_vals))
    next_delta_h[ok_next] = (evt_vals[idx_next[ok_next]] - vals[ok_next]) / 3_600_000_000_000.0
    prev_delta_h[ok_prev] = (vals[ok_prev] - evt_vals[idx_prev[ok_prev]]) / 3_600_000_000_000.0
    out["high_impact_in_next_60min"] = (next_delta_h >= 0.0) & (next_delta_h <= 1.0)
    out["high_impact_in_next_60min"] = out["high_impact_in_next_60min"].astype(int)
    out["hours_since_last_high_impact"] = np.where(np.isfinite(prev_delta_h), prev_delta_h, 1e9)
    return out


def fetch_dxy_strength(
    *,
    timeframe_mt5: Any,
    bars: int = 120,
    symbols: tuple[str, ...] | None = None,
) -> float:
    if mt5 is None or timeframe_mt5 is None:
        return 0.0
    symbols = symbols or CONFIG.live.dxy_symbols
    for sym in symbols:
        rates = mt5.copy_rates_from_pos(sym, timeframe_mt5, 0, bars)
        if rates is None or len(rates) < 30:
            continue
        df = pd.DataFrame(rates)
        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) < 30:
            continue
        # Proxy simples de força: retorno de 24 barras + slope EMA 20.
        ret24 = float(close.iloc[-1] / close.iloc[-25] - 1.0) if len(close) >= 25 else 0.0
        ema20 = close.ewm(span=20, adjust=False).mean()
        slope = float(ema20.iloc[-1] - ema20.iloc[-2]) if len(ema20) >= 2 else 0.0
        return float(ret24 + slope)
    return 0.0
