from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .utils_time import timeframe_minutes


@dataclass
class QualityReport:
    candles: int
    missing_pct: float
    largest_gap_minutes: float


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out


def remove_duplicates_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def remove_weekends(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["time"].dt.dayofweek < 5].reset_index(drop=True)


def remove_large_gap_rows(df: pd.DataFrame, timeframe: str, factor: int = 6) -> pd.DataFrame:
    if df.empty:
        return df
    expected = timeframe_minutes(timeframe)
    diffs = df["time"].diff().dt.total_seconds().div(60.0)
    keep = (diffs.isna()) | (diffs <= expected * factor)
    return df[keep].reset_index(drop=True)


def basic_quality_report(df: pd.DataFrame) -> QualityReport:
    if df.empty:
        return QualityReport(candles=0, missing_pct=100.0, largest_gap_minutes=0.0)
    numeric_cols = [c for c in ("open", "high", "low", "close", "tick_volume") if c in df.columns]
    missing = float(df[numeric_cols].isna().mean().mean() * 100.0) if numeric_cols else 0.0
    largest_gap = df["time"].diff().dt.total_seconds().div(60.0).max()
    largest_gap = float(np.nan_to_num(largest_gap, nan=0.0))
    return QualityReport(candles=len(df), missing_pct=missing, largest_gap_minutes=largest_gap)


def clean_ohlc(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, QualityReport]:
    out = to_utc(df)
    out = remove_duplicates_sort(out)
    out = remove_weekends(out)
    out = remove_large_gap_rows(out, timeframe=timeframe)
    report = basic_quality_report(out)
    return out, report


def save_raw_parquet(df: pd.DataFrame, symbol: str, timeframe: str, out_dir: Path | None = None) -> Path:
    CONFIG.ensure_dirs()
    directory = out_dir or CONFIG.data_raw_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(path, index=False)
    return path

