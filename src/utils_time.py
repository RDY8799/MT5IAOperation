from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


TIMEFRAME_TO_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def timeframe_to_pandas_freq(tf: str) -> str:
    minutes = TIMEFRAME_TO_MINUTES[tf]
    if minutes < 60:
        return f"{minutes}min"
    if minutes % 60 == 0 and minutes < 1440:
        return f"{minutes // 60}h"
    return "1d"


def timeframe_minutes(tf: str) -> int:
    return TIMEFRAME_TO_MINUTES[tf]


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def months_ago(months: int) -> datetime:
    return utc_now() - timedelta(days=30 * months)


def wait_seconds_to_next_candle(last_time: pd.Timestamp, tf: str) -> float:
    _ = last_time
    now = utc_now()
    step_min = timeframe_minutes(tf)
    minute_of_day = now.hour * 60 + now.minute
    next_minute_of_day = ((minute_of_day // step_min) + 1) * step_min
    day_offset = next_minute_of_day // (24 * 60)
    next_minute_of_day = next_minute_of_day % (24 * 60)
    next_hour = next_minute_of_day // 60
    next_min = next_minute_of_day % 60
    next_candle = datetime(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=next_hour,
        minute=next_min,
        second=0,
        tzinfo=timezone.utc,
    ) + timedelta(days=day_offset)
    remaining = (next_candle - now).total_seconds()
    return max(0.0, remaining)
