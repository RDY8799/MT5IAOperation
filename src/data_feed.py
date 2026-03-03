from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from .config import CONFIG
from .data_cleaning import clean_ohlc, save_raw_parquet
from .mt5_connect import MT5Credentials, ensure_logged_in, shutdown
from .utils_time import months_ago, utc_now

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


TF_TO_MT5 = {
    "M1": getattr(mt5, "TIMEFRAME_M1", None),
    "M5": getattr(mt5, "TIMEFRAME_M5", None),
    "M15": getattr(mt5, "TIMEFRAME_M15", None),
    "M30": getattr(mt5, "TIMEFRAME_M30", None),
    "H1": getattr(mt5, "TIMEFRAME_H1", None),
    "H4": getattr(mt5, "TIMEFRAME_H4", None),
    "D1": getattr(mt5, "TIMEFRAME_D1", None),
}


def fetch_rates(
    symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    tf_value = TF_TO_MT5.get(timeframe)
    if tf_value is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    rates = mt5.copy_rates_range(symbol, tf_value, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame(
            columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            ]
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]


def run_collection(
    symbol: str, timeframes: list[str], months: int, credentials: MT5Credentials | None = None
) -> None:
    CONFIG.ensure_dirs()
    if not ensure_logged_in(credentials=credentials):
        raise RuntimeError("Failed to initialize/login in MT5.")
    start_dt = months_ago(months)
    end_dt = utc_now()
    try:
        for tf in timeframes:
            raw = fetch_rates(symbol=symbol, timeframe=tf, start_dt=start_dt, end_dt=end_dt)
            clean, report = clean_ohlc(raw, timeframe=tf)
            path = save_raw_parquet(clean, symbol=symbol, timeframe=tf)
            print(
                f"[{tf}] saved={path} candles={report.candles} "
                f"missing_pct={report.missing_pct:.4f} largest_gap_min={report.largest_gap_minutes:.2f}"
            )
    finally:
        shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MT5 candles and save parquet.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tfs", nargs="+", required=True, help="Ex: M5 M15 H1")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--login", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    creds = None
    if args.login and args.password and args.server:
        creds = MT5Credentials(login=args.login, password=args.password, server=args.server)
    run_collection(symbol=args.symbol, timeframes=args.tfs, months=args.months, credentials=creds)


if __name__ == "__main__":
    main()
