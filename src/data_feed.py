from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from io import StringIO
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

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


def _as_utc_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _empty_rates() -> pd.DataFrame:
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


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0 if col not in {"time"} else pd.NaT
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.dropna(subset=["time"])
    out = out.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return out[cols]


def _timeframe_minutes(timeframe: str) -> int:
    tf_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return tf_map[timeframe]


def fetch_rates_mt5(
    symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    tf_value = TF_TO_MT5.get(timeframe)
    if tf_value is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    start = _as_utc_ts(start_dt)
    end = _as_utc_ts(end_dt)
    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=30), end)
        rates = mt5.copy_rates_range(symbol, tf_value, cursor.to_pydatetime(), nxt.to_pydatetime())
        if rates is not None and len(rates) > 0:
            part = pd.DataFrame(rates)
            part["time"] = pd.to_datetime(part["time"], unit="s", utc=True)
            chunks.append(part)
        cursor = nxt
    if not chunks:
        return _empty_rates()
    merged = pd.concat(chunks, ignore_index=True)
    return _normalize_ohlc(merged)


def _yahoo_ticker(symbol: str) -> str:
    s = symbol.upper()
    if "=" in s:
        return s
    # Forex/metals CFDs on Yahoo generally use "=X" suffix (EURUSD=X, XAUUSD=X).
    if len(s) in {6, 7} and s.isalnum():
        return f"{s}=X"
    return s


def _resample_from_1h_to_h4(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy().set_index("time").sort_index()
    agg = x.resample("4H", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "mean",
            "real_volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return _normalize_ohlc(agg)


def fetch_rates_yahoo(
    symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    tf = timeframe.upper()
    interval_map = {
        "M1": "1m",
        "M5": "5m",
        "M15": "15m",
        "M30": "30m",
        "H1": "60m",
        "H4": "60m",
        "D1": "1d",
    }
    interval = interval_map.get(tf)
    if interval is None:
        raise ValueError(f"Unsupported timeframe for Yahoo fallback: {timeframe}")

    p1 = int(_as_utc_ts(start_dt).timestamp())
    p2 = int(_as_utc_ts(end_dt).timestamp())
    ticker = quote(_yahoo_ticker(symbol), safe="=.")
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={p1}&period2={p2}&interval={interval}&events=history&includeAdjustedClose=true"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except (URLError, TimeoutError):
        return _empty_rates()
    if "Date,Open,High,Low,Close" not in text and "Datetime,Open,High,Low,Close" not in text:
        return _empty_rates()

    raw = pd.read_csv(StringIO(text))
    if raw.empty:
        return _empty_rates()
    dt_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["time"] = pd.to_datetime(raw[dt_col], utc=True, errors="coerce")
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    vol = raw["Volume"] if "Volume" in raw.columns else 0
    out = pd.DataFrame(
        {
            "time": raw["time"],
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "tick_volume": pd.to_numeric(vol, errors="coerce").fillna(0.0),
            "spread": 0.0,
            "real_volume": pd.to_numeric(vol, errors="coerce").fillna(0.0),
        }
    )
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = _normalize_ohlc(out)
    if tf == "H4":
        out = _resample_from_1h_to_h4(out)
    return out


def fetch_rates_with_fallback(
    symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    source: str = "auto",
    credentials: MT5Credentials | None = None,
) -> tuple[pd.DataFrame, str]:
    mode = source.lower().strip()
    if mode not in {"auto", "mt5", "yahoo"}:
        raise ValueError("source must be one of: auto, mt5, yahoo")

    def _try_mt5() -> pd.DataFrame:
        if not ensure_logged_in(credentials=credentials):
            return _empty_rates()
        return fetch_rates_mt5(symbol=symbol, timeframe=timeframe, start_dt=start_dt, end_dt=end_dt)

    if mode == "mt5":
        return _try_mt5(), "mt5"
    if mode == "yahoo":
        return fetch_rates_yahoo(symbol=symbol, timeframe=timeframe, start_dt=start_dt, end_dt=end_dt), "yahoo"

    mt5_df = _try_mt5()
    if len(mt5_df) > 0:
        return mt5_df, "mt5"
    yahoo_df = fetch_rates_yahoo(symbol=symbol, timeframe=timeframe, start_dt=start_dt, end_dt=end_dt)
    if len(yahoo_df) > 0:
        return yahoo_df, "yahoo"
    return _empty_rates(), "none"


def run_collection(
    symbol: str,
    timeframes: list[str],
    months: int,
    credentials: MT5Credentials | None = None,
    source: str = "auto",
) -> None:
    CONFIG.ensure_dirs()
    start_dt = months_ago(months)
    end_dt = utc_now()
    mt5_was_used = False
    try:
        for tf in timeframes:
            raw, used_source = fetch_rates_with_fallback(
                symbol=symbol,
                timeframe=tf,
                start_dt=start_dt,
                end_dt=end_dt,
                source=source,
                credentials=credentials,
            )
            if used_source == "mt5":
                mt5_was_used = True
            clean, report = clean_ohlc(raw, timeframe=tf)
            path = save_raw_parquet(clean, symbol=symbol, timeframe=tf)
            print(
                f"[{tf}] source={used_source} saved={path} candles={report.candles} "
                f"missing_pct={report.missing_pct:.4f} largest_gap_min={report.largest_gap_minutes:.2f}"
            )
    finally:
        if mt5_was_used:
            shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MT5 candles and save parquet.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tfs", nargs="+", required=True, help="Ex: M5 M15 H1")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--source", choices=["auto", "mt5", "yahoo"], default="auto")
    parser.add_argument("--login", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    creds = None
    if args.login and args.password and args.server:
        creds = MT5Credentials(login=args.login, password=args.password, server=args.server)
    run_collection(
        symbol=args.symbol,
        timeframes=args.tfs,
        months=args.months,
        credentials=creds,
        source=args.source,
    )


if __name__ == "__main__":
    main()
