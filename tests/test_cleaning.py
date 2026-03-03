import pandas as pd

from src.data_cleaning import clean_ohlc


def test_clean_ohlc_removes_duplicates_and_weekends():
    df = pd.DataFrame(
        {
            "time": [
                "2025-01-03T10:00:00Z",
                "2025-01-03T10:00:00Z",
                "2025-01-04T10:00:00Z",
                "2025-01-06T10:00:00Z",
            ],
            "open": [1, 1, 1, 1],
            "high": [2, 2, 2, 2],
            "low": [0.5, 0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5, 1.5],
            "tick_volume": [10, 10, 10, 10],
        }
    )
    out, report = clean_ohlc(df, timeframe="H1")
    assert len(out) == 2
    assert out["time"].dt.dayofweek.max() < 5
    assert report.candles == 2

