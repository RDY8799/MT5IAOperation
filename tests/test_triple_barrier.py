import pandas as pd

from src.labeling_triple_barrier import triple_barrier_labels


def test_triple_barrier_creates_labels():
    rows = []
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    for i in range(20):
        price = 1.1000 + i * 0.0002
        rows.append(
            {
                "time": base + pd.Timedelta(hours=i),
                "open": price,
                "high": price + 0.0003,
                "low": price - 0.0002,
                "close": price + 0.0001,
                "ATR_14": 0.0005,
            }
        )
    df = pd.DataFrame(rows)
    out = triple_barrier_labels(df, timeframe="H1", horizon=3, pt_mult=0.5, sl_mult=0.5)
    assert "y" in out.columns
    assert out["y"].isin([-1, 0, 1]).all()
    assert "t1" in out.columns

