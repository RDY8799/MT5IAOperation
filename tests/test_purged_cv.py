import pandas as pd

from src.cv_purged import PurgedKFold


def test_purged_kfold_produces_non_overlapping_splits():
    n = 100
    time = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    t1 = time + pd.Timedelta(hours=3)
    X = pd.DataFrame({"time": time, "x": range(n)})
    y = pd.Series([0] * n)
    splitter = PurgedKFold(n_splits=5, embargo_pct=0.01)
    for train_idx, test_idx in splitter.split(X, y, t1):
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(test_idx) > 0

