from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class PurgedKFold:
    n_splits: int = 5
    embargo_pct: float = 0.01
    shuffle: bool = False
    random_state: int | None = None

    def split(
        self, X: pd.DataFrame, y: pd.Series, t1: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.arange(n)
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        t1_dt = pd.to_datetime(t1, utc=True)
        times = pd.to_datetime(X["time"], utc=True)
        embargo = int(np.ceil(n * self.embargo_pct))

        for train_idx, test_idx in kf.split(indices):
            test_start = times.iloc[test_idx].min()
            test_end = times.iloc[test_idx].max()

            overlaps = (times <= test_end) & (t1_dt >= test_start)
            valid_train = np.setdiff1d(train_idx, np.where(overlaps)[0], assume_unique=False)

            test_max = int(np.max(test_idx))
            embargo_end = min(n, test_max + embargo + 1)
            embargo_idx = np.arange(test_max + 1, embargo_end)
            valid_train = np.setdiff1d(valid_train, embargo_idx, assume_unique=False)

            yield valid_train, test_idx

