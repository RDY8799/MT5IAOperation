import numpy as np
import pandas as pd

from src.fracdiff import fracdiff, get_weights


def test_weights_truncate_and_nonempty():
    w = get_weights(d=0.4, size=100, thresh=1e-3)
    assert len(w) > 1
    assert np.isfinite(w).all()


def test_fracdiff_returns_series():
    s = pd.Series(np.linspace(1, 100, 100))
    out = fracdiff(s, d=0.4)
    assert len(out) == len(s)
    assert out.notna().sum() > 0

