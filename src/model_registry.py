from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .config import CONFIG


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_model_bundle(
    model: Any,
    metadata: dict[str, Any],
    symbol: str,
    timeframe: str,
) -> tuple[Path, Path]:
    CONFIG.ensure_dirs()
    ts = _timestamp()
    model_path = CONFIG.models_dir / f"{symbol}_{timeframe}_{ts}.pkl"
    meta_path = CONFIG.models_dir / f"{symbol}_{timeframe}_{ts}.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, meta_path


def latest_model(symbol: str, timeframe: str) -> Path:
    pattern = f"{symbol}_{timeframe}_*.pkl"
    candidates = sorted(CONFIG.models_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No model files for {symbol}/{timeframe}")
    return candidates[-1]

