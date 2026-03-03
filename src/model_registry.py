from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .config import CONFIG


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _hash_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _models_root() -> Path:
    path = CONFIG.reports_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path() -> Path:
    return _models_root() / "index.json"


def _load_index() -> dict[str, Any]:
    path = _index_path()
    if not path.exists():
        return {"models": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            return data
    except Exception:
        pass
    return {"models": []}


def _save_index(data: dict[str, Any]) -> None:
    _index_path().write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


@dataclass
class RegisteredModel:
    symbol: str
    timeframe: str
    version: str
    trained_at: str
    model_path: Path
    features_schema_path: Path
    train_meta_path: Path
    metrics_oof_path: Path
    record: dict[str, Any]


def register_model(
    symbol: str,
    timeframe: str,
    artifacts_paths: dict[str, Path],
    meta: dict[str, Any],
) -> dict[str, Any]:
    idx = _load_index()
    records = [r for r in idx["models"] if not (r.get("symbol") == symbol and r.get("timeframe") == timeframe and r.get("version") == meta.get("version"))]
    record = {
        "symbol": symbol,
        "timeframe": timeframe,
        "version": meta.get("version"),
        "trained_at": meta.get("trained_at"),
        "model_path": str(artifacts_paths["model_path"]),
        "features_schema_path": str(artifacts_paths["features_schema_path"]),
        "train_meta_path": str(artifacts_paths["train_meta_path"]),
        "metrics_oof_path": str(artifacts_paths["metrics_oof_path"]),
        "calibration_path": str(artifacts_paths["calibration_path"]) if artifacts_paths.get("calibration_path") else None,
        "metrics_summary": meta.get("metrics_summary", {}),
        "dataset_range": meta.get("dataset_range", {}),
        "seed": meta.get("seed"),
        "params_hash": meta.get("params_hash"),
        "features_hash": meta.get("features_hash"),
        "features_count": meta.get("features_count"),
    }
    records.append(record)
    records.sort(key=lambda x: str(x.get("trained_at", "")))
    idx["models"] = records
    _save_index(idx)
    return record


def list_models() -> list[dict[str, Any]]:
    idx = _load_index()
    out = list(idx["models"])
    # Include legacy models if not yet registered.
    for pkl in sorted(CONFIG.models_dir.glob("*_*.pkl")):
        parts = pkl.stem.split("_")
        if len(parts) < 4:
            continue
        symbol = parts[0]
        timeframe = parts[1]
        version = "_".join(parts[2:])
        if any(
            r.get("symbol") == symbol and r.get("timeframe") == timeframe and str(r.get("version")) == version
            for r in out
        ):
            continue
        meta_path = pkl.with_suffix(".json")
        trained_at = datetime.fromtimestamp(pkl.stat().st_mtime, tz=timezone.utc).isoformat()
        metrics_summary: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                mm = meta.get("metrics", {})
                metrics_summary = {
                    "class_accuracy": mm.get("class_accuracy"),
                    "brier_score": mm.get("brier_score"),
                }
            except Exception:
                metrics_summary = {}
        out.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "version": version,
                "trained_at": trained_at,
                "model_path": str(pkl),
                "features_schema_path": "",
                "train_meta_path": str(meta_path) if meta_path.exists() else "",
                "metrics_oof_path": str(meta_path) if meta_path.exists() else "",
                "metrics_summary": metrics_summary,
            }
        )
    out = sorted(out, key=lambda x: str(x.get("trained_at", "")), reverse=True)
    return out


def get_latest_model(symbol: str, timeframe: str) -> RegisteredModel:
    matches = [r for r in list_models() if r.get("symbol") == symbol and r.get("timeframe") == timeframe]
    if not matches:
        # Legacy fallback in CONFIG.models_dir
        pattern = f"{symbol}_{timeframe}_*.pkl"
        candidates = sorted(CONFIG.models_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No model files for {symbol}/{timeframe}")
        model_path = candidates[-1]
        meta_path = model_path.with_suffix(".json")
        version = model_path.stem.split("_")[-2] + "_" + model_path.stem.split("_")[-1] if "_" in model_path.stem else _timestamp()
        record = {
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "trained_at": _utc_now().isoformat(),
            "model_path": str(model_path),
            "features_schema_path": "",
            "train_meta_path": str(meta_path) if meta_path.exists() else "",
            "metrics_oof_path": str(meta_path) if meta_path.exists() else "",
        }
        return RegisteredModel(
            symbol=symbol,
            timeframe=timeframe,
            version=version,
            trained_at=record["trained_at"],
            model_path=model_path,
            features_schema_path=Path(record["features_schema_path"]) if record["features_schema_path"] else Path(""),
            train_meta_path=Path(record["train_meta_path"]) if record["train_meta_path"] else Path(""),
            metrics_oof_path=Path(record["metrics_oof_path"]) if record["metrics_oof_path"] else Path(""),
            record=record,
        )
    rec = matches[0]
    return RegisteredModel(
        symbol=symbol,
        timeframe=timeframe,
        version=str(rec["version"]),
        trained_at=str(rec["trained_at"]),
        model_path=Path(rec["model_path"]),
        features_schema_path=Path(rec["features_schema_path"]) if rec.get("features_schema_path") else Path(""),
        train_meta_path=Path(rec["train_meta_path"]) if rec.get("train_meta_path") else Path(""),
        metrics_oof_path=Path(rec["metrics_oof_path"]) if rec.get("metrics_oof_path") else Path(""),
        record=rec,
    )


def get_model(symbol: str, timeframe: str, version: str | None = None) -> RegisteredModel:
    if not version:
        return get_latest_model(symbol=symbol, timeframe=timeframe)
    matches = [
        r for r in list_models() if r.get("symbol") == symbol and r.get("timeframe") == timeframe and str(r.get("version")) == str(version)
    ]
    if not matches:
        raise FileNotFoundError(f"No model for {symbol}/{timeframe} version={version}")
    rec = matches[0]
    return RegisteredModel(
        symbol=symbol,
        timeframe=timeframe,
        version=str(rec["version"]),
        trained_at=str(rec["trained_at"]),
        model_path=Path(rec["model_path"]),
        features_schema_path=Path(rec["features_schema_path"]) if rec.get("features_schema_path") else Path(""),
        train_meta_path=Path(rec["train_meta_path"]) if rec.get("train_meta_path") else Path(""),
        metrics_oof_path=Path(rec["metrics_oof_path"]) if rec.get("metrics_oof_path") else Path(""),
        record=rec,
    )


def save_model_bundle(
    model: Any,
    metadata: dict[str, Any],
    symbol: str,
    timeframe: str,
) -> tuple[Path, Path]:
    CONFIG.ensure_dirs()
    version = _timestamp()
    trained_at = _utc_now().isoformat()
    root = _models_root() / symbol / timeframe / version
    root.mkdir(parents=True, exist_ok=True)

    model_path = root / "model.bin"
    features_schema_path = root / "features_schema.json"
    train_meta_path = root / "train_meta.json"
    metrics_oof_path = root / "metrics_oof.json"
    calibration_path: Path | None = None

    joblib.dump(model, model_path)

    features = metadata.get("features", [])
    features_schema = {"symbol": symbol, "timeframe": timeframe, "features": features}
    features_schema_path.write_text(json.dumps(features_schema, indent=2), encoding="utf-8")

    metrics = metadata.get("metrics", {})
    metrics_oof_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dataset_range = metadata.get("dataset_range", {})
    params = metadata.get("params", {})
    train_meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "version": version,
        "trained_at": trained_at,
        "seed": metadata.get("seed", 42),
        "dataset_range": dataset_range,
        "params": params,
        "params_hash": _hash_json(params),
        "features_hash": _hash_json(features),
        "features_count": len(features),
    }
    train_meta_path.write_text(json.dumps(train_meta, indent=2), encoding="utf-8")

    register_model(
        symbol=symbol,
        timeframe=timeframe,
        artifacts_paths={
            "model_path": model_path,
            "features_schema_path": features_schema_path,
            "train_meta_path": train_meta_path,
            "metrics_oof_path": metrics_oof_path,
            "calibration_path": calibration_path,
        },
        meta={
            **train_meta,
            "metrics_summary": {
                "class_accuracy": metrics.get("class_accuracy"),
                "brier_score": metrics.get("brier_score"),
            },
        },
    )

    # Legacy copies for compatibility with old loaders.
    legacy_model = CONFIG.models_dir / f"{symbol}_{timeframe}_{version}.pkl"
    legacy_meta = CONFIG.models_dir / f"{symbol}_{timeframe}_{version}.json"
    joblib.dump(model, legacy_model)
    legacy_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, train_meta_path


def load_model_object(symbol: str, timeframe: str, version: str | None = None) -> tuple[Any, RegisteredModel]:
    entry = get_model(symbol=symbol, timeframe=timeframe, version=version)
    model = joblib.load(entry.model_path)
    return model, entry
