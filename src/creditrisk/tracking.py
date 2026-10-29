"""Thin MLflow helpers that behave the same locally and inside Azure ML jobs
(where the tracking URI and the active run are injected by the platform)."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import mlflow
import pandas as pd


def _clean(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out[f"{prefix}{k}"] = float(v)
    return out


def log_metrics(d: dict, prefix: str = "") -> None:
    m = _clean(d, prefix)
    if m:
        mlflow.log_metrics(m)


def log_params(d: dict) -> None:
    mlflow.log_params({k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else v) for k, v in d.items()})


def log_table(df: pd.DataFrame, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / Path(artifact_path).name
        df.to_csv(p, index=False)
        mlflow.log_artifact(str(p), artifact_path=str(Path(artifact_path).parent) if "/" in artifact_path else None)


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path
