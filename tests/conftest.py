"""Shared fixtures: a synthetic dataset in the official UCI layout and an
isolated MLflow tracking store/registry (SQLite) per test session."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Same schema and quirks as the UCI file (undocumented codes, duplicates), with real signal."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    limit = (np.exp(rng.normal(11.5, 0.7, n)) // 10_000 * 10_000).clip(10_000, 800_000).astype(int)
    pay = np.column_stack([np.clip(np.round(risk * 1.1 + rng.normal(0, 0.8, n) - 0.6), -2, 8) for _ in range(6)]).astype(int)
    util = np.clip(rng.beta(2, 3, n) + 0.15 * risk, -0.05, 1.3)
    bills = np.round(limit[:, None] * np.clip(util[:, None] + rng.normal(0, 0.05, (n, 6)), -0.05, 1.5)).astype(int)
    paid = np.round(np.clip(bills, 0, None) * np.clip(rng.beta(2, 5, (n, 6)) - 0.05 * risk[:, None], 0, 1)).astype(int)
    logit = -1.4 + 1.0 * risk + 0.4 * (pay[:, 0] > 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": limit,
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.01, 0.35, 0.45, 0.15, 0.02, 0.01, 0.01]),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n, p=[0.01, 0.45, 0.52, 0.02]),
            "AGE": rng.integers(21, 70, n),
            **{("PAY_0" if j == 0 else f"PAY_{j + 1}"): pay[:, j] for j in range(6)},
            **{f"BILL_AMT{j + 1}": bills[:, j] for j in range(6)},
            **{f"PAY_AMT{j + 1}": paid[:, j] for j in range(6)},
            "default payment next month": y,
        }
    )
    dup = df.iloc[:5].copy()
    dup["ID"] = np.arange(n + 1, n + 6)
    return pd.concat([df, dup], ignore_index=True)


def write_uci_xlsx(df: pd.DataFrame, path: Path) -> Path:
    top = [""] + [f"X{i}" for i in range(1, 24)] + ["Y"]
    pd.DataFrame([top, list(df.columns)] + df.values.tolist()).to_excel(path, index=False, header=False)
    return path


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("raw")
    return write_uci_xlsx(make_synthetic(), d / "default_of_credit_card_clients.xlsx")


@pytest.fixture(scope="session")
def mlflow_store(tmp_path_factory):
    import mlflow

    d = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{d / 'mlflow.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    mlflow.set_tracking_uri(uri)
    exp_id = mlflow.create_experiment("tests", artifact_location=(d / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return uri
