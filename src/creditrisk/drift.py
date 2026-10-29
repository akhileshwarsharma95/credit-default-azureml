"""Population stability monitoring (PSI/CSI).

The training step stores a reference profile (bin edges + expected shares for
every feature and for the PD score) inside the model package, so any later
batch can be checked for drift without access to the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policy import GRADES, assign_grade

PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4


_DECIMALS = 10  # values are rounded before binning: CSV round-trips can flip the last bit of a value sitting on an edge


def _numeric_spec(x: np.ndarray, n_bins: int) -> dict:
    x = np.round(np.asarray(x, dtype=float), _DECIMALS)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = [-np.inf, *qs.tolist(), np.inf]
    counts = np.histogram(x, bins=edges)[0]
    return {"type": "numeric", "edges": [float(e) for e in edges], "expected": (counts / counts.sum()).tolist()}


def _categorical_spec(x: pd.Series) -> dict:
    shares = x.round(_DECIMALS).value_counts(normalize=True).sort_index()
    return {"type": "categorical", "categories": [float(c) for c in shares.index], "expected": shares.tolist() + [0.0]}


def build_profile(features: pd.DataFrame, scores, n_bins: int = 10, max_categories: int = 12) -> dict:
    profile = {"n_reference": int(len(features)), "features": {}, "medians": {}}
    for col in features.columns:
        x = features[col].astype(float)
        spec = _categorical_spec(x) if x.nunique() <= max_categories else _numeric_spec(x.to_numpy(), n_bins)
        spec.update({"mean": float(x.mean()), "std": float(x.std()), "p01": float(x.quantile(0.01)), "p99": float(x.quantile(0.99))})
        profile["features"][col] = spec
        profile["medians"][col] = float(x.median())
    profile["score"] = _numeric_spec(np.asarray(scores, dtype=float), n_bins)
    profile["score"]["mean"] = float(np.mean(scores))
    grades = pd.Series(assign_grade(scores)).value_counts(normalize=True)
    profile["grades"] = {g: float(grades.get(g, 0.0)) for g in GRADES}
    return profile


def _actual_shares(x: np.ndarray, spec: dict) -> np.ndarray:
    x = np.round(np.asarray(x, dtype=float), _DECIMALS)
    if spec["type"] == "numeric":
        counts = np.histogram(x, bins=spec["edges"])[0]
    else:
        cats = np.asarray(spec["categories"])
        counts = np.array([(x == c).sum() for c in cats] + [(~np.isin(x, cats)).sum()])
    return counts / max(counts.sum(), 1)


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float, warn: float = PSI_WARN, alert: float = PSI_ALERT) -> str:
    return "alert" if value >= alert else "warn" if value >= warn else "stable"


def drift_report(profile: dict, features: pd.DataFrame, scores, warn: float = PSI_WARN, alert: float = PSI_ALERT) -> tuple[pd.DataFrame, dict]:
    rows = []
    for col, spec in profile["features"].items():
        if col not in features.columns:
            continue
        x = features[col].astype(float).to_numpy()
        value = psi(spec["expected"], _actual_shares(x, spec))
        rows.append(
            {
                "feature": col,
                "psi": value,
                "status": status(value, warn, alert),
                "reference_mean": spec["mean"],
                "current_mean": float(np.mean(x)),
                "out_of_range_share": float(((x < spec["p01"]) | (x > spec["p99"])).mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    score_psi = psi(profile["score"]["expected"], _actual_shares(np.asarray(scores, dtype=float), profile["score"]))
    grade_psi = None
    if "grades" in profile:
        current = pd.Series(assign_grade(scores)).value_counts(normalize=True)
        grade_psi = psi([profile["grades"][g] for g in GRADES], [float(current.get(g, 0.0)) for g in GRADES])
    summary = {
        "score_psi": score_psi,
        "grade_psi": grade_psi,
        "score_status": status(score_psi, warn, alert),
        "max_feature_psi": float(table["psi"].max()) if len(table) else 0.0,
        "n_features_warn": int((table["status"] == "warn").sum()),
        "n_features_alert": int((table["status"] == "alert").sum()),
        "reference_mean_pd": profile["score"]["mean"],
        "current_mean_pd": float(np.mean(scores)),
    }
    summary["drift_alert"] = bool(summary["score_status"] == "alert" or summary["n_features_alert"] > 0)
    summary["overall_status"] = "alert" if summary["drift_alert"] else ("warn" if summary["score_status"] == "warn" or summary["n_features_warn"] else "stable")
    return table, summary


def simulate_downturn(df: pd.DataFrame, severity: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Synthetic stressed batch for monitoring demos: more delinquency, higher
    balances, lower repayments. Labels (if any) are dropped - they would be unknown."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    hit = rng.random(n) < 0.35 * severity
    for i, col in enumerate([f"PAY_{k}" for k in range(1, 7)]):
        bump = hit & (rng.random(n) < 0.8 - 0.1 * i)
        out.loc[bump, col] = np.clip(np.maximum(out.loc[bump, col], 0) + 1, -2, 8)
    for col in [f"BILL_AMT{k}" for k in range(1, 7)]:
        out[col] = (out[col] * (1 + 0.20 * severity)).round()
    for col in [f"PAY_AMT{k}" for k in range(1, 7)]:
        out[col] = (out[col] * (1 - 0.35 * severity)).clip(lower=0).round()
    out = out.drop(columns=[c for c in ("default",) if c in out.columns])
    return out
