"""Univariate feature analysis: default rate by bin, information value and risk direction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Conventional information value bands for credit scorecards
IV_BANDS = [(0.02, "not predictive"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong"), (np.inf, "very strong")]


def binned(x: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Default rate per quantile bin (per value for low-cardinality features)."""
    groups = x if x.nunique() <= bins else pd.qcut(x.rank(method="first"), bins, labels=False)
    t = pd.DataFrame({"bin": groups, "x": x, "y": y}).groupby("bin", observed=True)
    out = t.agg(x_min=("x", "min"), x_max=("x", "max"), clients=("y", "size"), defaults=("y", "sum")).reset_index()
    out["default_rate"] = out["defaults"] / out["clients"]
    return out


def information_value(table: pd.DataFrame) -> float:
    bad = (table["defaults"] + 0.5) / (table["defaults"].sum() + 0.5)
    good = (table["clients"] - table["defaults"] + 0.5) / ((table["clients"] - table["defaults"]).sum() + 0.5)
    return float(((good - bad) * np.log(good / bad)).sum())


def feature_report(X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        t = binned(X[col], y, bins)
        rho = spearmanr(t["x_min"], t["default_rate"]).statistic if len(t) > 2 else 0.0
        iv = information_value(t)
        rows.append(
            {
                "feature": col,
                "iv": iv,
                "strength": next(label for cut, label in IV_BANDS if iv < cut),
                "trend": "increasing" if rho > 0.8 else "decreasing" if rho < -0.8 else "non-monotonic",
                "min_bin_default_rate": float(t["default_rate"].min()),
                "max_bin_default_rate": float(t["default_rate"].max()),
                "zero_share": float((X[col] == 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)
