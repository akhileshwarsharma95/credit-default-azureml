"""Lifetime PDs from delinquency migration (cohort method, homogeneous Markov chain).

The mean monthly migration matrix between delinquency buckets is raised to the t-th power;
its default column is the cumulative PD after t months and the month-on-month difference is
the marginal PD vector an impairment engine consumes. Default (90+ dpd) is absorbing.
Preconditions: stable migration rates over time and a portfolio of stable size and mix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_STATE = "3 90+ dpd"


def lifetime_pd(matrix: pd.DataFrame, months: int = 12, default_state: str = DEFAULT_STATE) -> pd.DataFrame:
    m = matrix.to_numpy(dtype=float).copy()
    d = list(matrix.columns).index(default_state)
    m[d] = np.eye(len(m))[d]  # default is absorbing
    rows, power, previous = [], np.eye(len(m)), np.zeros(len(m))
    for month in range(1, months + 1):
        power = power @ m
        cumulative = power[:, d]
        rows += [
            {"from_bucket": b, "month": month, "cumulative_pd": float(c), "marginal_pd": float(c - p)}
            for b, c, p in zip(matrix.index, cumulative, previous, strict=True)
            if b != default_state
        ]
        previous = cumulative
    return pd.DataFrame(rows)
