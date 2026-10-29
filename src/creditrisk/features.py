"""Feature strategy: stateless behavioural features computed row by row.

They live inside the model pipeline (not in the preprocessing step), so the
registered model takes raw columns and training/serving skew is impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import BILL_AMT, PAY_AMT, PAY_STATUS

ENGINEERED = [
    "util_1",
    "util_avg",
    "util_max",
    "bill_trend",
    "pay_ratio_1",
    "pay_ratio_avg",
    "n_delays",
    "n_delays_recent3",
    "max_delay",
    "delay_trend",
    "n_no_consumption",
    "n_paid_full",
    "months_since_delay",
]

# Expected direction of each feature's effect on default risk (+1 up, -1 down).
# Used as LightGBM monotonic constraints: the model can never learn that paying
# later lowers risk, which keeps reason codes intuitive and defensible.
MONOTONE_DIRECTION = {
    "LIMIT_BAL": -1,
    **{c: 1 for c in PAY_STATUS},
    "n_delays": 1,
    "n_delays_recent3": 1,
    "max_delay": 1,
    "pay_ratio_avg": -1,
    "months_since_delay": -1,
}


def signed_log1p(x):
    """Log-compress heavy-tailed monetary amounts while keeping the sign."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds utilisation, repayment-ratio and delinquency-pattern features."""

    def fit(self, X: pd.DataFrame, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=cols)

    def transform(self, X) -> pd.DataFrame:
        X = self._as_frame(X).copy()
        limit = X["LIMIT_BAL"].clip(lower=1).astype(float)
        bills = X[BILL_AMT].astype(float)
        pays = X[PAY_AMT].astype(float)
        status = X[PAY_STATUS].astype(float)

        X["util_1"] = (bills["BILL_AMT1"] / limit).clip(-1, 5)
        X["util_avg"] = (bills.mean(axis=1) / limit).clip(-1, 5)
        X["util_max"] = (bills.max(axis=1) / limit).clip(-1, 5)
        X["bill_trend"] = ((bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit).clip(-5, 5)
        # PAY_AMT1 (paid in Sep) settles the Aug statement (BILL_AMT2), and so on.
        X["pay_ratio_1"] = (pays["PAY_AMT1"] / bills["BILL_AMT2"].clip(lower=1)).clip(0, 5)
        X["pay_ratio_avg"] = (
            pays[PAY_AMT[:5]].sum(axis=1).to_numpy()
            / bills[BILL_AMT[1:]].clip(lower=0).sum(axis=1).clip(lower=1).to_numpy()
        ).clip(0, 5)
        late = status >= 1
        X["n_delays"] = late.sum(axis=1)
        X["n_delays_recent3"] = late[PAY_STATUS[:3]].sum(axis=1)
        X["max_delay"] = status.max(axis=1).clip(lower=0)
        X["delay_trend"] = status["PAY_1"].clip(lower=0) - status["PAY_4"].clip(lower=0)
        X["n_no_consumption"] = (status == -2).sum(axis=1)
        X["n_paid_full"] = (status == -1).sum(axis=1)
        first_late = late.to_numpy().argmax(axis=1) + 1  # 1 = latest month
        X["months_since_delay"] = np.where(late.any(axis=1), first_late, 7)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + ENGINEERED, dtype=object)
