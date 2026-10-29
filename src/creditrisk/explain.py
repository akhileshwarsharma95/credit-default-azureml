"""Explainability: exact per-client SHAP contributions and reason codes.

* LightGBM: native TreeSHAP via ``pred_contrib=True`` (exact, fast, no extra
  dependency at serving time).
* Logistic regression: exact linear SHAP, coef * (x - background mean),
  aggregated back from one-hot columns to source features.

Contributions are in log-odds of the base model. Calibration is monotone, so the
ranking of drivers (what reason codes need) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from .modeling import unwrap_base
from .schema import EDUCATION_LABELS, MARRIAGE_LABELS, MONTHS, PAY_STATUS_LABELS, SEX_LABELS, describe


def feature_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    """Raw + engineered features exactly as the model sees them."""
    base = unwrap_base(model)
    fe = base.named_steps["features"]
    return fe.transform(X[list(fe.feature_names_in_)])


def _source_of(name: str, sources: list[str]) -> str:
    s = name.split("__", 1)[-1]
    if s in sources:
        return s
    matches = [c for c in sources if s.startswith(c + "_")]
    return max(matches, key=len) if matches else s


def contributions(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (contributions per feature in log-odds, base value per row)."""
    base = unwrap_base(model)
    est = base.named_steps["model"]
    Xf = feature_frame(model, X)

    if isinstance(est, LGBMClassifier):
        raw = est.predict(Xf, pred_contrib=True)
        return pd.DataFrame(raw[:, :-1], columns=Xf.columns, index=X.index), raw[:, -1]

    if isinstance(est, LogisticRegression):
        if background is None:
            raise ValueError("Linear explanations need a background sample")
        prep = base.named_steps["prep"]
        Z = prep.transform(Xf)
        Zb = prep.transform(feature_frame(model, background))
        mu = Zb.mean(axis=0)
        coef = est.coef_[0]
        names = list(prep.get_feature_names_out())
        mapping = [_source_of(n, list(Xf.columns)) for n in names]
        per_col = pd.DataFrame((Z - mu) * coef, columns=names, index=X.index)
        agg = per_col.T.groupby(mapping).sum().T.reindex(columns=Xf.columns, fill_value=0.0)
        base_value = np.full(len(X), est.intercept_[0] + float(mu @ coef))
        return agg, base_value

    raise TypeError(f"Unsupported estimator for explanations: {type(est).__name__}")


def _fmt_amount(v) -> str:
    return f"{float(v):,.0f}"


def reason_text(feature: str, value, medians: dict | None = None) -> str:
    """Plain-language reason for a risk-increasing feature value."""
    v = value
    if feature.startswith("PAY_") and feature[4:].isdigit():
        month = MONTHS[int(feature[4:]) - 1]
        s = int(v)
        if s >= 1:
            return f"Payment {s} month{'s' if s > 1 else ''} late in {month}"
        if s == 0:
            return f"Only the minimum was paid in {month} (revolving balance)"
        if s == -2:
            return f"No card usage in {month}"
        return f"Repayment status in {month}: {PAY_STATUS_LABELS.get(s, s)}"
    if feature.startswith("PAY_AMT"):
        return f"Low payment in {MONTHS[int(feature[7:]) - 1]} ({_fmt_amount(v)})"
    if feature.startswith("BILL_AMT"):
        month = MONTHS[int(feature[8:]) - 1]
        if medians and feature in medians:
            rel = "above" if float(v) > medians[feature] else "below"
            return f"Bill amount in {month} ({_fmt_amount(v)}) {rel} typical level"
        return f"Bill amount in {month}: {_fmt_amount(v)}"
    templates = {
        "LIMIT_BAL": lambda: f"Low credit limit ({_fmt_amount(v)})",
        "util_1": lambda: f"High credit utilisation ({float(v):.0%} of limit)",
        "util_avg": lambda: f"High average utilisation ({float(v):.0%} of limit)",
        "util_max": lambda: f"Peak utilisation of {float(v):.0%} of limit",
        "bill_trend": lambda: "Balance has been growing over the last 6 months",
        "pay_ratio_1": lambda: f"Latest repayment covered only {float(v):.0%} of the previous bill",
        "pay_ratio_avg": lambda: f"Repays only {float(v):.0%} of billed amounts on average",
        "n_delays": lambda: f"{int(v)} late payment{'s' if int(v) != 1 else ''} in the last 6 months",
        "n_delays_recent3": lambda: f"{int(v)} late payment{'s' if int(v) != 1 else ''} in the last 3 months",
        "max_delay": lambda: f"Longest payment delay: {int(v)} month{'s' if int(v) != 1 else ''}",
        "delay_trend": lambda: "Delinquency is worsening compared with 3 months ago",
        "n_no_consumption": lambda: f"Card unused in {int(v)} of the last 6 months",
        "n_paid_full": lambda: "Rarely pays the full balance",
        "months_since_delay": lambda: (
            f"Most recent late payment {int(v)} month{'s' if int(v) != 1 else ''} ago"
            if int(v) <= 6
            else "No recent late payment"
        ),
        "EDUCATION": lambda: f"Education: {EDUCATION_LABELS.get(int(v), v)}",
        "MARRIAGE": lambda: f"Marital status: {MARRIAGE_LABELS.get(int(v), v)}",
        "SEX": lambda: f"Sex: {SEX_LABELS.get(int(v), v)}",
        "AGE": lambda: f"Age: {int(v)}",
    }
    if feature in templates:
        return templates[feature]()
    return f"{describe(feature)}: {v}"


def reason_codes(
    contrib: pd.DataFrame,
    features: pd.DataFrame,
    top_n: int = 4,
    medians: dict | None = None,
    min_contribution: float = 0.02,
) -> list[list[dict]]:
    """Top risk-increasing drivers per row (adverse-action style reason codes)."""
    values = contrib.to_numpy()
    cols = np.asarray(contrib.columns)
    order = np.argsort(-values, axis=1)[:, :top_n]
    out = []
    for i in range(values.shape[0]):
        row = []
        for j in order[i]:
            c = float(values[i, j])
            if c < min_contribution:
                break
            f = cols[j]
            row.append(
                {
                    "code": f"RC_{f.upper()}",
                    "feature": f,
                    "description": describe(f),
                    "text": reason_text(f, features.iloc[i][f], medians),
                    "contribution": round(c, 4),
                }
            )
        out.append(row)
    return out


def mitigating_factors(contrib: pd.DataFrame, features: pd.DataFrame, top_n: int = 2) -> list[list[dict]]:
    values = contrib.to_numpy()
    cols = np.asarray(contrib.columns)
    order = np.argsort(values, axis=1)[:, :top_n]
    return [
        [
            {
                "feature": cols[j],
                "description": describe(cols[j]),
                "value": features.iloc[i][cols[j]],
                "contribution": round(float(values[i, j]), 4),
            }
            for j in order[i]
            if values[i, j] < 0
        ]
        for i in range(values.shape[0])
    ]


def global_importance(contrib: pd.DataFrame) -> pd.DataFrame:
    imp = contrib.abs().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {"feature": imp.index, "description": [describe(f) for f in imp.index], "mean_abs_shap": imp.values}
    )
