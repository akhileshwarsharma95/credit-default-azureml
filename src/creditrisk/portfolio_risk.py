"""Portfolio credit risk beyond the single client: loss distribution, stress and concentration.

Single-factor (Vasicek / Basel ASRF) model: all clients depend on one systematic factor, so
defaults cluster in a downturn. The loss at confidence q is the sum of EAD x LGD x the PD
conditional on a 1-in-(1/(1-q)) factor; unexpected loss = that loss minus expected loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def conditional_pd(pd_values, rho: float, q: float) -> np.ndarray:
    p = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
    return norm.cdf((norm.ppf(p) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho))


def credit_var(ead, lgd, pd_values, rho: float, q: float) -> float:
    return float((np.asarray(ead) * np.asarray(lgd) * conditional_pd(pd_values, rho, q)).sum())


def loss_quantiles(ead, lgd, pd_values, rho: float, levels=(0.5, 0.9, 0.99, 0.999)) -> pd.DataFrame:
    el = float((np.asarray(ead) * np.asarray(lgd) * np.asarray(pd_values)).sum())
    rows = [{"confidence": "expected", "loss": el, "unexpected_loss": 0.0}]
    for q in levels:
        loss = credit_var(ead, lgd, pd_values, rho, q)
        rows.append({"confidence": f"{q:.1%}", "loss": loss, "unexpected_loss": loss - el})
    return pd.DataFrame(rows)


def concentration(ead: pd.Series, by: pd.Series) -> dict:
    """Herfindahl index of single-name EAD (1/n = fully granular) and the share of the largest exposures."""
    share = ead / ead.sum()
    top = ead.sort_values(ascending=False)
    return {
        "hhi_single_name": float((share**2).sum()),
        "top10_ead_share": float(top.head(10).sum() / ead.sum()),
        "top1pct_ead_share": float(top.head(max(1, len(ead) // 100)).sum() / ead.sum()),
        "largest_group_ead_share": float(ead.groupby(by).sum().max() / ead.sum()),
    }
