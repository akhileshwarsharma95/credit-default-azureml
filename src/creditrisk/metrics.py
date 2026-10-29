"""Credit-risk model validation metrics.

Discrimination (AUC, Gini, KS with bootstrap CIs), calibration (Brier, ECE,
Hosmer-Lemeshow, calibration slope/intercept), cut-off economics, gains tables,
PD backtesting per rating grade (Jeffreys test, traffic lights) and fairness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from .policy import MASTER_SCALE

_EPS = 1e-6


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def discrimination(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
    }


def bootstrap_ci(y, p, n_boot: int = 500, seed: int = 42, alpha: float = 0.05) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs, kss = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        kss.append(ks_statistic(y[idx], p[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    aucs = np.asarray(aucs)
    return {
        "auc_ci_low": float(np.percentile(aucs, lo)),
        "auc_ci_high": float(np.percentile(aucs, hi)),
        "gini_ci_low": float(2 * np.percentile(aucs, lo) - 1),
        "gini_ci_high": float(2 * np.percentile(aucs, hi) - 1),
        "ks_ci_low": float(np.percentile(kss, lo)),
        "ks_ci_high": float(np.percentile(kss, hi)),
    }


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    out = (
        df.groupby("bin")
        .agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"), observed_dr=("y", "mean"))
        .reset_index()
    )
    return out


def hosmer_lemeshow(y, p, n_bins: int = 10) -> tuple[float, float]:
    t = calibration_table(y, p, n_bins)
    expected = t["mean_pd"] * t["n"]
    observed = t["defaults"]
    denom = (expected * (1 - t["mean_pd"])).clip(lower=_EPS)
    stat = float((((observed - expected) ** 2) / denom).sum())
    return stat, float(stats.chi2.sf(stat, df=max(len(t) - 2, 1)))


def calibration(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), _EPS, 1 - _EPS)
    t = calibration_table(y, p, n_bins)
    ece = float((t["n"] * (t["mean_pd"] - t["observed_dr"]).abs()).sum() / t["n"].sum())
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    hl_stat, hl_p = hosmer_lemeshow(y, p, n_bins)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": ece,
        "mean_pd": float(p.mean()),
        "observed_dr": float(y.mean()),
        "calibration_slope": float(lr.coef_[0][0]),
        "calibration_intercept": float(lr.intercept_[0]),
        "hosmer_lemeshow_stat": hl_stat,
        "hosmer_lemeshow_p": hl_p,
    }


def threshold_metrics(y, p, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    flag = p >= threshold
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    n = len(y)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    cost = (cost_fn * fn + cost_fp * fp) / n
    baseline = cost_fn * y.mean()  # no model: nobody is flagged, every default is a loss
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, _EPS)),
        "flagged_rate": float(flag.mean()),
        "expected_cost_per_client": float(cost),
        "baseline_cost_per_client": float(baseline),
        "cost_saving_vs_no_model": float(1 - cost / baseline) if baseline > 0 else 0.0,
    }


def cost_curve(y, p, cost_fn: float, cost_fp: float, grid: np.ndarray | None = None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2) if grid is None else grid
    rows = [threshold_metrics(y, p, t, cost_fn, cost_fp) for t in grid]
    return pd.DataFrame(rows)[["threshold", "expected_cost_per_client", "flagged_rate", "recall", "precision"]]


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = n - pd.qcut(df["p"].rank(method="first"), q=n, labels=False)  # 1 = riskiest
    t = (
        df.groupby("decile")
        .agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"), min_pd=("p", "min"))
        .reset_index()
    )
    t["observed_dr"] = t["defaults"] / t["n"]
    t["cum_default_capture"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["cum_population"] = t["n"].cumsum() / t["n"].sum()
    t["lift"] = t["observed_dr"] / df["y"].mean()
    return t


def jeffreys_pvalue(defaults: int, n: int, pd_estimate: float) -> float:
    """One-sided Jeffreys test (ECB/TRIM): small p => PD significantly underestimates risk."""
    return float(stats.beta.cdf(pd_estimate, defaults + 0.5, n - defaults + 0.5))


def traffic_light(p_value: float) -> str:
    if p_value < 0.001:
        return "red"
    if p_value < 0.05:
        return "amber"
    return "green"


MIN_OBLIGORS = 30


def backtest_by_grade(y, p, grades) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "grade": np.asarray(grades)})
    rows = []
    for grade, lo, hi in MASTER_SCALE:
        g = df[df["grade"] == grade]
        if g.empty:
            continue
        n, d, mean_pd = len(g), int(g["y"].sum()), float(g["p"].mean())
        p_under = jeffreys_pvalue(d, n, mean_pd)
        p_over = 1 - p_under
        rows.append(
            {
                "grade": grade,
                "pd_range": f"{lo:.0%}-{min(hi, 1):.0%}",
                "n": n,
                "defaults": d,
                "observed_dr": d / n,
                "mean_pd": mean_pd,
                "jeffreys_p_underestimation": p_under,
                "traffic_light": traffic_light(p_under) if n >= MIN_OBLIGORS else "insufficient",
                "conservative": bool(p_over < 0.05),
            }
        )
    return pd.DataFrame(rows)


def fairness_table(y, p, threshold: float, groups: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Outcome and error-rate parity by protected attribute (attributes may be excluded from the model)."""
    y, p = np.asarray(y), np.asarray(p)
    flag = p >= threshold
    rows, summary = [], {}
    for attr in groups.columns:
        g = groups[attr].astype(str).to_numpy()
        attr_rows = []
        for val in sorted(set(g)):
            m = g == val
            if m.sum() < 30:
                continue
            yy, pp, ff = y[m], p[m], flag[m]
            attr_rows.append(
                {
                    "attribute": attr,
                    "group": val,
                    "n": int(m.sum()),
                    "observed_dr": float(yy.mean()),
                    "mean_pd": float(pp.mean()),
                    "flagged_rate": float(ff.mean()),
                    "tpr": float(ff[yy == 1].mean()) if (yy == 1).any() else np.nan,
                    "fpr": float(ff[yy == 0].mean()) if (yy == 0).any() else np.nan,
                    "auc": float(roc_auc_score(yy, pp)) if 0 < yy.mean() < 1 else np.nan,
                }
            )
        if attr_rows:
            t = pd.DataFrame(attr_rows)
            approval = 1 - t["flagged_rate"]
            summary[f"{attr}_disparate_impact_ratio"] = float(approval.min() / max(approval.max(), _EPS))
            summary[f"{attr}_equal_opportunity_gap"] = float(t["tpr"].max() - t["tpr"].min())
            summary[f"{attr}_calibration_gap"] = float((t["mean_pd"] - t["observed_dr"]).abs().max())
            rows += attr_rows
    return pd.DataFrame(rows), summary
