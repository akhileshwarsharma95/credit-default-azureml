"""Business layer: rating master scale, cut-off economics, exposure & expected
loss, illustrative IFRS 9 staging, early-warning triggers and limit steering.

All monetary assumptions (LGD, CCF, cost ratio) are explicit parameters, meant
to be replaced by values from the internal loss database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import load

_BUSINESS = load()["business"]

# Rating master scale: grade, lower PD bound (inclusive), upper bound (exclusive).
# Calibrated to this portfolio's PD range so every grade holds a testable population.
MASTER_SCALE = [(g, float(lo), float(hi)) for g, lo, hi in _BUSINESS["master_scale"]]
PD_FLOOR = _BUSINESS["pd_floor"]
GRADES = [g for g, _, _ in MASTER_SCALE]


@dataclass
class Assumptions:
    lgd: float = _BUSINESS["loss"]["lgd"]
    ccf: float = _BUSINESS["loss"]["ccf"]
    cost_fn: float = _BUSINESS["costs"]["cost_fn"]
    cost_fp: float = _BUSINESS["costs"]["cost_fp"]
    lifetime_years: float = _BUSINESS["loss"]["lifetime_years"]
    sicr_grades: tuple = tuple(_BUSINESS["sicr_grades"])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sicr_grades"] = list(self.sicr_grades)
        return d


def cost_optimal_threshold(cost_fn: float, cost_fp: float) -> float:
    """With calibrated PDs, acting is optimal when PD * cost_fn > (1 - PD) * cost_fp."""
    return cost_fp / (cost_fp + cost_fn)


def assign_grade(pd_values) -> np.ndarray:
    pd_values = np.asarray(pd_values, dtype=float)
    edges = [lo for _, lo, _ in MASTER_SCALE] + [MASTER_SCALE[-1][2]]
    idx = np.clip(np.searchsorted(edges, pd_values, side="right") - 1, 0, len(GRADES) - 1)
    return np.asarray(GRADES)[idx]


def master_scale_frame() -> pd.DataFrame:
    return pd.DataFrame([{"grade": g, "pd_min": lo, "pd_max": min(hi, 1.0)} for g, lo, hi in MASTER_SCALE])


def exposure_at_default(df: pd.DataFrame, ccf: float) -> pd.Series:
    drawn = df["BILL_AMT1"].clip(lower=0)
    undrawn = (df["LIMIT_BAL"] - drawn).clip(lower=0)
    return drawn + ccf * undrawn


def ifrs9_stage(df: pd.DataFrame, grades: np.ndarray, sicr_grades=("F", "G")) -> np.ndarray:
    """Illustrative staging: 90+ days past due -> stage 3; 30+ dpd backstop or a
    SICR rating -> stage 2; else stage 1."""
    stage = np.ones(len(df), dtype=int)
    stage[np.isin(grades, list(sicr_grades)) | (df["PAY_1"].to_numpy() >= 1)] = 2
    stage[df["PAY_1"].to_numpy() >= 3] = 3
    return stage


def expected_credit_loss(pd_values, lgd: float, ead, stage, lifetime_years: float) -> np.ndarray:
    pd_values, ead, stage = np.asarray(pd_values), np.asarray(ead), np.asarray(stage)
    lifetime_pd = 1 - (1 - pd_values) ** lifetime_years
    horizon_pd = np.where(stage == 1, pd_values, np.where(stage == 2, lifetime_pd, 1.0))
    return horizon_pd * lgd * ead


EWS_TRIGGERS = {
    "ews_new_delinquency": "New delinquency this month",
    "ews_worsening_delinquency": "Delinquency rolling to a worse bucket",
    "ews_utilisation_spike": "Utilisation spike above 80% of limit",
    "ews_payment_shortfall": "Paid less than 10% of the previous bill",
    "ews_high_pd": "PD in grade F/G",
}


def early_warning(features: pd.DataFrame, grades: np.ndarray) -> pd.DataFrame:
    """Rule-based early-warning triggers on top of the PD (behavioural red flags)."""
    f = features
    prior_util = (f[["BILL_AMT2", "BILL_AMT3", "BILL_AMT4"]].mean(axis=1) / f["LIMIT_BAL"].clip(lower=1)).to_numpy()
    out = pd.DataFrame(index=features.index)
    out["ews_new_delinquency"] = (f["PAY_1"] >= 1) & (f["PAY_2"] <= 0)
    out["ews_worsening_delinquency"] = (f["PAY_1"] > f["PAY_2"]) & (f["PAY_2"] >= 1)
    out["ews_utilisation_spike"] = (f["util_1"] > 0.8) & (f["util_1"] - prior_util > 0.2)
    out["ews_payment_shortfall"] = (f["BILL_AMT2"] > 0) & (f["PAY_AMT1"] < 0.1 * f["BILL_AMT2"])
    out["ews_high_pd"] = np.isin(grades, ["F", "G"])
    out["ews_count"] = out[list(EWS_TRIGGERS)].sum(axis=1)
    out["ews_triggers"] = [
        "; ".join(EWS_TRIGGERS[k] for k in EWS_TRIGGERS if row[k]) for _, row in out[list(EWS_TRIGGERS)].iterrows()
    ]
    return out


def limit_recommendation(df: pd.DataFrame, pd_values, grades, lgd: float, ccf: float) -> pd.DataFrame:
    """Exposure steering: cap undrawn limits for high-risk clients, offer increases
    to low-risk clients who are constrained by their limit."""
    pd_values = np.asarray(pd_values)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)
    drawn = df["BILL_AMT1"].clip(lower=0).to_numpy(dtype=float)
    util = drawn / np.clip(limit, 1, None)
    action = np.full(len(df), "Keep", dtype=object)
    new_limit = limit.copy()

    reduce = np.isin(grades, ["F", "G"]) & (util < 0.8)
    new_limit[reduce] = np.maximum(np.ceil(drawn[reduce] * 1.1 / 1000) * 1000, 10_000)
    action[reduce & (new_limit < limit)] = "Reduce limit"
    new_limit[action != "Reduce limit"] = limit[action != "Reduce limit"]

    increase = np.isin(grades, ["A", "B"]) & (util > 0.8)
    new_limit[increase] = np.round(limit[increase] * 1.2, -3)
    action[increase] = "Increase limit"

    ead_old = drawn + ccf * np.clip(limit - drawn, 0, None)
    ead_new = drawn + ccf * np.clip(new_limit - drawn, 0, None)
    return pd.DataFrame(
        {
            "limit_action": action,
            "recommended_limit": new_limit,
            "ead_change": ead_new - ead_old,
            "el_change": pd_values * lgd * (ead_new - ead_old),
        },
        index=df.index,
    )


def action_label(pd_values, threshold: float, ews_count) -> np.ndarray:
    pd_values, ews_count = np.asarray(pd_values), np.asarray(ews_count)
    action = np.full(len(pd_values), "Standard", dtype=object)
    action[(pd_values >= threshold / 2) | (ews_count >= 2)] = "Watch"
    action[pd_values >= threshold] = "Intervene"
    return action
