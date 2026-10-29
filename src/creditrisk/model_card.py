"""Auto-generated model card stored with every registered model version."""

from __future__ import annotations

import pandas as pd


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _f(v, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def build(summary: dict, m: dict, decision: dict, backtest: pd.DataFrame | None, importance: pd.DataFrame | None,
          fairness_summary: dict, model_name: str, version: str | None = None, git_sha: str = "unknown", data_version: str = "unknown") -> str:
    a = summary["assumptions"]
    lines = [
        f"# Model card - {model_name}{f' v{version}' if version else ''}",
        "",
        "## Purpose",
        "Probability of default (PD) for credit-card clients: the probability that a client misses the next monthly payment.",
        "Used to rank and grade the portfolio, steer limits, and trigger early collections or review. Not an automated decline engine:",
        "intervention decisions stay with credit management.",
        "",
        "## Training data and lineage",
        f"- Source: UCI *Default of Credit Card Clients* (Taiwan, Apr-Sep 2005), data asset version `{data_version}`",
        f"- Training rows: {summary['train_rows']:,} | default rate {_pct(summary['train_default_rate'])}",
        f"- Git commit: `{git_sha}` | trained {summary['trained_at']}",
        "- Libraries: " + ", ".join(f"{k} {v}" for k, v in summary["library_versions"].items()),
        "",
        "## Model",
        f"- Family: **{summary['model_family']}** (selected out-of-fold against the alternatives below)",
        f"- Monotonic constraints: {summary['monotone_constraints']} | calibration: {summary['calibration']}",
        f"- Protected attributes (SEX, MARRIAGE, AGE) excluded from features: {summary['exclude_sensitive']}",
        f"- Cut-off for intervention: PD >= {summary['threshold']:.3f} ({summary['threshold_rule']})",
        "",
        "| Candidate | CV AUC | CV Gini | Brier |",
        "|---|---|---|---|",
        *[f"| {c['candidate']} | {c['auc']:.4f} | {c['gini']:.4f} | {c['brier']:.4f} |" for c in summary["candidates"]],
        "",
        "## Test-set performance",
        f"- AUC {_f(m.get('auc'))} (95% CI {_f(m.get('auc_ci_low'))}-{_f(m.get('auc_ci_high'))}) | Gini {_f(m.get('gini'))} | KS {_f(m.get('ks'))}",
        f"- Brier {_f(m.get('brier'), 4)} | ECE {_f(m.get('ece'), 4)} | calibration slope {_f(m.get('calibration_slope'), 2)} | Hosmer-Lemeshow p {_f(m.get('hosmer_lemeshow_p'))}",
        f"- Mean PD {_pct(m.get('mean_pd'))} vs observed default rate {_pct(m.get('observed_dr'))}",
        f"- At the cut-off: recall {_pct(m.get('recall'))}, precision {_pct(m.get('precision'))}, flagged {_pct(m.get('flagged_rate'))},"
        f" expected cost {m.get('cost_saving_vs_no_model', 0):.0%} below the no-model baseline",
        "",
        "## PD backtest by grade (Jeffreys test)",
    ]
    if backtest is not None and not backtest.empty:
        lines += ["| Grade | PD range | Clients | Observed DR | Mean PD | Jeffreys p | Light |", "|---|---|---|---|---|---|---|"]
        lines += [
            f"| {r.grade} | {r.pd_range} | {r.n:,} | {r.observed_dr:.1%} | {r.mean_pd:.1%} | {r.jeffreys_p_underestimation:.3f} | {r.traffic_light} |"
            for r in backtest.itertuples()
        ]
    lines += ["", "## Main risk drivers (mean |SHAP|)"]
    if importance is not None:
        lines += [f"{i + 1}. {r.description} (`{r.feature}`)" for i, r in enumerate(importance.head(10).itertuples())]
    lines += [
        "",
        "## Fairness (attributes not used as features)",
        *[f"- {k.replace('_', ' ')}: {v:.3f}" for k, v in fairness_summary.items()],
        "",
        "## Business assumptions (to be replaced by internal loss data)",
        f"- LGD {a['lgd']:.0%}, CCF {a['ccf']:.0%}, cost of a missed default {a['cost_fn']:g} vs a false alarm {a['cost_fp']:g}",
        f"- IFRS 9 staging is illustrative: 30+ dpd or grade in {a['sicr_grades']} -> stage 2, 90+ dpd -> stage 3; lifetime {a['lifetime_years']:g} years",
        "",
        "## Registration decision",
        f"- Registered: **{decision.get('register')}**",
        *[f"- {k}: {v}" for k, v in decision.get("checks", {}).items()],
        "",
        "## Limitations",
        "- Cross-sectional data from 2005 (Taiwan, consumer cards); no out-of-time sample, so backtesting is on a held-out split.",
        "- Target is next-month payment default, not the regulatory 90-dpd definition; PDs need recalibration before IFRS 9/IRB use.",
        "- Exposure, LGD and cost figures are assumptions for illustration.",
        "- Monitor population stability (PSI) monthly; retrain when the score PSI exceeds 0.25 or a grade turns red.",
    ]
    return "\n".join(lines) + "\n"
