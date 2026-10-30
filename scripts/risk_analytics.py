"""Feature analysis and portfolio risk analytics for the deck and the data report.

    uv run python scripts/risk_analytics.py      # writes docs/analytics/analytics.json and feature_report.md

Feature analysis runs on the full cleaned UCI data (SEX, MARRIAGE and AGE are analysed, never model
inputs). Portfolio figures use the held-out test portfolio scored by the registered model package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from creditrisk import feature_report, insights, policy, quality
from creditrisk.config import load
from creditrisk.features import FeatureEngineer
from creditrisk.observability import get_logger
from creditrisk.schema import TARGET, describe, load_raw
from creditrisk.scoring import load_package

LOGGER = get_logger("risk_analytics")
ROOT = Path(__file__).resolve().parents[1]
TOP_FEATURES = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--model_dir", default=str(ROOT / "app" / "model"))
    p.add_argument("--out", default=str(ROOT / "docs" / "analytics"))
    return p.parse_args()


def feature_analysis(raw_path: str) -> tuple[list[dict], dict]:
    df, report = quality.validate(load_raw(raw_path))
    df = quality.clean(df, report)
    X = FeatureEngineer().fit_transform(df.drop(columns=[TARGET, "ID"], errors="ignore"))
    y = df[TARGET].astype(int)
    rep = feature_report.feature_report(X, y)
    rep["description"] = rep["feature"].map(describe)
    bins = {f: feature_report.binned(X[f], y).to_dict(orient="records") for f in rep["feature"].head(TOP_FEATURES)}
    return rep.to_dict(orient="records"), bins


def portfolio_analysis(model_dir: str) -> dict:
    pkg = load_package(model_dir)
    scored = pkg.read_csv("evaluation/test_scored.csv")
    kpis, tables = insights.portfolio_insights(scored)
    business = load()["business"]
    ttc = policy.ttc_shift(scored["pd"], business["ttc"]["target_pd"])
    by_grade = (
        scored.assign(pd_ttc=ttc)
        .groupby("grade")
        .agg(clients=("pd", "size"), pd_pit=("pd", "mean"), pd_ttc=("pd_ttc", "mean"), observed_dr=("default", "mean"))
        .reset_index()
    )
    pricing = policy.risk_based_price(by_grade["pd_pit"], business["loss"]["lgd"]).assign(grade=by_grade["grade"])
    return {
        "kpis": kpis,
        "lifetime_pd": tables["lifetime_pd"].to_dict(orient="records"),
        "default_forecast": tables["default_forecast"].to_dict(orient="records"),
        "loss_quantiles": tables["loss_quantiles"].to_dict(orient="records"),
        "roll_rate_matrix": tables["roll_rate_matrix"].to_dict(orient="records"),
        "ttc_by_grade": by_grade.to_dict(orient="records"),
        "ttc_target_pd": business["ttc"]["target_pd"],
        "pricing_by_grade": pricing.to_dict(orient="records"),
        "actions": scored["action"].value_counts().to_dict(),
        "escalation": business["escalation"],
        "asset_correlation": business["portfolio"]["asset_correlation"],
    }


def markdown(features: list[dict]) -> str:
    lines = [
        "# Feature analysis (UCI credit card clients, 30,000 rows)",
        "",
        "Information value (IV) bands: < 0.02 not predictive, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5 very strong.",
        "",
        "| Feature | Description | IV | Strength | Risk direction | Default rate, lowest -> highest bin |",
        "|---|---|---|---|---|---|",
    ]
    for f in features:
        lines.append(
            f"| `{f['feature']}` | {f['description']} | {f['iv']:.3f} | {f['strength']} | {f['trend']} | "
            f"{f['min_bin_default_rate']:.1%} -> {f['max_bin_default_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    features, bins = feature_analysis(args.raw_data)
    data = {"features": features, "feature_bins": bins, **portfolio_analysis(args.model_dir)}
    (out / "analytics.json").write_text(
        json.dumps(data, indent=1, default=lambda v: float(v) if isinstance(v, np.floating) else str(v))
    )
    (out / "feature_report.md").write_text(markdown(features))
    LOGGER.info(f"Wrote {out / 'analytics.json'} and {out / 'feature_report.md'} ({len(features)} features)")


if __name__ == "__main__":
    main()
