"""Monitoring - population stability (PSI per feature and for the PD score)
against the reference profile stored in the model package, data-quality
signals and, when outcomes are available, realised performance and backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from creditrisk import drift, metrics, plots
from creditrisk.config import load
from creditrisk.schema import TARGET, read_table
from creditrisk.scoring import load_package
from creditrisk.tracking import log_metrics, write_json

MONITORING = load()["business"]["monitoring"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_input", required=True)
    p.add_argument("--scored_data", required=True)
    p.add_argument("--monitoring_output", required=True)
    p.add_argument("--psi_warn", type=float, default=drift.PSI_WARN)
    p.add_argument("--psi_alert", type=float, default=drift.PSI_ALERT)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.monitoring_output)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    with mlflow.start_run():
        mlflow.set_tags({"stage": "monitor"})
        pkg = load_package(args.model_input)
        scored = read_table(args.scored_data, "scored.csv")
        feats = scored[[c for c in pkg.profile["features"] if c in scored.columns]]
        table, summary = drift.drift_report(pkg.profile, feats, scored["pd"], args.psi_warn, args.psi_alert)

        labelled = TARGET in scored.columns and scored[TARGET].notna().all()
        if labelled:
            y, p = scored[TARGET].astype(int), scored["pd"]
            perf = metrics.discrimination(y, p) | metrics.calibration(y, p)
            bt = metrics.backtest_by_grade(y, p, scored["grade"])
            bt.to_csv(out / "backtest.csv", index=False)
            summary["performance"] = perf
            summary["backtest_red_grades"] = int((bt["traffic_light"] == "red").sum())
            validation_auc = pkg.read_json("evaluation/metrics.json").get("auc")
            if validation_auc is not None:
                drop = float(validation_auc - perf["auc"])
                summary["auc_drop_vs_validation"] = drop
                summary["performance_status"] = (
                    "alert"
                    if drop >= MONITORING["auc_drop_alert"]
                    else "warn"
                    if drop >= MONITORING["auc_drop_warn"]
                    else "stable"
                )
                if summary["performance_status"] == "alert" or summary["backtest_red_grades"] > 0:
                    summary["drift_alert"] = True
                    summary["overall_status"] = "alert"
            log_metrics(perf, prefix="live_")

        table.to_csv(out / "drift_table.csv", index=False)
        write_json(summary, out / "drift_summary.json")
        write_json(
            {
                "alert": summary["drift_alert"],
                "status": summary["overall_status"],
                "action": "Open incident per RUNBOOK section 'Drift alert'" if summary["drift_alert"] else "none",
            },
            out / "alert.json",
        )
        plots.save(plots.psi_bars(table, warn=args.psi_warn, alert=args.psi_alert), out / "plots" / "psi.png")
        plots.save(plots.score_shift(pkg.profile["score"], scored["pd"]), out / "plots" / "score_shift.png")

        log_metrics({k: v for k, v in summary.items() if not isinstance(v, dict | str)})
        mlflow.set_tag("drift_status", summary["overall_status"])
        mlflow.log_artifacts(str(out), artifact_path="monitoring")
        banner = "DRIFT ALERT" if summary["drift_alert"] else summary["overall_status"].upper()
        print(
            f"[{banner}] score PSI {summary['score_psi']:.3f} | max feature PSI {summary['max_feature_psi']:.3f} | "
            f"{summary['n_features_alert']} features in alert | mean PD {summary['reference_mean_pd']:.3f} -> {summary['current_mean_pd']:.3f}"
        )
        if not table.empty:
            print(pd.DataFrame(table.head(5)).to_string(index=False))


if __name__ == "__main__":
    main()
