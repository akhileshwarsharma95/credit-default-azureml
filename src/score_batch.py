"""Batch scoring - PD, rating grade, action, reason codes, exposure, expected
loss, IFRS 9 stage, early-warning triggers and limit recommendations for a
portfolio snapshot. Output is a flat fact table (Power BI ready)."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from creditrisk import quality
from creditrisk.observability import get_logger
from creditrisk.schema import TARGET, load_batch
from creditrisk.scoring import load_package, score_frame
from creditrisk.tracking import log_metrics, write_json

LOGGER = get_logger("score_batch")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_input", required=True, help="Registered MLflow model folder")
    p.add_argument("--scoring_data", required=True, help="CSV/XLS(X) file or folder with raw client features")
    p.add_argument("--scored_output", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.scored_output)
    out.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run():
        mlflow.set_tags({"stage": "score"})
        pkg = load_package(args.model_input)
        df = load_batch(args.scoring_data)
        df, report = quality.validate(df, require_target=TARGET in df.columns)
        if not report.passed:
            write_json(report.to_dict(), out / "quality_report.json")
            raise SystemExit(f"Scoring data failed the data-quality gate: {report.hard_failures}")
        df = quality.clean_for_scoring(df, report)
        write_json(report.to_dict(), out / "quality_report.json")

        scored = score_frame(pkg, df)
        scored.to_csv(out / "scored.csv", index=False)
        log_metrics(
            {
                "clients_scored": len(scored),
                "mean_pd": scored["pd"].mean(),
                "intervene_rate": (scored["action"] == "Intervene").mean(),
                "watch_rate": (scored["action"] == "Watch").mean(),
                "total_ead": scored["ead"].sum(),
                "total_el": scored["el"].sum(),
                "dq_soft_violations": sum(f["n_failures"] for f in report.soft_failures),
            }
        )
        mlflow.log_artifact(str(out / "quality_report.json"), artifact_path="scoring")
        LOGGER.info(
            f"Scored {len(scored):,} clients | mean PD {scored['pd'].mean():.3f} | intervene {(scored['action'] == 'Intervene').mean():.1%}"
        )


if __name__ == "__main__":
    main()
