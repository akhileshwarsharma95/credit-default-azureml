"""Batch scoring + monitoring pipeline (Azure ML SDK v2):

    score_batch --> monitor_drift   (PSI, data quality, realised performance)
               \\-> portfolio_insights (SQL analytics, Power BI tables)

    python pipeline/run_batch_pipeline.py --batch data/batches/portfolio_downturn.csv --wait
    python pipeline/run_batch_pipeline.py --batch data/batches/portfolio_stable.csv --schedule "0 7 * * 1"
    python pipeline/run_batch_pipeline.py --dry-run

Create the demo batches first with: python scripts/make_batches.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pipeline.azure_utils import (
    COMPONENTS,
    DEFAULTS,
    ensure_compute,
    ensure_data_asset,
    get_ml_client,
    resolve_model,
    stream_and_check,
)


def build_pipeline(model_input, scoring_input, compute: str):
    from azure.ai.ml import dsl, load_component

    score_component = load_component(source=COMPONENTS / "score_batch.yml")
    monitor_component = load_component(source=COMPONENTS / "monitor_drift.yml")
    insights_component = load_component(source=COMPONENTS / "portfolio_insights.yml")

    @dsl.pipeline(
        name="credit_default_batch_monitoring",
        display_name="credit-default-batch-scoring-and-monitoring",
        description="Scores a portfolio snapshot, checks drift against the model's reference profile and builds portfolio insights.",
        default_compute=compute,
    )
    def batch_pipeline(model_input, scoring_data):
        score = score_component(model_input=model_input, scoring_data=scoring_data)
        score.display_name = "score portfolio"
        monitor = monitor_component(model_input=model_input, scored_data=score.outputs.scored_output)
        monitor.display_name = "monitor drift"
        insights = insights_component(scored_data=score.outputs.scored_output)
        insights.display_name = "portfolio insights"
        return {
            "scored": score.outputs.scored_output,
            "monitoring": monitor.outputs.monitoring_output,
            "insights": insights.outputs.insights_output,
        }

    job = batch_pipeline(model_input=model_input, scoring_data=scoring_input)
    job.settings.default_datastore = "workspaceblobstore"
    return job


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", default="data/batches/portfolio_downturn.csv", help="Local CSV with client features")
    p.add_argument("--model-name", default=DEFAULTS["model_name"])
    p.add_argument("--model-version", default="latest")
    p.add_argument("--compute", default=DEFAULTS["compute"])
    p.add_argument("--experiment", default=f"{DEFAULTS['experiment']}-monitoring")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--schedule", help="Cron expression, e.g. weekly '0 7 * * 1'")
    args = p.parse_args()

    from azure.ai.ml import Input

    if args.dry_run:
        job = build_pipeline(
            Input(type="mlflow_model", path=f"azureml:{args.model_name}@latest"),
            Input(type="uri_file", path="azureml:credit-default-batch@latest"),
            args.compute,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "batch.yml"
            job.dump(out)
            print(out.read_text())
        print(f"Dry run OK: {len(job.jobs)} steps -> {', '.join(job.jobs)}")
        return

    ml_client = get_ml_client()
    ensure_compute(ml_client, args.compute, DEFAULTS["vm_size"], DEFAULTS["max_nodes"])
    model = resolve_model(ml_client, args.model_name, args.model_version)
    batch = Path(args.batch)
    asset = ensure_data_asset(
        ml_client, f"credit-default-batch-{batch.stem.replace('_', '-')}", batch, f"Scoring batch {batch.name}"
    )
    print(f"Scoring {asset.name}:{asset.version} with {model.name}:{model.version} ({model.type})")
    job = build_pipeline(Input(type=model.type, path=model.id), Input(type="uri_file", path=asset.id), args.compute)
    job.tags = {"model_version": str(model.version), "batch": f"{asset.name}:{asset.version}"}

    if args.schedule:
        from azure.ai.ml.constants import TimeZone
        from azure.ai.ml.entities import CronTrigger, JobSchedule

        schedule = JobSchedule(
            name="credit-default-weekly-monitoring",
            trigger=CronTrigger(expression=args.schedule, time_zone=TimeZone.W_EUROPE_STANDARD_TIME),
            create_job=job,
            display_name="Credit default - scheduled scoring and drift monitoring",
        )
        ml_client.schedules.begin_create_or_update(schedule).result()
        print(f"Schedule created: {args.schedule}")
        return

    submitted = ml_client.jobs.create_or_update(job, experiment_name=args.experiment)
    print(f"Submitted {submitted.name}\nStudio: {submitted.studio_url}")
    if args.wait:
        stream_and_check(ml_client, submitted)


if __name__ == "__main__":
    main()
