"""Azure ML model monitoring for the online endpoint (built-in signals).

Creates a recurring monitor on the data the endpoint's data collector writes:
data drift, prediction drift and data quality, compared with the training
data, with e-mail alerts. Runs on serverless Spark (billed per run).

    python monitoring/setup_model_monitor.py --endpoint-name credit-risk-xxxxxx --email you@example.com
    python monitoring/setup_model_monitor.py --dry-run

Complements the pipeline-based PSI monitor (pipeline/run_batch_pipeline.py),
which covers batch-scored portfolios and also checks realised performance.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from deploy.deploy_endpoint import default_endpoint_name
from pipeline.azure_utils import get_ml_client


def build_schedule(endpoint: str, deployment: str, emails: list[str], hour: int):
    from azure.ai.ml.entities import (
        AlertNotification,
        MonitorDefinition,
        MonitoringTarget,
        MonitorSchedule,
        RecurrencePattern,
        RecurrenceTrigger,
        ServerlessSparkCompute,
    )

    definition = MonitorDefinition(
        compute=ServerlessSparkCompute(instance_type="standard_e4s_v3", runtime_version="3.4"),
        monitoring_target=MonitoringTarget(
            ml_task="classification", endpoint_deployment_id=f"azureml:{endpoint}:{deployment}"
        ),
        alert_notification=AlertNotification(emails=emails) if emails else None,
    )
    return MonitorSchedule(
        name="credit-default-endpoint-monitor",
        display_name="Credit default endpoint - drift and data quality",
        trigger=RecurrenceTrigger(frequency="day", interval=1, schedule=RecurrencePattern(hours=hour, minutes=0)),
        create_monitor=definition,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint-name", default=default_endpoint_name())
    p.add_argument("--deployment", default=None, help="Deployment to monitor (default: the one with most traffic)")
    p.add_argument("--email", action="append", default=[])
    p.add_argument("--hour", type=int, default=6, help="Daily run hour (UTC)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        schedule = build_schedule(
            args.endpoint_name, args.deployment or "blue", args.email or ["risk-team@example.com"], args.hour
        )
        with tempfile.TemporaryDirectory() as tmp:
            schedule.dump(Path(tmp) / "monitor.yml")
            print((Path(tmp) / "monitor.yml").read_text())
        print("Dry run OK: model monitor schedule spec")
        return

    ml_client = get_ml_client()
    deployment = args.deployment
    if deployment is None:
        traffic = ml_client.online_endpoints.get(args.endpoint_name).traffic or {}
        deployment = max(traffic, key=traffic.get)
    ml_client.schedules.begin_create_or_update(
        build_schedule(args.endpoint_name, deployment, args.email, args.hour)
    ).result()
    print(f"Model monitor created for {args.endpoint_name}/{deployment} (daily at {args.hour:02d}:00 UTC)")


if __name__ == "__main__":
    main()
