"""Training pipeline (Azure ML SDK v2): preprocess -> train -> evaluate -> register.

    python pipeline/run_pipeline.py                     # create compute + data asset, submit
    python pipeline/run_pipeline.py --wait              # ... and stream logs until done
    python pipeline/run_pipeline.py --dry-run           # build and print the pipeline, no Azure calls
    python pipeline/run_pipeline.py --schedule "0 6 1 * *"   # monthly retraining schedule

Workspace from env vars (AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,
AZUREML_WORKSPACE, optionally in .env) or a config.json downloaded from the portal.
Components are the YAML files in components/ - the CLI route
(pipeline/training_pipeline.yml) uses the very same definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from pipeline.azure_utils import COMPONENTS, DEFAULTS, ROOT, ensure_compute, ensure_data_asset, get_ml_client, git_sha, stream_and_check


def build_pipeline(raw_data, compute: str, args, data_version: str, sha: str):
    from azure.ai.ml import dsl, load_component

    preprocess_component = load_component(source=COMPONENTS / "preprocess.yml")
    train_component = load_component(source=COMPONENTS / "train.yml")
    evaluate_component = load_component(source=COMPONENTS / "evaluate.yml")
    register_component = load_component(source=COMPONENTS / "register.yml")

    @dsl.pipeline(
        name="credit_default_training",
        display_name="credit-default-training",
        description="UCI credit default PD model: data-quality gate, Optuna-tuned monotonic LightGBM vs scorecard, "
        "calibration, backtest, explainability, champion/challenger registration.",
        default_compute=compute,
    )
    def training_pipeline(raw_data, model_name, n_trials, exclude_sensitive, cost_fn, cost_fp):  # types inferred by the SDK
        preprocess = preprocess_component(raw_data=raw_data)
        preprocess.display_name = "1 - preprocess"
        train = train_component(
            train_data=preprocess.outputs.train_data,
            n_trials=n_trials,
            exclude_sensitive=exclude_sensitive,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        train.display_name = "2 - train"
        evaluate = evaluate_component(model_input=train.outputs.model_output, test_data=preprocess.outputs.test_data, model_name=model_name)
        evaluate.display_name = "3 - evaluate"
        register = register_component(
            model_input=train.outputs.model_output,
            evaluation_output=evaluate.outputs.evaluation_output,
            model_name=model_name,
            git_sha=sha,
            data_version=data_version,
        )
        register.display_name = "4 - register"
        return {
            "model_info": register.outputs.model_info_output,
            "evaluation": evaluate.outputs.evaluation_output,
            "quality_report": preprocess.outputs.quality_report,
        }

    job = training_pipeline(
        raw_data=raw_data,
        model_name=args.model_name,
        n_trials=args.n_trials,
        exclude_sensitive=not args.include_sensitive,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
    )
    job.settings.default_datastore = "workspaceblobstore"
    job.settings.continue_on_step_failure = False
    job.tags = {"git_sha": sha, "data_version": data_version, "trigger": "manual" if not args.schedule else "schedule"}
    return job


def find_raw_file(path: str | None) -> Path:
    if path:
        return Path(path)
    raw_dir = ROOT / "data" / "raw"
    files = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in {".xls", ".xlsx", ".csv"})
    if not files:
        print("No raw file found - downloading from UCI ...")
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True)
        files = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in {".xls", ".xlsx", ".csv"})
    return files[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-file", help="Local raw file to register (default: data/raw/*, downloaded if missing)")
    p.add_argument("--compute", default=DEFAULTS["compute"])
    p.add_argument("--vm-size", default=DEFAULTS["vm_size"])
    p.add_argument("--max-nodes", type=int, default=DEFAULTS["max_nodes"])
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--model-name", dest="model_name", default=DEFAULTS["model_name"])
    p.add_argument("--n-trials", dest="n_trials", type=int, default=25)
    p.add_argument("--include-sensitive", action="store_true", help="Use SEX, MARRIAGE, AGE as features (off by default)")
    p.add_argument("--cost-fn", dest="cost_fn", type=float, default=5.0)
    p.add_argument("--cost-fp", dest="cost_fp", type=float, default=1.0)
    p.add_argument("--wait", action="store_true", help="Stream logs and fail if the pipeline fails")
    p.add_argument("--dry-run", action="store_true", help="Build the pipeline locally and print it; no Azure calls")
    p.add_argument("--schedule", help="Cron expression for a recurring retraining schedule, e.g. '0 6 1 * *'")
    args = p.parse_args()
    sha = git_sha()

    if args.dry_run:
        from azure.ai.ml import Input

        job = build_pipeline(Input(type="uri_file", path="azureml:credit-default-raw@latest"), args.compute, args, "dry-run", sha)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pipeline.yml"
            job.dump(out)
            print(out.read_text())
        print(f"Dry run OK: {len(job.jobs)} steps -> {', '.join(job.jobs)}")
        return

    from azure.ai.ml import Input

    ml_client = get_ml_client()
    print(f"Workspace: {ml_client.workspace_name} (resource group {ml_client.resource_group_name})")
    ensure_compute(ml_client, args.compute, args.vm_size, args.max_nodes)
    raw = find_raw_file(args.raw_file)
    asset = ensure_data_asset(ml_client, DEFAULTS["raw_asset"], raw, "UCI Default of Credit Card Clients (id 350), raw file", {"source": "UCI"})
    job = build_pipeline(Input(type="uri_file", path=asset.id), args.compute, args, f"{asset.name}:{asset.version}", sha)

    if args.schedule:
        from azure.ai.ml.constants import TimeZone
        from azure.ai.ml.entities import CronTrigger, JobSchedule

        schedule = JobSchedule(
            name="credit-default-retraining",
            trigger=CronTrigger(expression=args.schedule, time_zone=TimeZone.W_EUROPE_STANDARD_TIME),
            create_job=job,
            display_name="Credit default - scheduled retraining",
            description="Retrains, evaluates and (if the quality gate passes) registers a challenger model.",
        )
        ml_client.schedules.begin_create_or_update(schedule).result()
        print(f"Schedule 'credit-default-retraining' created: {args.schedule} (W. Europe time)")
        return

    submitted = ml_client.jobs.create_or_update(job, experiment_name=args.experiment)
    print(f"Submitted pipeline {submitted.name}\nStudio: {submitted.studio_url}")
    if args.wait:
        stream_and_check(ml_client, submitted)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                # Pipeline-level outputs have no path of their own, so download from the register step
                register_job = next(
                    j for j in ml_client.jobs.list(parent_job_name=submitted.name) if "register" in (j.display_name or "")
                )
                ml_client.jobs.download(register_job.name, output_name="model_info_output", download_path=tmp)
                info_file = next(Path(tmp).rglob("model_info.json"))
                info = json.loads(info_file.read_text())
                print(f"Model info: {json.dumps(info, indent=2)}")
                Path(ROOT / "outputs").mkdir(exist_ok=True)
                (ROOT / "outputs" / "last_model_info.json").write_text(json.dumps(info, indent=2))
                card = next(Path(tmp).rglob("MODEL_CARD.md"), None)
                if card:
                    (ROOT / "outputs" / "MODEL_CARD.md").write_text(card.read_text())
            except Exception as exc:  # noqa: BLE001
                print(f"Could not download model_info output: {exc}")


if __name__ == "__main__":
    main()
