"""Run the whole solution locally, exactly as the Azure ML pipelines chain it.

Each step runs as its own process with the same CLI arguments as the Azure ML
components; MLflow tracking and the model registry use a local SQLite store.
Useful for development, CI, and as a demo fallback.

    uv run python scripts/run_local.py                 # full run (Optuna trials from config.yaml)
    uv run python scripts/run_local.py --n_trials 5    # quicker
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from creditrisk.config import load
from creditrisk.observability import get_logger

LOGGER = get_logger("local")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(step: str, params: dict, env: dict, cwd: Path) -> None:
    """Run one pipeline step as its own process, with the same CLI flags as its Azure ML component."""
    LOGGER.info(f"=== {step} ===")
    start = time.time()
    flags = [x for k, v in params.items() for x in (f"--{k}", str(v))]
    subprocess.run([sys.executable, str(SRC / f"{step}.py"), *flags], check=True, env=env, cwd=cwd)
    LOGGER.info(f"--- {step} finished in {time.time() - start:.0f}s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "local"))
    p.add_argument("--n_trials", type=int, default=load()["technical"]["training"]["n_trials"])
    p.add_argument("--model_name", default=load()["technical"]["azure"]["model_name"])
    p.add_argument("--clean", action="store_true", help="Delete previous local outputs and registry first")
    return p.parse_args()


def fetch_registered_package(mlflow, info: dict, model_name: str, package: Path) -> None:
    """Download the newly registered version from the local registry, as the app and batch steps use it."""
    if info.get("registered"):
        if package.exists():
            shutil.rmtree(package)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{info['version']}", dst_path=str(package)
        )
        (package / "REGISTRY_VERSION").write_text(f"{model_name}:{info['version']} (local registry)\n")
        LOGGER.info(f"Registered version {info['version']} downloaded to {package}")
    elif not package.exists():
        raise SystemExit("Model was not registered and no previous package exists")


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    import mlflow

    tracking = f"sqlite:///{out / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    exp = "credit-default-local"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(out / "mlruns").as_uri())
    env = os.environ | {"MLFLOW_TRACKING_URI": tracking, "MLFLOW_EXPERIMENT_NAME": exp, "PYTHONHASHSEED": "0"}

    w = out / "runs" / time.strftime("%Y%m%d-%H%M%S")
    prep, model, evaluation = w / "prep", w / "train/model", w / "evaluate"
    run(
        "preprocess",
        {
            "raw_data": args.raw_data,
            "train_data": prep / "train",
            "test_data": prep / "test",
            "quality_report": prep / "quality",
        },
        env,
        out,
    )
    run("train", {"train_data": prep / "train", "model_output": model, "n_trials": args.n_trials}, env, out)
    run(
        "evaluate",
        {
            "model_input": model,
            "test_data": prep / "test",
            "evaluation_output": evaluation,
            "model_name": args.model_name,
        },
        env,
        out,
    )
    run(
        "register",
        {
            "model_input": model,
            "evaluation_output": evaluation,
            "model_name": args.model_name,
            "model_info_output": w / "register",
            "git_sha": "local",
            "data_version": "local",
        },
        env,
        out,
    )

    package = out / "model_package"
    fetch_registered_package(
        mlflow, json.loads((w / "register" / "model_info.json").read_text()), args.model_name, package
    )

    batches = out / "batches"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_batches.py"),
            "--raw_data",
            args.raw_data,
            "--dest",
            str(batches),
        ],
        check=True,
        env=env,
    )
    for name in ["stable", "downturn"]:
        b = w / f"batch_{name}"
        run(
            "score_batch",
            {"model_input": package, "scoring_data": batches / f"portfolio_{name}.csv", "scored_output": b / "scored"},
            env,
            out,
        )
        run(
            "monitor_drift",
            {"model_input": package, "scored_data": b / "scored", "monitoring_output": b / "monitoring"},
            env,
            out,
        )
        run("portfolio_insights", {"scored_data": b / "scored", "insights_output": b / "insights"}, env, out)

    latest = out / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink() if latest.is_symlink() else shutil.rmtree(latest)
    latest.symlink_to(w, target_is_directory=True)
    print(
        f"\nDone. Outputs in {w}\nMLflow UI: mlflow ui --backend-store-uri {tracking}\nApp: MODEL_DIR={package} uv run streamlit run app/streamlit_app.py"
    )


if __name__ == "__main__":
    main()
