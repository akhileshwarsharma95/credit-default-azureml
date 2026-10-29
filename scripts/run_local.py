"""Run the whole solution locally, exactly as the Azure ML pipelines chain it.

Each step runs as its own process with the same CLI arguments as the Azure ML
components; MLflow tracking and the model registry use a local SQLite store.
Useful for development, CI, and as a demo fallback.

    python scripts/run_local.py                 # full run (25 Optuna trials)
    python scripts/run_local.py --n_trials 5    # quicker
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(step: str, args: list[str], env: dict, cwd: Path) -> None:
    print(f"\n=== {step} ===", flush=True)
    start = time.time()
    subprocess.run([sys.executable, str(SRC / f"{step}.py"), *args], check=True, env=env, cwd=cwd)
    print(f"--- {step} finished in {time.time() - start:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "local"))
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--model_name", default="credit-default-model")
    p.add_argument("--clean", action="store_true", help="Delete previous local outputs and registry first")
    args = p.parse_args()

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

    run_id = time.strftime("%Y%m%d-%H%M%S")
    w = out / "runs" / run_id
    run(
        "preprocess",
        [
            "--raw_data",
            args.raw_data,
            "--train_data",
            str(w / "prep/train"),
            "--test_data",
            str(w / "prep/test"),
            "--quality_report",
            str(w / "prep/quality"),
        ],
        env,
        out,
    )
    run(
        "train",
        [
            "--train_data",
            str(w / "prep/train"),
            "--model_output",
            str(w / "train/model"),
            "--n_trials",
            str(args.n_trials),
        ],
        env,
        out,
    )
    run(
        "evaluate",
        [
            "--model_input",
            str(w / "train/model"),
            "--test_data",
            str(w / "prep/test"),
            "--evaluation_output",
            str(w / "evaluate"),
            "--model_name",
            args.model_name,
        ],
        env,
        out,
    )
    run(
        "register",
        [
            "--model_input",
            str(w / "train/model"),
            "--evaluation_output",
            str(w / "evaluate"),
            "--model_name",
            args.model_name,
            "--model_info_output",
            str(w / "register"),
            "--git_sha",
            "local",
            "--data_version",
            "local",
        ],
        env,
        out,
    )

    info = json.loads((w / "register" / "model_info.json").read_text())
    package = out / "model_package"
    if info.get("registered"):
        if package.exists():
            shutil.rmtree(package)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{args.model_name}/{info['version']}", dst_path=str(package)
        )
        (package / "REGISTRY_VERSION").write_text(f"{args.model_name}:{info['version']} (local registry)\n")
        print(f"\nRegistered version {info['version']} downloaded to {package}")
    elif not package.exists():
        raise SystemExit("Model was not registered and no previous package exists")

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
            [
                "--model_input",
                str(package),
                "--scoring_data",
                str(batches / f"portfolio_{name}.csv"),
                "--scored_output",
                str(b / "scored"),
            ],
            env,
            out,
        )
        run(
            "monitor_drift",
            [
                "--model_input",
                str(package),
                "--scored_data",
                str(b / "scored"),
                "--monitoring_output",
                str(b / "monitoring"),
            ],
            env,
            out,
        )
        run(
            "portfolio_insights",
            ["--scored_data", str(b / "scored"), "--insights_output", str(b / "insights")],
            env,
            out,
        )

    latest = out / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink() if latest.is_symlink() else shutil.rmtree(latest)
    latest.symlink_to(w, target_is_directory=True)
    print(
        f"\nDone. Outputs in {w}\nMLflow UI: mlflow ui --backend-store-uri {tracking}\nApp: MODEL_DIR={package} streamlit run app/streamlit_app.py"
    )


if __name__ == "__main__":
    main()
