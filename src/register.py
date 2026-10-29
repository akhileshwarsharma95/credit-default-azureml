"""Step 4 - Register the model in the Azure ML (MLflow) model registry, but
only if the evaluation quality gate passed (champion/challenger).

The registered artifact is a complete package: MLflow model, drift reference,
evaluation evidence (metrics, backtest, plots, scored test set) and a model card.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from creditrisk import model_card
from creditrisk.observability import get_logger
from creditrisk.tracking import write_json

LOGGER = get_logger("register")


def str2bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_input", required=True)
    p.add_argument("--evaluation_output", required=True)
    p.add_argument("--model_name", default="credit-default-model")
    p.add_argument("--model_info_output", required=True)
    p.add_argument("--git_sha", default="unknown")
    p.add_argument("--data_version", default="unknown")
    p.add_argument("--force", type=str2bool, default=False, help="Register even if the quality gate failed")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = Path(args.evaluation_output)
    decision = json.loads((eval_dir / "decision.json").read_text())
    m = json.loads((eval_dir / "metrics.json").read_text())
    fair = json.loads((eval_dir / "fairness_summary.json").read_text())
    summary = json.loads((Path(args.model_input) / "training_summary.json").read_text())
    info_dir = Path(args.model_info_output)
    info_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.set_tags({"stage": "register", "git_sha": args.git_sha, "data_version": args.data_version})
        if not (decision["register"] or args.force):
            write_json(
                {"registered": False, "reason": decision["checks"], "champion": decision["champion"]},
                info_dir / "model_info.json",
            )
            mlflow.log_metric("registered", 0)
            LOGGER.warning(f"Quality gate failed - model NOT registered: {decision['checks']}")
            return

        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "model"
            shutil.copytree(args.model_input, pkg)
            shutil.copytree(eval_dir, pkg / "evaluation")
            card = model_card.build(
                summary,
                m,
                decision,
                pd.read_csv(eval_dir / "backtest.csv"),
                pd.read_csv(eval_dir / "importance.csv"),
                fair,
                args.model_name,
                git_sha=args.git_sha,
                data_version=args.data_version,
            )
            (pkg / "MODEL_CARD.md").write_text(card)
            mlflow.log_artifacts(str(pkg), artifact_path="model")
            mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", args.model_name)

            # Re-render the card with the version number now that it is known
            card = card.replace(
                f"# Model card - {args.model_name}", f"# Model card - {args.model_name} v{mv.version}", 1
            )
            (info_dir / "MODEL_CARD.md").write_text(card)
            mlflow.log_artifact(str(info_dir / "MODEL_CARD.md"))

        tags = {
            "model_family": summary["model_family"],
            "test_auc": f"{m['auc']:.4f}",
            "test_gini": f"{m['gini']:.4f}",
            "test_ece": f"{m['ece']:.4f}",
            "threshold": f"{summary['threshold']:.4f}",
            "git_sha": args.git_sha,
            "data_version": args.data_version,
            "exclude_sensitive": str(summary["exclude_sensitive"]),
            "champion_comparison": decision["champion"].get("status", ""),
        }
        client = MlflowClient()
        for k, v in tags.items():
            try:
                client.set_model_version_tag(args.model_name, mv.version, k, v)
            except Exception as exc:  # noqa: BLE001 - tags are metadata, never block registration
                LOGGER.warning(f"Could not set tag {k}: {exc}")
        try:
            client.update_model_version(
                args.model_name,
                mv.version,
                description=f"PD model ({summary['model_family']}), test AUC {m['auc']:.3f}",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Could not set description: {exc}")

        info = {
            "registered": True,
            "name": args.model_name,
            "version": str(mv.version),
            "run_id": run.info.run_id,
            "tags": tags,
        }
        write_json(info, info_dir / "model_info.json")
        mlflow.log_metric("registered", 1)
        LOGGER.info(f"Registered {args.model_name} version {mv.version}")


if __name__ == "__main__":
    main()
