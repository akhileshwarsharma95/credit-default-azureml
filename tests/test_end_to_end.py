"""End-to-end: every pipeline step with the same CLI arguments as the Azure ML
components, on synthetic UCI-format data, with a local MLflow registry. Also
covers champion/challenger registration, the endpoint scoring script, the
agent and the Streamlit app."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def step(name: str, args: list[str]) -> None:
    sys.path.insert(0, str(SRC))
    module = importlib.import_module(name)
    module.main([str(a) for a in args])


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory, raw_file, mlflow_store):
    w = tmp_path_factory.mktemp("run")
    step(
        "preprocess",
        ["--raw_data", raw_file, "--train_data", w / "train", "--test_data", w / "test", "--quality_report", w / "dq"],
    )
    step(
        "train",
        [
            "--train_data",
            w / "train",
            "--model_output",
            w / "model",
            "--n_trials",
            2,
            "--tuning_folds",
            2,
            "--cv_folds",
            3,
        ],
    )
    step(
        "evaluate",
        [
            "--model_input",
            w / "model",
            "--test_data",
            w / "test",
            "--evaluation_output",
            w / "eval",
            "--model_name",
            "test-model",
            "--n_bootstrap",
            50,
            "--min_auc",
            0.6,
            "--max_ece",
            0.2,
        ],
    )
    step(
        "register",
        [
            "--model_input",
            w / "model",
            "--evaluation_output",
            w / "eval",
            "--model_name",
            "test-model",
            "--model_info_output",
            w / "reg",
            "--git_sha",
            "abc123",
            "--data_version",
            "raw:1",
        ],
    )
    import mlflow

    info = json.loads((w / "reg" / "model_info.json").read_text())
    pkg = w / "package"
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/test-model/{info['version']}", dst_path=str(pkg))
    return w, pkg, info


def test_preprocess_outputs(pipeline_run):
    w, _, _ = pipeline_run
    dq = json.loads((w / "dq" / "quality_report.json").read_text())
    assert dq["passed"] and dq["cleaning_actions"]["duplicates_removed"] == 5
    train, test = pd.read_csv(w / "train" / "train.csv"), pd.read_csv(w / "test" / "test.csv")
    assert len(train) + len(test) == dq["rows_out"]
    assert abs(train["default"].mean() - test["default"].mean()) < 0.02  # stratified


def test_model_package_contents(pipeline_run):
    _, pkg, info = pipeline_run
    assert info["registered"] and info["version"] == "1"
    for rel in [
        "MLmodel",
        "training_summary.json",
        "reference/reference_profile.json",
        "reference/background.csv",
        "evaluation/metrics.json",
        "evaluation/backtest.csv",
        "evaluation/test_scored.csv",
        "MODEL_CARD.md",
    ]:
        assert (pkg / rel).exists(), rel
    summary = json.loads((pkg / "training_summary.json").read_text())
    assert not set(summary["features"]) & {"SEX", "MARRIAGE", "AGE"}  # protected attributes excluded by default
    assert summary["threshold"] == pytest.approx(1 / 6)


def test_evaluation_quality(pipeline_run):
    w, _, _ = pipeline_run
    m = json.loads((w / "eval" / "metrics.json").read_text())
    assert m["auc"] > 0.7 and m["auc_ci_low"] < m["auc"] < m["auc_ci_high"]
    assert 0 < m["ece"] < 0.1
    assert (w / "eval" / "plots" / "backtest.png").exists()


def test_champion_challenger_second_registration(pipeline_run, tmp_path):
    w, _, _ = pipeline_run
    step(
        "evaluate",
        [
            "--model_input",
            w / "model",
            "--test_data",
            w / "test",
            "--evaluation_output",
            tmp_path / "eval2",
            "--model_name",
            "test-model",
            "--n_bootstrap",
            20,
            "--min_auc",
            0.6,
            "--max_ece",
            0.2,
        ],
    )
    decision = json.loads((tmp_path / "eval2" / "decision.json").read_text())
    assert decision["champion"]["status"] == "compared" and decision["champion"]["version"] == "1"
    assert decision["register"]  # identical model is not worse than the champion
    step(
        "register",
        [
            "--model_input",
            w / "model",
            "--evaluation_output",
            tmp_path / "eval2",
            "--model_name",
            "test-model",
            "--model_info_output",
            tmp_path / "reg2",
        ],
    )
    assert json.loads((tmp_path / "reg2" / "model_info.json").read_text())["version"] == "2"


def test_quality_gate_blocks_registration(pipeline_run, tmp_path):
    w, _, _ = pipeline_run
    step(
        "evaluate",
        [
            "--model_input",
            w / "model",
            "--test_data",
            w / "test",
            "--evaluation_output",
            tmp_path / "eval3",
            "--model_name",
            "test-model",
            "--n_bootstrap",
            20,
            "--min_auc",
            0.99,
        ],
    )
    step(
        "register",
        [
            "--model_input",
            w / "model",
            "--evaluation_output",
            tmp_path / "eval3",
            "--model_name",
            "test-model",
            "--model_info_output",
            tmp_path / "reg3",
        ],
    )
    assert json.loads((tmp_path / "reg3" / "model_info.json").read_text())["registered"] is False


def test_batch_scoring_monitoring_insights(pipeline_run, tmp_path):
    w, pkg, _ = pipeline_run
    from creditrisk import drift

    test = pd.read_csv(w / "test" / "test.csv")
    drift.simulate_downturn(test).to_csv(tmp_path / "downturn.csv", index=False)
    for name, batch in [("stable", w / "test" / "test.csv"), ("downturn", tmp_path / "downturn.csv")]:
        out = tmp_path / name
        step("score_batch", ["--model_input", pkg, "--scoring_data", batch, "--scored_output", out / "scored"])
        step(
            "monitor_drift", ["--model_input", pkg, "--scored_data", out / "scored", "--monitoring_output", out / "mon"]
        )
        step("portfolio_insights", ["--scored_data", out / "scored", "--insights_output", out / "ins"])
    stable = json.loads((tmp_path / "stable" / "mon" / "drift_summary.json").read_text())
    downturn = json.loads((tmp_path / "downturn" / "mon" / "drift_summary.json").read_text())
    assert stable["overall_status"] in {"stable", "warn"} and stable["score_psi"] < 0.1
    assert downturn["drift_alert"] is True
    assert "performance" in stable  # labelled batch -> realised performance
    scored = pd.read_csv(tmp_path / "stable" / "scored" / "scored.csv")
    assert {"pd", "grade", "action", "el", "ifrs9_stage", "reason_1", "limit_action"} <= set(scored.columns)
    for table in ["grade_summary", "watchlist", "segment_summary", "roll_rate_matrix", "delinquency_projection"]:
        assert (tmp_path / "stable" / "ins" / f"{table}.csv").exists()


def test_endpoint_scoring_script(pipeline_run, monkeypatch):
    w, pkg, _ = pipeline_run
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(pkg))
    sys.path.insert(0, str(SRC / "serving"))
    score = importlib.import_module("score")
    score.init()
    test = pd.read_csv(w / "test" / "test.csv").head(3)
    out = score.run(json.dumps({"clients": test.drop(columns=["default"]).to_dict(orient="records")}))
    assert len(out["predictions"]) == 3 and all(0 < p["pd"] < 1 for p in out["predictions"])
    assert out["predictions"][0]["ID"] == int(test["ID"].iloc[0])
    split = {"input_data": {"columns": list(test.columns), "data": test.values.tolist()}}
    assert len(score.run(json.dumps(split))["predictions"]) == 3
    assert "error" in score.run(json.dumps({"clients": [{"LIMIT_BAL": 1}]}))


def test_agent_offline_answers(pipeline_run):
    _, pkg, _ = pipeline_run
    from creditrisk import insights
    from creditrisk.agent import PortfolioAgent

    scored = pd.read_csv(pkg / "evaluation" / "test_scored.csv")
    _, tables = insights.portfolio_insights(scored)
    con = insights.connect(
        {
            "portfolio": tables["fact_scores"],
            "backtest": pd.read_csv(pkg / "evaluation" / "backtest.csv"),
            **{k: v for k, v in tables.items() if k != "fact_scores"},
        }
    )
    agent = PortfolioAgent(con)
    for q in [
        "exposure by grade",
        "segments",
        "IFRS 9 stages",
        "did the backtest flag anything",
        f"explain client {int(scored['ID'].iloc[0])}",
    ]:
        r = agent.ask(q)
        assert r.tables and len(r.tables[0]) > 0, q


def test_streamlit_app_renders(pipeline_run, monkeypatch):
    _, pkg, _ = pipeline_run
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MODEL_DIR", str(pkg))
    for k in ("AZURE_OPENAI_ENDPOINT", "OPENAI_API_KEY", "SCORING_URI"):
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6


@pytest.mark.skipif(importlib.util.find_spec("azure.ai.ml") is None, reason="azure-ai-ml not installed")
@pytest.mark.parametrize(
    "script",
    [
        "pipeline/run_pipeline.py",
        "pipeline/run_batch_pipeline.py",
        "deploy/deploy_endpoint.py",
        "monitoring/setup_model_monitor.py",
    ],
)
def test_azure_specs_build(script):
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / script), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=os.environ | {"PYTHONWARNINGS": "ignore"},
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Dry run OK" in r.stdout + r.stderr  # logged on stderr; stdout carries only the spec
