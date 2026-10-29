"""Unit tests for the creditrisk library."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import ROOT, make_synthetic
from creditrisk import drift, metrics, policy, quality
from creditrisk.agent import SqlGuardError, guard_sql
from creditrisk.config import load
from creditrisk.features import ENGINEERED, FeatureEngineer
from creditrisk.schema import RAW_FEATURES, _standardise_columns


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    df = _standardise_columns(make_synthetic(800, seed=1))
    return df


def test_quality_gate_passes_and_cleans(raw):
    df, report = quality.validate(raw)
    assert report.passed
    assert any(f["column"] == "EDUCATION" for f in report.soft_failures)  # undocumented codes are reported
    clean = quality.clean(df, report)
    assert report.actions["duplicates_removed"] == 5
    assert set(clean["EDUCATION"]) <= {1, 2, 3, 4} and set(clean["MARRIAGE"]) <= {1, 2, 3}
    _, post = quality.validate(clean)
    assert post.passed and not post.soft_failures


def test_quality_gate_fails_on_impossible_values(raw):
    bad = raw.copy()
    bad.loc[0, "AGE"] = 5
    bad.loc[1, "PAY_1"] = 42
    _, report = quality.validate(bad)
    assert not report.passed
    assert {f["column"] for f in report.hard_failures} >= {"AGE", "PAY_1"}


def test_feature_engineering_is_finite_and_stateless(raw):
    X = raw[RAW_FEATURES]
    out = FeatureEngineer().fit(X).transform(X)
    assert list(out.columns[-len(ENGINEERED) :]) == ENGINEERED
    assert np.isfinite(out[ENGINEERED].to_numpy(dtype=float)).all()
    first = FeatureEngineer().fit(X.iloc[:10]).transform(X.iloc[:10])
    pd.testing.assert_frame_equal(first, out.iloc[:10])  # row-wise: no dependence on other rows


def test_discrimination_and_ks_known_values():
    y = np.array([0, 0, 1, 1])
    assert metrics.discrimination(y, np.array([0.1, 0.2, 0.8, 0.9]))["auc"] == 1.0
    assert metrics.ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert metrics.discrimination(y, np.array([0.9, 0.8, 0.2, 0.1]))["gini"] == -1.0


def test_jeffreys_flags_underestimation():
    assert metrics.jeffreys_pvalue(30, 100, 0.10) < 0.001  # 30% observed vs 10% predicted
    assert metrics.jeffreys_pvalue(10, 100, 0.10) > 0.05
    bt = metrics.backtest_by_grade(np.r_[np.ones(30), np.zeros(70)], np.full(100, 0.1), np.full(100, "D"))
    assert bt.iloc[0]["traffic_light"] == "red"


def test_cost_optimal_threshold_and_grades():
    assert policy.cost_optimal_threshold(5, 1) == pytest.approx(1 / 6)
    assert list(policy.assign_grade([0.001, 0.06, 0.1, 0.15, 0.3, 0.5, 0.99])) == list("ABCDEFG")


def test_pd_floor_wrapper():
    from sklearn.dummy import DummyClassifier

    from creditrisk.modeling import FlooredPD

    X, y = np.zeros((10, 1)), np.r_[np.zeros(10)]
    y[0] = 1
    m = FlooredPD.wrap(DummyClassifier(strategy="prior").fit(X, y), 0.2)
    assert m.predict_proba(X)[:, 1].min() == pytest.approx(0.2)


def test_psi_identical_is_zero_and_shift_alerts(raw):
    X = FeatureEngineer().fit(raw[RAW_FEATURES]).transform(raw[RAW_FEATURES])
    scores = np.random.default_rng(0).uniform(size=len(X))
    profile = drift.build_profile(X, scores)
    table, summary = drift.drift_report(profile, X, scores)
    assert summary["score_psi"] < 1e-6 and summary["overall_status"] == "stable"
    shocked = drift.simulate_downturn(raw[RAW_FEATURES], severity=1.5)
    Xs = FeatureEngineer().fit(shocked).transform(shocked)
    _, s2 = drift.drift_report(profile, Xs, np.clip(scores + 0.3, 0, 1))
    assert s2["drift_alert"]


def test_psi_is_robust_to_csv_float_roundtrip():
    rng = np.random.default_rng(3)
    steps = rng.choice(np.linspace(0.03, 0.8, 40) + 1e-9 / 3, size=5000)  # isotonic-like discrete PDs
    spec = drift._numeric_spec(steps, 10)
    noisy = steps * (1 + rng.choice([-1, 0, 1], size=len(steps)) * 1e-16)  # last-bit differences
    assert drift.psi(spec["expected"], drift._actual_shares(noisy, spec)) < 1e-6


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE portfolio",
        "SELECT 1; DELETE FROM portfolio",
        "COPY portfolio TO 'x.csv'",
        "SELECT * FROM read_csv('/etc/passwd')",
        "ATTACH 'x.db'",
        "WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM a",
        "PRAGMA database_list",
    ],
)
def test_sql_guard_blocks_unsafe(query):
    with pytest.raises(SqlGuardError):
        guard_sql(query)


def test_sql_guard_allows_select_and_caps_rows():
    q = guard_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x;")
    assert q.startswith("SELECT * FROM (WITH x") and q.endswith("LIMIT 200")


def test_duckdb_external_access_locked():
    import duckdb

    from creditrisk.insights import connect

    con = connect({"t": pd.DataFrame({"a": [1, 2]})})
    assert con.execute("SELECT SUM(a) FROM t").fetchone()[0] == 3
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv_auto('/etc/hostname')").fetchall()


def test_step_scripts_import_only_creditrisk_and_third_party():
    # Azure step environments do not install pipeline/ or deploy/
    for script in (ROOT / "src").glob("*.py"):
        tree = ast.parse(script.read_text())
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not names & {"pipeline", "deploy", "azure_utils"}, script.name


def test_config_drives_policy_and_monitoring_constants():
    cfg = load()
    assert policy.PD_FLOOR == cfg["business"]["pd_floor"] == 0.001
    assert policy.cost_optimal_threshold(**cfg["business"]["costs"]) == pytest.approx(1 / 6)
    assert [g for g, _, _ in policy.MASTER_SCALE] == [g for g, _, _ in cfg["business"]["master_scale"]]
    assert (drift.PSI_WARN, drift.PSI_ALERT) == (0.10, 0.25)


@pytest.mark.parametrize(
    ("component", "name", "section", "key"),
    [
        ("train", "n_trials", "training", "n_trials"),
        ("train", "cost_fn", "costs", "cost_fn"),
        ("train", "cost_fp", "costs", "cost_fp"),
        ("train", "lgd", "loss", "lgd"),
        ("train", "ccf", "loss", "ccf"),
        ("evaluate", "min_auc", "registration_gate", "min_auc"),
        ("evaluate", "max_ece", "registration_gate", "max_ece"),
        ("evaluate", "champion_tolerance", "registration_gate", "champion_tolerance"),
        ("monitor_drift", "psi_warn", "monitoring", "psi_warn"),
        ("monitor_drift", "psi_alert", "monitoring", "psi_alert"),
        ("preprocess", "test_size", "data", "test_size"),
    ],
)
def test_component_defaults_match_config(component, name, section, key):
    # Azure ML passes component defaults to the steps, so they must not drift from config.yaml
    cfg = load()
    expected = (cfg["business"] | cfg["technical"])[section][key]
    spec = yaml.safe_load((ROOT / "components" / f"{component}.yml").read_text())
    assert spec["inputs"][name]["default"] == expected


def test_cli_pipeline_defaults_match_config():
    inputs = yaml.safe_load((ROOT / "pipeline" / "training_pipeline.yml").read_text())["inputs"]
    cfg = load()
    training, costs = cfg["technical"]["training"], cfg["business"]["costs"]
    assert (inputs["n_trials"], inputs["exclude_sensitive"]) == (training["n_trials"], training["exclude_sensitive"])
    assert (inputs["cost_fn"], inputs["cost_fp"]) == (costs["cost_fn"], costs["cost_fp"])
    assert inputs["model_name"] == cfg["technical"]["azure"]["model_name"]
