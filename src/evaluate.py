"""Step 3 - Evaluate the trained model on the untouched test set.

Discrimination (with bootstrap CIs), calibration, cut-off economics, PD
backtest by rating grade, fairness, SHAP explainability and reason codes.
It also loads the currently registered model (champion) and scores it on the
same test set, then writes a quality-gate decision for the register step.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from creditrisk import explain, metrics, plots, policy
from creditrisk.modeling import unwrap_base
from creditrisk.schema import MARRIAGE_LABELS, SEX_LABELS, TARGET, read_table
from creditrisk.scoring import load_package, score_frame
from creditrisk.tracking import log_metrics, log_table, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_input", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--evaluation_output", required=True)
    p.add_argument("--model_name", default="credit-default-model")
    p.add_argument("--min_auc", type=float, default=0.70)
    p.add_argument("--max_ece", type=float, default=0.05)
    p.add_argument("--champion_tolerance", type=float, default=0.005)
    p.add_argument("--n_bootstrap", type=int, default=500)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args(argv)


def champion_auc(model_name: str, test: pd.DataFrame, y: pd.Series) -> dict:
    """Score the latest registered version (the champion) on the same test set."""
    try:
        versions = MlflowClient().search_model_versions(f"name='{model_name}'")
    except Exception as exc:  # noqa: BLE001 - registry unavailable is not fatal
        return {"status": f"skipped: {exc.__class__.__name__}"}
    if not versions:
        return {"status": "no champion yet"}
    latest = max(versions, key=lambda v: int(v.version))
    try:
        champ = mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}")
        cols = list(unwrap_base(champ).named_steps["features"].feature_names_in_)
        auc = metrics.discrimination(y, champ.predict_proba(test[cols])[:, 1])["auc"]
        return {"status": "compared", "version": str(latest.version), "auc": float(auc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": f"skipped: could not score champion v{latest.version} ({exc.__class__.__name__})"}


def fairness_groups(test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SEX": test["SEX"].map(SEX_LABELS),
            "AGE": pd.cut(test["AGE"], [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]).astype(str),
            "MARRIAGE": test["MARRIAGE"].map(MARRIAGE_LABELS),
        },
        index=test.index,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.evaluation_output)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.set_tags({"stage": "evaluate"})
        pkg = load_package(args.model_input)
        test = read_table(args.test_data, "test.csv")
        X, y = test[pkg.features], test[TARGET].astype(int)
        a = pkg.assumptions
        t = pkg.threshold

        p = pkg.model.predict_proba(X)[:, 1]
        p_uncal = unwrap_base(pkg.model).predict_proba(X)[:, 1]

        # Discrimination and calibration
        m = metrics.discrimination(y, p)
        m |= metrics.bootstrap_ci(y, p, args.n_bootstrap, args.random_state)
        m |= metrics.calibration(y, p)
        m |= {
            f"uncalibrated_{k}": v
            for k, v in metrics.calibration(y, p_uncal).items()
            if k in {"brier", "ece", "log_loss", "calibration_slope"}
        }
        tm = metrics.threshold_metrics(y, p, t, a.cost_fn, a.cost_fp)
        m |= tm
        m |= {"recall_at_0.5": metrics.threshold_metrics(y, p, 0.5, a.cost_fn, a.cost_fp)["recall"]}

        # Business validation tables
        curve = metrics.cost_curve(y, p, a.cost_fn, a.cost_fp)
        dec = metrics.decile_table(y, p)
        grades = policy.assign_grade(p)
        bt = metrics.backtest_by_grade(y, p, grades)
        fair, fair_summary = metrics.fairness_table(y, p, t, fairness_groups(test))
        m |= {
            "backtest_red_grades": int((bt["traffic_light"] == "red").sum()),
            "backtest_amber_grades": int((bt["traffic_light"] == "amber").sum()),
        }
        m |= fair_summary

        # Explainability
        contrib, base_values = explain.contributions(pkg.model, X, pkg.background)
        feats = explain.feature_frame(pkg.model, X)
        imp = explain.global_importance(contrib)
        reasons = explain.reason_codes(contrib, feats, top_n=4, medians=pkg.profile.get("medians"))
        flagged_idx = np.where(p >= t)[0]
        freq = Counter(r["description"] for i in flagged_idx for r in reasons[i][:1])
        reason_freq = pd.DataFrame(freq.most_common(), columns=["primary_reason", "clients"])

        # Champion / challenger
        champ = champion_auc(args.model_name, test, y)
        checks = {
            "auc_above_minimum": bool(m["auc"] >= args.min_auc),
            "calibration_ece_ok": bool(m["ece"] <= args.max_ece),
            "not_worse_than_champion": bool(
                champ.get("auc") is None or m["auc"] >= champ["auc"] - args.champion_tolerance
            ),
        }
        decision = {
            "register": all(checks.values()),
            "checks": checks,
            "candidate_auc": m["auc"],
            "champion": champ,
            "thresholds": {
                "min_auc": args.min_auc,
                "max_ece": args.max_ece,
                "champion_tolerance": args.champion_tolerance,
            },
            "backtest_warning": bool(m["backtest_red_grades"] > 0),
        }

        # Persist outputs (consumed by register step, model card and the app)
        write_json(m, out / "metrics.json")
        write_json(decision, out / "decision.json")
        write_json(fair_summary, out / "fairness_summary.json")
        for name, df in {
            "backtest": bt,
            "deciles": dec,
            "fairness": fair,
            "importance": imp,
            "cost_curve": curve,
            "calibration_table": metrics.calibration_table(y, p),
            "reason_frequency": reason_freq,
        }.items():
            df.to_csv(out / f"{name}.csv", index=False)
        score_frame(pkg, test).to_csv(out / "test_scored.csv", index=False)

        plots.save(plots.roc_pr(y, p, m), plot_dir / "roc_pr.png")
        plots.save(
            plots.calibration(
                {"Calibrated": metrics.calibration_table(y, p), "Uncalibrated": metrics.calibration_table(y, p_uncal)},
                f"ECE {m['ece']:.4f} calibrated vs {m['uncalibrated_ece']:.4f} uncalibrated",
            ),
            plot_dir / "calibration.png",
        )
        plots.save(plots.score_distribution(y, p, t), plot_dir / "ks_chart.png")
        plots.save(plots.cost_curve(curve, t, a.cost_fn, a.cost_fp), plot_dir / "cost_curve.png")
        plots.save(plots.confusion(tm), plot_dir / "confusion_matrix.png")
        plots.save(plots.deciles(dec), plot_dir / "deciles.png")
        plots.save(plots.backtest(bt), plot_dir / "backtest.png")
        plots.save(plots.importance(imp), plot_dir / "shap_importance.png")
        plots.save(plots.fairness(fair, t), plot_dir / "fairness.png")
        sample = np.random.default_rng(args.random_state).choice(len(X), size=min(2000, len(X)), replace=False)
        bees = plots.shap_beeswarm(contrib.iloc[sample], feats.iloc[sample])
        if bees is not None:
            plots.save(bees, plot_dir / "shap_beeswarm.png")
        for label, idx in [("high_risk", int(np.argmax(p))), ("median_risk", int(np.argsort(p)[len(p) // 2]))]:
            title = f"Why PD = {p[idx]:.1%} for client {int(test.iloc[idx]['ID'])} ({label.replace('_', ' ')})"
            plots.save(
                plots.waterfall(contrib.iloc[idx], feats.iloc[idx], float(base_values[idx]), title),
                plot_dir / f"waterfall_{label}.png",
            )

        log_metrics(m, prefix="test_")
        log_metrics({"register_decision": decision["register"], "champion_auc": champ.get("auc", float("nan"))})
        mlflow.log_artifacts(str(out), artifact_path="evaluation")
        log_table(bt, "evaluation/backtest_table.csv")
        print(
            f"Test AUC {m['auc']:.4f} [{m['auc_ci_low']:.4f}, {m['auc_ci_high']:.4f}] | Gini {m['gini']:.4f} | KS {m['ks']:.4f} | "
            f"ECE {m['ece']:.4f} | champion: {champ['status']} | register: {decision['register']}"
        )


if __name__ == "__main__":
    main()
