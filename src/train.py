"""Step 2 - Train, select and calibrate the PD model.

1. Tune LightGBM (with monotonic constraints) with Optuna on cross-validated AUC.
2. Compare candidates out-of-fold: LightGBM vs. a logistic-regression scorecard
   (plus an unconstrained LightGBM to quantify the cost of monotonicity).
3. Refit the winner with probability calibration (CalibratedClassifierCV).
4. Derive the cost-optimal intervention cut-off from the cost assumptions.
5. Save one self-contained MLflow model: features + model + calibration, plus
   the drift reference profile and an explanation background sample.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import lightgbm
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import scipy
import sklearn
from mlflow.models import infer_signature

import creditrisk
from creditrisk import drift, explain, metrics, modeling, policy
from creditrisk.config import load
from creditrisk.schema import TARGET, model_features, read_table
from creditrisk.tracking import log_metrics, log_params, log_table, write_json

TRAINING = load()["technical"]["training"]


def str2bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_data", required=True)
    p.add_argument("--model_output", required=True)
    p.add_argument(
        "--exclude_sensitive",
        type=str2bool,
        default=TRAINING["exclude_sensitive"],
        help="Drop SEX, MARRIAGE, AGE from the model",
    )
    p.add_argument(
        "--monotone",
        type=str2bool,
        default=TRAINING["monotone"],
        help="Monotonic constraints on delinquency/limit features",
    )
    p.add_argument(
        "--n_trials", type=int, default=TRAINING["n_trials"], help="Optuna trials (0 = default hyperparameters)"
    )
    p.add_argument("--tuning_folds", type=int, default=TRAINING["tuning_folds"])
    p.add_argument("--cv_folds", type=int, default=TRAINING["cv_folds"])
    p.add_argument("--calibration", choices=["isotonic", "sigmoid", "none"], default=TRAINING["calibration"])
    p.add_argument(
        "--cost_fn", type=float, default=policy.Assumptions.cost_fn, help="Relative cost of a missed default"
    )
    p.add_argument("--cost_fp", type=float, default=policy.Assumptions.cost_fp, help="Relative cost of a false alarm")
    p.add_argument("--lgd", type=float, default=policy.Assumptions.lgd)
    p.add_argument("--ccf", type=float, default=policy.Assumptions.ccf)
    p.add_argument("--random_state", type=int, default=load()["technical"]["data"]["random_state"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    with mlflow.start_run():
        train = read_table(args.train_data, "train.csv")
        features = model_features(args.exclude_sensitive)
        X, y = train[features], train[TARGET].astype(int)
        log_params({k: v for k, v in vars(args).items() if k not in {"train_data", "model_output"}})
        mlflow.set_tags({"stage": "train", "model_type": "pd_scorecard"})

        # 1. Hyperparameter search
        best_params, trials = modeling.tune_lgbm(
            X, y, features, args.n_trials, args.tuning_folds, args.random_state, args.monotone
        )
        if not trials.empty:
            log_table(trials, "tuning/optuna_trials.csv")
            mlflow.log_metric("optuna_best_cv_auc", float(trials["value"].max()))
        log_params({f"lgbm_{k}": v for k, v in best_params.items()})

        # 2. Out-of-fold candidate comparison
        lgbm_name = "lightgbm_monotone" if args.monotone else "lightgbm"
        selectable = {
            lgbm_name: modeling.build_lgbm(features, best_params, args.monotone, args.random_state),
            "logistic_scorecard": modeling.build_logreg(features),
        }
        sensitivity = (
            {"lightgbm_unconstrained": modeling.build_lgbm(features, best_params, False, args.random_state)}
            if args.monotone
            else {}
        )
        rows, oof = [], {}
        for name, model in {**selectable, **sensitivity}.items():
            oof[name] = modeling.oof_proba(model, X, y, args.cv_folds, args.random_state)
            d = metrics.discrimination(y, oof[name])
            rows.append(
                {
                    "candidate": name,
                    "selectable": name in selectable,
                    **d,
                    "brier": float(np.mean((oof[name] - y) ** 2)),
                }
            )
            log_metrics(d, prefix=f"cv_{name}_")
        comparison = pd.DataFrame(rows).sort_values("auc", ascending=False)
        log_table(comparison, "model_selection/candidate_comparison.csv")
        best_name = comparison[comparison["selectable"]].iloc[0]["candidate"]
        print(comparison.to_string(index=False))
        print(f"Selected: {best_name}")

        # 3. Calibrated refit on the full training set
        calibrated = modeling.calibrate(selectable[best_name], args.calibration, args.cv_folds, args.random_state).fit(
            X, y
        )
        final = modeling.FlooredPD.wrap(calibrated, policy.PD_FLOOR)
        pd_train = np.clip(
            modeling.oof_calibrated_pd(
                calibrated, selectable[best_name], X, y, oof[best_name], args.cv_folds, args.random_state
            ),
            policy.PD_FLOOR,
            1 - policy.PD_FLOOR,
        )

        # 4. Cut-off from cost assumptions (valid because PDs are calibrated)
        threshold = policy.cost_optimal_threshold(args.cost_fn, args.cost_fp)
        oof_cost = metrics.cost_curve(y, oof[best_name], args.cost_fn, args.cost_fp)
        empirical = float(oof_cost.loc[oof_cost["expected_cost_per_client"].idxmin(), "threshold"])
        assumptions = policy.Assumptions(lgd=args.lgd, ccf=args.ccf, cost_fn=args.cost_fn, cost_fp=args.cost_fp)

        # 5. Package: model + drift reference + explanation background + summary
        out = Path(args.model_output)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"Model output folder {out} is not empty")
        sample = X.head(200).astype(float)  # double schema: tolerant of ints and missing values at serving time
        pip_reqs = [
            f"scikit-learn=={sklearn.__version__}",
            f"lightgbm=={lightgbm.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            f"scipy=={scipy.__version__}",
            f"mlflow=={mlflow.__version__}",
        ]
        mlflow.sklearn.save_model(
            final,
            path=str(out),
            signature=infer_signature(sample, final.predict_proba(sample)),
            input_example=X.head(3).astype(float),
            code_paths=[str(Path(creditrisk.__file__).parent)],
            pip_requirements=pip_reqs,
            pyfunc_predict_fn="predict_proba",
            metadata={"model_family": best_name, "threshold": threshold, "calibration": args.calibration},
        )
        feats = explain.feature_frame(final, X)
        write_json(drift.build_profile(feats, pd_train), out / "reference" / "reference_profile.json")
        X.sample(n=min(500, len(X)), random_state=args.random_state).to_csv(
            out / "reference" / "background.csv", index=False
        )

        selected = comparison.set_index("candidate").loc[best_name]
        summary = {
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "model_family": best_name,
            "features": features,
            "exclude_sensitive": args.exclude_sensitive,
            "monotone_constraints": args.monotone,
            "calibration": args.calibration,
            "hyperparameters": best_params,
            "candidates": comparison.to_dict(orient="records"),
            "cv_auc": float(selected["auc"]),
            "cv_gini": float(selected["gini"]),
            "monotonicity_auc_cost": float(
                comparison.set_index("candidate").loc["lightgbm_unconstrained", "auc"]
                - comparison.set_index("candidate").loc[lgbm_name, "auc"]
            )
            if sensitivity
            else None,
            "threshold": threshold,
            "threshold_rule": "cost_fp / (cost_fp + cost_fn) on calibrated PD",
            "pd_floor": policy.PD_FLOOR,
            "oof_empirical_cost_minimum": empirical,
            "assumptions": assumptions.to_dict(),
            "train_rows": int(len(X)),
            "train_default_rate": float(y.mean()),
            "train_mean_pd": float(pd_train.mean()),  # out-of-fold calibrated PD
            "library_versions": {r.split("==")[0]: r.split("==")[1] for r in pip_reqs},
            "random_state": args.random_state,
        }
        write_json(summary, out / "training_summary.json")
        mlflow.log_artifact(str(out / "training_summary.json"))
        log_metrics(
            {
                "cv_auc": summary["cv_auc"],
                "cv_gini": summary["cv_gini"],
                "threshold": threshold,
                "oof_empirical_cost_minimum": empirical,
                "train_mean_pd": summary["train_mean_pd"],
                "train_default_rate": summary["train_default_rate"],
                "monotonicity_auc_cost": summary["monotonicity_auc_cost"] or 0.0,
            }
        )
        print(f"Model saved to {out} (CV AUC {summary['cv_auc']:.4f}, cut-off {threshold:.3f})")


if __name__ == "__main__":
    main()
