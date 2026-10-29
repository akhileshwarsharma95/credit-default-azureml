"""Candidate models, hyperparameter search (Optuna) and probability calibration."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import ENGINEERED, MONOTONE_DIRECTION, FeatureEngineer, signed_log1p
from .schema import BILL_AMT, CATEGORICAL, PAY_AMT

log = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 24,
    "min_child_samples": 80,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
}


def cv_splitter(n_splits: int, random_state: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def engineered_columns(input_cols: list[str]) -> list[str]:
    return list(input_cols) + ENGINEERED


def build_lgbm(
    input_cols: list[str], params: dict | None = None, monotone: bool = True, random_state: int = 42
) -> Pipeline:
    cols = engineered_columns(input_cols)
    constraints = [MONOTONE_DIRECTION.get(c, 0) if monotone else 0 for c in cols]
    est = LGBMClassifier(
        objective="binary",
        subsample_freq=1,
        monotone_constraints=constraints,
        monotone_constraints_method="intermediate",
        random_state=random_state,
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        verbose=-1,
        **(params or DEFAULT_LGBM_PARAMS),
    )
    return Pipeline([("features", FeatureEngineer()), ("model", est)])


def build_logreg(input_cols: list[str], C: float = 0.5) -> Pipeline:
    """Interpretable scorecard-style benchmark."""
    cols = engineered_columns(input_cols)
    cats = [c for c in CATEGORICAL if c in cols]
    money = [c for c in cols if c in {"LIMIT_BAL", *BILL_AMT, *PAY_AMT}]
    other = [c for c in cols if c not in cats and c not in money]
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            (
                "money",
                Pipeline(
                    [
                        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                money,
            ),
            ("num", StandardScaler(), other),
        ]
    )
    return Pipeline(
        [("features", FeatureEngineer()), ("prep", prep), ("model", LogisticRegression(C=C, max_iter=5000))]
    )


def tune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    input_cols: list[str],
    n_trials: int,
    cv_folds: int,
    random_state: int,
    monotone: bool,
) -> tuple[dict, pd.DataFrame]:
    """TPE search maximising mean cross-validated ROC AUC."""
    if n_trials <= 0:
        return dict(DEFAULT_LGBM_PARAMS), pd.DataFrame()
    import optuna  # training-only dependency; keeps the serving image lean

    cv = cv_splitter(cv_folds, random_state)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
        model = build_lgbm(input_cols, params, monotone, random_state)
        return float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lgbm_pd")
    study.enqueue_trial(DEFAULT_LGBM_PARAMS)  # the default config is always evaluated
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    log.info("Optuna best CV AUC %.4f with %s", study.best_value, study.best_params)
    return dict(study.best_params), trials


def oof_proba(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=cv_splitter(cv_folds, random_state), method="predict_proba")[:, 1]


def calibrate(model: Pipeline, method: str, cv_folds: int, random_state: int):
    """Wrap in CalibratedClassifierCV(ensemble=False): one base model fitted on all
    training data plus a calibrator fitted on its out-of-fold scores."""
    if method == "none":
        return model
    return CalibratedClassifierCV(model, method=method, cv=cv_splitter(cv_folds, random_state), ensemble=False)


def oof_calibrated_pd(
    final,
    base_unfitted: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    oof_probability: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    """Out-of-sample PDs for the training population: out-of-fold base scores
    passed through the fitted calibrator. Used as the drift baseline, because
    in-sample scores of a boosted model are more extreme than production scores."""
    if not isinstance(final, CalibratedClassifierCV):
        return oof_probability
    if hasattr(base_unfitted, "decision_function"):  # calibrator was fitted on decision scores
        scores = cross_val_predict(
            base_unfitted, X, y, cv=cv_splitter(cv_folds, random_state), method="decision_function"
        )
    else:
        scores = oof_probability
    return np.clip(final.calibrated_classifiers_[0].calibrators[0].predict(scores), 0.0, 1.0)


class FlooredPD(BaseEstimator, ClassifierMixin):
    """Wraps a fitted classifier and floors/caps the PD (no PD of exactly 0 or 1)."""

    def __init__(self, estimator=None, floor: float = 0.001):
        self.estimator = estimator
        self.floor = floor

    @classmethod
    def wrap(cls, fitted, floor: float) -> FlooredPD:
        obj = cls(fitted, floor)
        obj.classes_ = fitted.classes_
        return obj

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        p = np.clip(self.estimator.predict_proba(X)[:, 1], self.floor, 1 - self.floor)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def unwrap_base(model):
    """The fitted (uncalibrated) sklearn pipeline inside a calibrated model."""
    if isinstance(model, FlooredPD):
        model = model.estimator
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model
