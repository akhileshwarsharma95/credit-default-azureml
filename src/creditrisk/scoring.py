"""One scoring function for batch, endpoint and app: PD -> grade -> action,
reason codes, exposure, expected loss, IFRS 9 stage, early warnings, limit advice."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import explain, policy
from .schema import ID_COL, TARGET


@dataclass
class ModelPackage:
    model: object
    summary: dict
    path: Path
    profile: dict = field(default_factory=dict)
    background: pd.DataFrame | None = None

    @property
    def features(self) -> list[str]:
        return self.summary["features"]

    @property
    def threshold(self) -> float:
        return float(self.summary["threshold"])

    @property
    def assumptions(self) -> policy.Assumptions:
        a = dict(self.summary.get("assumptions", {}))
        if "sicr_grades" in a:
            a["sicr_grades"] = tuple(a["sicr_grades"])
        return policy.Assumptions(**a)

    def read_json(self, relative: str) -> dict:
        p = self.path / relative
        return json.loads(p.read_text()) if p.exists() else {}

    def read_csv(self, relative: str) -> pd.DataFrame | None:
        p = self.path / relative
        return pd.read_csv(p) if p.exists() else None


def find_model_dir(root: str | Path) -> Path:
    """Locate the folder holding the MLmodel file (registry downloads nest it)."""
    root = Path(root)
    if (root / "MLmodel").exists():
        return root
    hits = sorted(root.rglob("MLmodel"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"No MLmodel file under {root}")
    return hits[0].parent


def load_package(root: str | Path) -> ModelPackage:
    import mlflow.sklearn

    path = find_model_dir(root)
    model = mlflow.sklearn.load_model(str(path))
    summary = json.loads((path / "training_summary.json").read_text())
    profile_path = path / "reference" / "reference_profile.json"
    background_path = path / "reference" / "background.csv"
    return ModelPackage(
        model=model,
        summary=summary,
        path=path,
        profile=json.loads(profile_path.read_text()) if profile_path.exists() else {},
        background=pd.read_csv(background_path) if background_path.exists() else None,
    )


def score_frame(
    pkg: ModelPackage,
    df: pd.DataFrame,
    assumptions: policy.Assumptions | None = None,
    threshold: float | None = None,
    top_reasons: int = 4,
    with_explanations: bool = True,
) -> pd.DataFrame:
    a = assumptions or pkg.assumptions
    t = pkg.threshold if threshold is None else threshold
    X = df[pkg.features]
    pd_values = pkg.model.predict_proba(X)[:, 1]
    grades = policy.assign_grade(pd_values)
    feats = explain.feature_frame(pkg.model, X)

    out = pd.DataFrame(index=df.index)
    if ID_COL in df.columns:
        out[ID_COL] = df[ID_COL].to_numpy()
    out = pd.concat([out, feats], axis=1)
    out["pd"] = pd_values
    out["grade"] = grades
    out["flag"] = (pd_values >= t).astype(int)

    ews = policy.early_warning(feats, grades)
    out = pd.concat([out, ews], axis=1)
    out["action"] = policy.action_label(pd_values, t, ews["ews_count"])

    out["ead"] = policy.exposure_at_default(df, a.ccf).to_numpy()
    out["el"] = pd_values * a.lgd * out["ead"]
    out["ifrs9_stage"] = policy.ifrs9_stage(df, grades, a.sicr_grades)
    out["ecl"] = policy.expected_credit_loss(pd_values, a.lgd, out["ead"], out["ifrs9_stage"], a.lifetime_years)
    out = pd.concat([out, policy.limit_recommendation(df, pd_values, grades, a.lgd, a.ccf)], axis=1)

    if with_explanations:
        contrib, _ = explain.contributions(pkg.model, X, pkg.background)
        reasons = explain.reason_codes(contrib, feats, top_n=top_reasons, medians=pkg.profile.get("medians"))
        for k in range(top_reasons):
            out[f"reason_{k + 1}"] = [r[k]["text"] if len(r) > k else "" for r in reasons]
            out[f"reason_code_{k + 1}"] = [r[k]["code"] if len(r) > k else "" for r in reasons]

    out[TARGET] = df[TARGET].to_numpy() if TARGET in df.columns else np.nan
    for col in ("SEX", "MARRIAGE", "AGE"):
        if col in df.columns and col not in out.columns:
            out[col] = df[col].to_numpy()
    return out
