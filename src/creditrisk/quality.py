"""Data-quality gates (pandera) and deterministic cleaning.

Two layers:
* hard checks: violations stop the pipeline (wrong types, impossible values, nulls)
* soft checks: violations are reported and fixed by documented cleaning rules
  (undocumented category codes, duplicate records)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from .schema import BILL_AMT, ID_COL, PAY_AMT, PAY_STATUS, TARGET


def _amount_columns(checks_bill, checks_pay) -> dict:
    cols = {c: Column(int, checks_bill, coerce=True, nullable=False) for c in BILL_AMT}
    cols.update({c: Column(int, checks_pay, coerce=True, nullable=False) for c in PAY_AMT})
    return cols


def hard_schema(require_target: bool = True) -> DataFrameSchema:
    """Physically impossible or structurally broken data -> fail fast."""
    columns = {
        ID_COL: Column(int, Check.ge(0), coerce=True, nullable=False),
        "LIMIT_BAL": Column(int, Check.in_range(1, 10_000_000), coerce=True, nullable=False),
        "SEX": Column(int, Check.isin([1, 2]), coerce=True, nullable=False),
        "EDUCATION": Column(int, Check.in_range(0, 6), coerce=True, nullable=False),
        "MARRIAGE": Column(int, Check.in_range(0, 3), coerce=True, nullable=False),
        "AGE": Column(int, Check.in_range(18, 100), coerce=True, nullable=False),
        **{c: Column(int, Check.in_range(-2, 9), coerce=True, nullable=False) for c in PAY_STATUS},
        **_amount_columns(Check.in_range(-10_000_000, 10_000_000), Check.in_range(0, 10_000_000)),
    }
    if require_target:
        columns[TARGET] = Column(int, Check.isin([0, 1]), coerce=True, nullable=False)
    return DataFrameSchema(columns, strict=False, name="credit_default_hard")


def soft_schema() -> DataFrameSchema:
    """Documented-domain checks -> violations are cleaned and reported."""
    return DataFrameSchema(
        {
            "EDUCATION": Column(int, Check.isin([1, 2, 3, 4]), coerce=True),
            "MARRIAGE": Column(int, Check.isin([1, 2, 3]), coerce=True),
        },
        strict=False,
        name="credit_default_documented_domain",
    )


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    hard_failures: list = field(default_factory=list)
    soft_failures: list = field(default_factory=list)
    actions: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "hard_failures": self.hard_failures,
            "soft_failures": self.soft_failures,
            "cleaning_actions": self.actions,
        }


def _summarise(err: pa.errors.SchemaErrors) -> list[dict]:
    fc = err.failure_cases
    out = []
    for (col, check), grp in fc.groupby(["column", "check"], dropna=False):
        out.append(
            {
                "column": None if pd.isna(col) else str(col),
                "check": str(check),
                "n_failures": int(len(grp)),
                "examples": [str(v) for v in grp["failure_case"].head(5).tolist()],
            }
        )
    return out


def validate(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, QualityReport]:
    """Run hard + soft checks. Returns the coerced frame and a report (never raises)."""
    report = QualityReport(rows_in=len(df))
    try:
        df = hard_schema(require_target).validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        report.hard_failures = _summarise(err)
        return df, report
    try:
        soft_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        report.soft_failures = _summarise(err)
    return df, report


def clean(df: pd.DataFrame, report: QualityReport, require_target: bool = True) -> pd.DataFrame:
    """Deterministic, stateless cleaning (safe to apply before the train/test split)."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != ID_COL]

    n_dup = int(df.duplicated(subset=feature_cols).sum())
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    report.actions["duplicates_removed"] = n_dup

    # Undocumented codes -> the documented "other" bucket (per UCI data dictionary).
    edu_bad = ~df["EDUCATION"].isin([1, 2, 3, 4])
    mar_bad = ~df["MARRIAGE"].isin([1, 2, 3])
    report.actions["education_remapped_to_other"] = int(edu_bad.sum())
    report.actions["marriage_remapped_to_other"] = int(mar_bad.sum())
    df.loc[edu_bad, "EDUCATION"] = 4
    df.loc[mar_bad, "MARRIAGE"] = 3

    # PAY_x = -2 / 0 are undocumented but meaningful (no consumption / revolving); keep them.
    report.actions["pay_status_minus2_or_0_kept"] = int(df[PAY_STATUS].isin([-2, 0]).any(axis=1).sum())

    if require_target:
        df[TARGET] = df[TARGET].astype(int)
    report.rows_out = len(df)
    return df.reset_index(drop=True)


def clean_for_scoring(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Same code remapping as training, but every row is kept (each row is a client to score)."""
    df = df.copy()
    edu_bad = ~df["EDUCATION"].isin([1, 2, 3, 4])
    mar_bad = ~df["MARRIAGE"].isin([1, 2, 3])
    report.actions["education_remapped_to_other"] = int(edu_bad.sum())
    report.actions["marriage_remapped_to_other"] = int(mar_bad.sum())
    df.loc[edu_bad, "EDUCATION"] = 4
    df.loc[mar_bad, "MARRIAGE"] = 3
    report.rows_out = len(df)
    return df


def report_markdown(report: QualityReport) -> str:
    lines = [
        "# Data quality report",
        "",
        f"- Status: **{'PASSED' if report.passed else 'FAILED'}**",
        f"- Rows in: {report.rows_in:,} | rows out: {report.rows_out:,}",
        "",
        "## Hard checks (pipeline fails on violation)",
    ]
    lines += [f"- `{f['column']}` {f['check']}: {f['n_failures']} rows" for f in report.hard_failures] or ["- none"]
    lines += ["", "## Soft checks (cleaned and reported)"]
    lines += [f"- `{f['column']}` {f['check']}: {f['n_failures']} rows" for f in report.soft_failures] or ["- none"]
    lines += ["", "## Cleaning actions"] + [f"- {k}: {v:,}" for k, v in report.actions.items()]
    return "\n".join(lines) + "\n"
