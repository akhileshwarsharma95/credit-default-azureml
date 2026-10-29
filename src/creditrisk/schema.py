"""Column names, raw-file loading and human-readable feature descriptions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "default"
ID_COL = "ID"

MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]  # index 1..6 (most recent first)
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]

SENSITIVE = ["SEX", "MARRIAGE", "AGE"]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_STATUS, *BILL_AMT, *PAY_AMT]

# Names used by different copies of the UCI file for the target column.
_TARGET_ALIASES = {
    "default payment next month",
    "default.payment.next.month",
    "default_payment_next_month",
    "y",
    "default",
}

EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}
SEX_LABELS = {1: "Male", 2: "Female"}
PAY_STATUS_LABELS = {
    -2: "No consumption",
    -1: "Paid in full",
    0: "Revolving (min. paid)",
    **{k: f"{k} month{'s' if k > 1 else ''} late" for k in range(1, 10)},
}

FEATURE_DESCRIPTIONS = {
    "LIMIT_BAL": "Credit limit",
    "SEX": "Sex",
    "EDUCATION": "Education level",
    "MARRIAGE": "Marital status",
    "AGE": "Age",
    **{f"PAY_{i}": f"Repayment status in {m}" for i, m in enumerate(MONTHS, start=1)},
    **{f"BILL_AMT{i}": f"Bill amount in {m}" for i, m in enumerate(MONTHS, start=1)},
    **{f"PAY_AMT{i}": f"Amount paid in {m}" for i, m in enumerate(MONTHS, start=1)},
    "util_1": "Credit utilisation (latest bill / limit)",
    "util_avg": "Average 6-month utilisation",
    "util_max": "Peak 6-month utilisation",
    "bill_trend": "Balance growth over 6 months (share of limit)",
    "pay_ratio_1": "Latest repayment as share of previous bill",
    "pay_ratio_avg": "Average repayment ratio (paid / billed)",
    "n_delays": "Months with a late payment (last 6)",
    "n_delays_recent3": "Months with a late payment (last 3)",
    "max_delay": "Longest payment delay (months)",
    "delay_trend": "Delinquency trend (latest vs. 3 months earlier)",
    "n_no_consumption": "Months without card usage",
    "n_paid_full": "Months paid in full",
    "months_since_delay": "Months since most recent late payment",
}


def describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature)


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        name = str(col).strip()
        if name.lower() in _TARGET_ALIASES:
            rename[col] = TARGET
        elif name.upper() == "PAY_0":  # the UCI file skips PAY_1; align with BILL_AMT1/PAY_AMT1
            rename[col] = "PAY_1"
        elif name.upper() == "ID":
            rename[col] = ID_COL
        else:
            rename[col] = name.upper() if name.upper() in RAW_FEATURES else name
    return df.rename(columns=rename)


def _read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.suffix.lower() in {".xls", ".xlsx", ".csv"})
        if not candidates:
            raise FileNotFoundError(f"No .xls/.xlsx/.csv file in {path}")
        path = candidates[0]

    suffix = path.suffix.lower()
    reader = (
        (lambda **kw: pd.read_excel(path, **kw))
        if suffix in {".xls", ".xlsx"}
        else (lambda **kw: pd.read_csv(path, **kw))
    )
    df = reader()
    first_cols = {str(c).strip().upper() for c in df.columns}
    if {"X1", "X2", "Y"}.issubset(first_cols):
        df = reader(header=1)

    df = _standardise_columns(df)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, range(1, len(df) + 1))
    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load any common copy of the UCI file (.xls/.xlsx with the 2-row header, or .csv).

    The official UCI workbook has a first header row of X1..X23/Y and the real
    names in the second row; we detect that and re-read with ``header=1``.
    """
    df = _read_any(path)
    missing = [c for c in [*RAW_FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    return df[[ID_COL, *RAW_FEATURES, TARGET]]


def load_batch(path: str | Path) -> pd.DataFrame:
    """Scoring input: raw feature columns; ID and the target are optional."""
    df = _read_any(path)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Scoring file is missing expected columns: {missing}")
    keep = [ID_COL, *RAW_FEATURES] + ([TARGET] if TARGET in df.columns else [])
    return df[keep]


def read_table(path: str | Path, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV from a file path or from a folder (Azure ML uri_folder outputs)."""
    path = Path(path)
    if path.is_dir():
        if filename and (path / filename).exists():
            path = path / filename
        else:
            csvs = sorted(path.glob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No CSV in {path}")
            path = csvs[0]
    return pd.read_csv(path)


def model_features(exclude_sensitive: bool) -> list[str]:
    return [c for c in RAW_FEATURES if not (exclude_sensitive and c in SENSITIVE)]
