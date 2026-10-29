"""Step 1 - Load, validate, clean and split the raw UCI credit default data.

Stateless cleaning happens here (safe before the split). Anything that learns
from data (scaling, encoding, calibration) is fitted later, on training data
only, inside the model pipeline - no leakage into the test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split

from creditrisk import quality
from creditrisk.schema import TARGET, load_raw
from creditrisk.tracking import log_metrics, log_params, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_data", required=True, help="Raw UCI file (.xls/.xlsx/.csv) or folder containing it")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--train_data", required=True, help="Output folder for train.csv")
    p.add_argument("--test_data", required=True, help="Output folder for test.csv")
    p.add_argument("--quality_report", required=True, help="Output folder for the data-quality report")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    with mlflow.start_run():
        log_params({"test_size": args.test_size, "random_state": args.random_state})

        raw = load_raw(args.raw_data)
        df, report = quality.validate(raw)
        out = Path(args.quality_report)
        out.mkdir(parents=True, exist_ok=True)
        if not report.passed:
            write_json(report.to_dict(), out / "quality_report.json")
            (out / "quality_report.md").write_text(quality.report_markdown(report))
            mlflow.log_artifacts(str(out), artifact_path="data_quality")
            raise SystemExit(f"Data-quality gate failed: {report.hard_failures}")

        clean = quality.clean(df, report)
        clean_checked, post = quality.validate(clean)
        if not post.passed or post.soft_failures:
            raise SystemExit(f"Cleaned data still violates the schema: {post.hard_failures + post.soft_failures}")

        train, test = train_test_split(clean_checked, test_size=args.test_size, stratify=clean_checked[TARGET], random_state=args.random_state)
        for frame, folder, name in [(train, args.train_data, "train.csv"), (test, args.test_data, "test.csv")]:
            Path(folder).mkdir(parents=True, exist_ok=True)
            frame.to_csv(Path(folder) / name, index=False)

        summary = report.to_dict() | {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_default_rate": float(train[TARGET].mean()),
            "test_default_rate": float(test[TARGET].mean()),
        }
        write_json(summary, out / "quality_report.json")
        (out / "quality_report.md").write_text(quality.report_markdown(report))
        mlflow.log_artifacts(str(out), artifact_path="data_quality")
        log_metrics(
            {
                "rows_raw": report.rows_in,
                "rows_clean": report.rows_out,
                "dq_soft_violations": sum(f["n_failures"] for f in report.soft_failures),
                **report.actions,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_default_rate": summary["train_default_rate"],
                "test_default_rate": summary["test_default_rate"],
            }
        )
        print(f"Preprocessing done: {len(train):,} train / {len(test):,} test rows, default rate {summary['train_default_rate']:.3f}")


if __name__ == "__main__":
    main()
