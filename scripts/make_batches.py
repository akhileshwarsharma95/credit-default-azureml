"""Create two demo scoring batches for the monitoring pipeline.

* portfolio_stable.csv   - the held-out test clients (same deterministic split as
                           preprocess.py), with outcomes, so realised performance
                           can be monitored.
* portfolio_downturn.csv - the same clients after a simulated downturn (more
                           delinquency, higher balances, lower repayments), no
                           outcomes. Should trigger the PSI drift alert.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import train_test_split  # noqa: E402

from creditrisk import drift, quality  # noqa: E402
from creditrisk.schema import TARGET, load_raw  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--dest", default=str(ROOT / "data" / "batches"))
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    df, report = quality.validate(load_raw(args.raw_data))
    df = quality.clean(df, report)
    _, test = train_test_split(df, test_size=args.test_size, stratify=df[TARGET], random_state=args.random_state)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    test.to_csv(dest / "portfolio_stable.csv", index=False)
    drift.simulate_downturn(test, severity=1.0).to_csv(dest / "portfolio_downturn.csv", index=False)
    print(f"Wrote {len(test):,}-client batches to {dest}")


if __name__ == "__main__":
    main()
