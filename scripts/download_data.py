"""Download the UCI 'Default of Credit Card Clients' dataset (id 350).

Saves data/raw/default_of_credit_card_clients.xls. If the UCI site is not
reachable, falls back to a public GitHub copy of the same 30,000 rows (CSV).
"""

from __future__ import annotations

import argparse
import http.client
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from creditrisk.observability import get_logger

LOGGER = get_logger("data")

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
MIRROR_URL = "https://raw.githubusercontent.com/PacktPublishing/Ensemble-Machine-Learning-Cookbook/master/Chapter06/UCI_Credit_Card.csv"
ROOT = Path(__file__).resolve().parents[1]


def from_uci(dest: Path) -> Path:
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx")))
        target = dest / ("default_of_credit_card_clients" + Path(name).suffix.lower())
        with z.open(name) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def from_mirror(dest: Path) -> Path:
    target = dest / "UCI_Credit_Card.csv"
    urllib.request.urlretrieve(MIRROR_URL, target)
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default=str(ROOT / "data" / "raw"))
    p.add_argument("--source", choices=["uci", "mirror"], default="uci")
    args = p.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if args.source == "uci":
        try:
            path = from_uci(dest)
        except (OSError, http.client.HTTPException, zipfile.BadZipFile, StopIteration) as exc:  # network or archive
            LOGGER.warning(f"UCI download failed ({exc}); using GitHub mirror")
            path = from_mirror(dest)
    else:
        path = from_mirror(dest)
    LOGGER.info(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
