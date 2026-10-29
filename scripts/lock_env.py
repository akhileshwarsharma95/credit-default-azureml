"""Regenerate the pinned Azure ML environments for Python 3.10 / linux-64.

Resolves the top-level requirements with pip (no install) for the Azure base
image's platform and writes fully pinned conda files, so cloud builds install
exactly the tested package set.

    python scripts/lock_env.py            # environments/conda.yaml (+ train-lock.txt) and serving.yaml
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64", "linux_x86_64"]


def resolve(requirements: Path) -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet", "--report", str(report),
               "--python-version", "3.10", "--implementation", "cp", "--abi", "cp310", "--only-binary=:all:",
               "--target", str(Path(tmp) / "t"), "-r", str(requirements)]
        for p in PLATFORMS:
            cmd += ["--platform", p]
        subprocess.run(cmd, check=True)
        data = json.loads(report.read_text())
    return sorted((i["metadata"]["name"].lower(), i["metadata"]["version"]) for i in data["install"])


def write_conda(pins, path: Path, name: str, header: str) -> None:
    body = "\n".join(f"      - {n}=={v}" for n, v in pins)
    path.write_text(f"{header}\nname: {name}\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.10\n  - pip=24.2\n  - pip:\n{body}\n")
    print(f"{path.relative_to(ROOT)}: {len(pins)} packages")


def main() -> None:
    env = ROOT / "environments"
    train = resolve(env / "train-requirements.in")
    write_conda(train, env / "conda.yaml", "credit-default-train",
                "# Azure ML job environment (training, evaluation, batch scoring, monitoring).\n"
                "# Fully pinned lock for Python 3.10 / linux-64, resolved from environments/train-requirements.in\n"
                "# and tested end to end with scripts/run_local.py. Regenerate with scripts/lock_env.py.")
    (env / "train-lock.txt").write_text("# pip-format copy of the pinned job environment (same pins as environments/conda.yaml)\n"
                                        + "\n".join(f"{n}=={v}" for n, v in train) + "\n")
    serving = resolve(env / "serving-requirements.in")
    write_conda(serving, env / "serving.yaml", "credit-default-serving",
                "# Online endpoint environment (managed online deployment, custom scoring script).\n"
                "# Fully pinned lock for Python 3.10 / linux-64 from environments/serving-requirements.in.")
    print("Re-run `pytest` and `python scripts/run_local.py` with the new pins before committing.")


if __name__ == "__main__":
    main()
