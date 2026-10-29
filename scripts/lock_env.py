"""Regenerate the pinned Azure ML environments for Python 3.10 / linux-64.

Resolves the top-level requirements with uv for the Azure base image's platform, constrained by
uv.lock, and writes fully pinned conda files, so cloud builds install exactly the tested package set.

    uv run python scripts/lock_env.py      # environments/conda.yaml (+ train-lock.txt) and serving.yaml
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import tomllib

from creditrisk.observability import get_logger

LOGGER = get_logger("lock_env")

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_NEWER = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["exclude-newer"]


def resolve(requirements: Path, constraints: Path | None = None) -> list[tuple[str, str]]:
    cmd = [
        "uv",
        "pip",
        "compile",
        str(requirements),
        "--exclude-newer",
        EXCLUDE_NEWER,
        "--python-platform",
        "x86_64-manylinux_2_28",
        "--python-version",
        "3.10",
        "--no-header",
        "--no-annotate",
        "--quiet",
    ]
    if constraints:
        cmd += ["--constraint", str(constraints)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return sorted(
        (name.lower(), version) for name, version in (line.split("==") for line in out.splitlines() if "==" in line)
    )


def write_conda(pins, path: Path, name: str, header: str) -> None:
    body = "\n".join(f"      - {n}=={v}" for n, v in pins)
    path.write_text(
        f"{header}\nname: {name}\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.10\n  - pip=24.2\n  - pip:\n{body}\n"
    )
    LOGGER.info(f"{path.relative_to(ROOT)}: {len(pins)} packages")


def main() -> None:
    env = ROOT / "environments"
    # The job environment runs the same code as local and CI, so it is held to uv.lock
    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / "constraints.txt"
        subprocess.run(
            ["uv", "export", "--no-hashes", "--no-emit-project", "--quiet", "-o", str(constraints)],
            check=True,
            cwd=ROOT,
        )
        train = resolve(env / "train-requirements.in", constraints)
    write_conda(
        train,
        env / "conda.yaml",
        "credit-default-train",
        "# Azure ML job environment (training, evaluation, batch scoring, monitoring).\n"
        "# Fully pinned lock for Python 3.10 / linux-64, resolved from environments/train-requirements.in\n"
        "# and tested end to end with scripts/run_local.py. Regenerate with scripts/lock_env.py.",
    )
    (env / "train-lock.txt").write_text(
        "# pip-format copy of the pinned job environment (same pins as environments/conda.yaml)\n"
        + "\n".join(f"{n}=={v}" for n, v in train)
        + "\n"
    )
    serving = resolve(env / "serving-requirements.in")
    write_conda(
        serving,
        env / "serving.yaml",
        "credit-default-serving",
        "# Online endpoint environment (managed online deployment, custom scoring script).\n"
        "# Fully pinned lock for Python 3.10 / linux-64 from environments/serving-requirements.in.",
    )
    LOGGER.info("Re-run `pytest` and `python scripts/run_local.py` with the new pins before committing.")


if __name__ == "__main__":
    main()
