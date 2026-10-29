"""Download a registered model version (full package: model, drift reference,
evaluation evidence, model card) from the Azure ML registry, e.g. for the app.

    python scripts/fetch_model.py --output app/model              # latest version
    python scripts/fetch_model.py --version 3 --output app/model
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from creditrisk.observability import get_logger
from creditrisk.scoring import find_model_dir
from pipeline.azure_utils import DEFAULTS, get_ml_client, resolve_model

LOGGER = get_logger("fetch_model")

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default=DEFAULTS["model_name"])
    p.add_argument("--version", default="latest")
    p.add_argument("--output", default=str(ROOT / "app" / "model"))
    args = p.parse_args()

    ml_client = get_ml_client()
    model = resolve_model(ml_client, args.model_name, args.version)
    out = Path(args.output)
    with tempfile.TemporaryDirectory() as tmp:
        ml_client.models.download(name=model.name, version=model.version, download_path=tmp)
        src = find_model_dir(tmp)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(src, out)
    (out / "REGISTRY_VERSION").write_text(f"{model.name}:{model.version}\n")
    LOGGER.info(f"Downloaded {model.name}:{model.version} to {out}")


if __name__ == "__main__":
    main()
