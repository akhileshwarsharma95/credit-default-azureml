"""Business and technical parameters from config.yaml.

The file lives inside the package so it ships with every Azure job, the endpoint and each
registered model (MLflow bundles the creditrisk folder).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


@cache
def load() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())
