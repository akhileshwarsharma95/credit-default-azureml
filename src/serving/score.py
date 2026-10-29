"""Online scoring script for the Azure ML managed online endpoint.

Request (JSON):
    {"clients": [{"ID": 1, "LIMIT_BAL": 50000, "EDUCATION": 2, "PAY_1": 0, ..., "PAY_AMT6": 800}]}
    (the Azure ML MLflow format {"input_data": {"columns": [...], "data": [[...]]}} is also accepted)

Response: per client PD, rating grade, recommended action, exposure, expected
loss, IFRS 9 stage, early-warning triggers, limit advice and reason codes.

Observability: inputs/outputs go to the Azure ML data collector (model
monitoring), structured logs and timings go to Application Insights.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# The endpoint runs from the uploaded src/ folder, where creditrisk is not installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PKG = None
INPUTS = None
OUTPUTS = None
LOGGER = None
MODEL_VERSION = "unknown"
OUTPUT_COLUMNS = [
    "pd",
    "grade",
    "action",
    "flag",
    "ead",
    "el",
    "ifrs9_stage",
    "ecl",
    "ews_triggers",
    "limit_action",
    "recommended_limit",
]


def init():
    global PKG, INPUTS, OUTPUTS, LOGGER, MODEL_VERSION
    from creditrisk.observability import get_logger, log_event
    from creditrisk.scoring import load_package

    LOGGER = get_logger("endpoint")
    model_dir = os.getenv(
        "AZUREML_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "local" / "model_package")
    )
    m = re.search(r"/([^/]+)/(\d+)/?$", model_dir.rstrip("/") + "/")
    MODEL_VERSION = f"{m.group(1)}:{m.group(2)}" if m else "local"
    PKG = load_package(model_dir)
    try:
        from azureml.ai.monitoring import Collector

        INPUTS = Collector(name="model_inputs", on_error=lambda e: LOGGER.warning("collector error: %s", e))
        OUTPUTS = Collector(name="model_outputs", on_error=lambda e: LOGGER.warning("collector error: %s", e))
    except Exception as exc:  # noqa: BLE001 - data collection is optional outside Azure
        LOGGER.info("Data collector not active: %s", exc)
    log_event(
        LOGGER, "model_loaded", model_version=MODEL_VERSION, family=PKG.summary["model_family"], threshold=PKG.threshold
    )


def _bad_request(message: str, **extra):
    body = {"error": message, **extra}
    try:
        from azureml_inference_server_http.api.aml_response import AMLResponse

        return AMLResponse(body, 400, json_str=True)
    except ImportError:  # running outside the inference server (tests)
        return body


def _parse(raw_data) -> pd.DataFrame:
    payload = json.loads(raw_data) if isinstance(raw_data, str | bytes) else raw_data
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object with 'clients' or 'input_data'")
    if "input_data" in payload:
        d = payload["input_data"]
        return pd.DataFrame(d["data"], columns=d["columns"])
    records = payload.get("clients") or payload.get("data")
    if records is None:
        raise ValueError("Expected 'clients' (list of objects) or 'input_data' (columns/data)")
    return pd.DataFrame(records if isinstance(records, list) else [records])


def run(raw_data):
    from creditrisk.observability import log_event, timed
    from creditrisk.scoring import score_frame

    start = time.perf_counter()
    try:
        df = _parse(raw_data)
        missing = [c for c in PKG.features if c not in df.columns]
        if missing:
            return _bad_request(f"missing fields: {missing}", required_fields=PKG.features)
        df[PKG.features] = df[PKG.features].apply(pd.to_numeric, errors="raise")
    except (ValueError, KeyError, TypeError) as exc:  # malformed JSON, unknown layout or non-numeric fields
        LOGGER.warning("bad request: %s", exc)
        return _bad_request(f"invalid request: {exc}")

    with timed("endpoint.score", model_version=MODEL_VERSION, n_clients=len(df)):
        context = INPUTS.collect(df[PKG.features]) if INPUTS else None
        scored = score_frame(PKG, df)
        if OUTPUTS:
            OUTPUTS.collect(scored[["pd", "grade", "action"]], context)

    reason_cols = [c for c in scored.columns if re.fullmatch(r"reason_\d", c)]
    predictions = []
    for i, row in scored.reset_index(drop=True).iterrows():
        item = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in OUTPUT_COLUMNS}
        item["pd"] = round(float(item["pd"]), 5)
        if "ID" in df.columns:
            item["ID"] = int(df["ID"].iloc[i])
        item["reason_codes"] = [{"code": row[f"reason_code_{c[-1]}"], "text": row[c]} for c in reason_cols if row[c]]
        predictions.append(item)

    latency = round((time.perf_counter() - start) * 1000, 1)
    log_event(
        LOGGER,
        "scored",
        model_version=MODEL_VERSION,
        n_clients=len(df),
        mean_pd=float(scored["pd"].mean()),
        latency_ms=latency,
    )
    return {
        "model_version": MODEL_VERSION,
        "threshold": PKG.threshold,
        "latency_ms": latency,
        "predictions": predictions,
    }
