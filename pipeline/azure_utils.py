"""Azure ML helpers shared by the pipeline, deployment and monitoring scripts."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "components"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()  # before DEFAULTS, so overrides such as AZUREML_VM_SIZE in .env take effect

DEFAULTS = {
    "compute": os.getenv("AZUREML_COMPUTE", "cpu-cluster"),
    "vm_size": os.getenv("AZUREML_VM_SIZE", "Standard_DS11_v2"),
    "max_nodes": int(os.getenv("AZUREML_MAX_NODES", "2")),
    "experiment": os.getenv("AZUREML_EXPERIMENT", "credit-default"),
    "model_name": os.getenv("AZUREML_MODEL_NAME", "credit-default-model"),
    "raw_asset": "credit-default-raw",
}


def get_ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    load_dotenv()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    sub, rg, ws = (os.getenv(k) for k in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZUREML_WORKSPACE"))
    if sub and rg and ws:
        return MLClient(credential, sub, rg, ws)
    return MLClient.from_config(credential=credential)  # falls back to config.json from the portal


def ensure_compute(ml_client, name: str, vm_size: str, max_nodes: int):
    """Create the compute cluster if it does not exist. Scales to zero when idle."""
    from azure.ai.ml.entities import AmlCompute
    from azure.core.exceptions import ResourceNotFoundError

    try:
        cluster = ml_client.compute.get(name)
        print(f"Compute cluster '{name}' exists ({cluster.size}, max {cluster.max_instances} nodes)")
        return cluster
    except ResourceNotFoundError:
        pass
    print(f"Creating compute cluster '{name}' ({vm_size}, 0-{max_nodes} nodes, scale-down after 3 min idle) ...")
    cluster = AmlCompute(
        name=name,
        size=vm_size,
        min_instances=0,
        max_instances=max_nodes,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    return ml_client.compute.begin_create_or_update(cluster).result()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data_asset(ml_client, name: str, path: Path, description: str, extra_tags: dict | None = None):
    """Register a file as a versioned uri_file data asset; reuse the version if the content hash is unchanged."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data

    digest = sha256(path)
    existing = []
    try:
        existing = list(ml_client.data.list(name=name))
    except Exception:  # noqa: BLE001 - first registration
        existing = []
    for asset in existing:
        if (asset.tags or {}).get("sha256") == digest:
            print(f"Data asset {name}:{asset.version} already holds this file (sha256 {digest[:12]})")
            return asset
    numeric = [int(a.version) for a in existing if str(a.version).isdigit()]
    version = str(max(numeric) + 1) if numeric else "1"
    asset = Data(
        name=name,
        version=version,
        type=AssetTypes.URI_FILE,
        path=str(path),
        description=description,
        tags={"sha256": digest, "file": path.name, **(extra_tags or {})},
    )
    asset = ml_client.data.create_or_update(asset)
    print(f"Registered data asset {name}:{asset.version}")
    return asset


def resolve_model(ml_client, name: str, version: str = "latest"):
    if version != "latest":
        return ml_client.models.get(name, version)
    versions = list(ml_client.models.list(name=name))
    if not versions:
        raise SystemExit(f"No registered versions of model '{name}' - run the training pipeline first")
    return max(versions, key=lambda m: int(m.version))


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def stream_and_check(ml_client, job) -> str:
    ml_client.jobs.stream(job.name)
    status = ml_client.jobs.get(job.name).status
    print(f"Job {job.name} finished with status {status}")
    if status != "Completed":
        raise SystemExit(1)
    return status
