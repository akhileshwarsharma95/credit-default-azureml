"""Deploy the registered model to an Azure ML managed online endpoint (blue/green).

    python deploy/deploy_endpoint.py                      # latest model -> new deployment, smoke test, switch traffic
    python deploy/deploy_endpoint.py --canary 10          # new deployment receives 10% of traffic first
    python deploy/deploy_endpoint.py --rollback           # send 100% back to the previous deployment
    python deploy/deploy_endpoint.py --delete             # delete the endpoint (stops billing)
    python deploy/deploy_endpoint.py --dry-run            # build the deployment spec locally, no Azure calls

The deployment uses the custom scoring script (src/serving/score.py), so the
response carries PD, grade, action, expected loss and reason codes. Application
Insights logging and the Azure ML data collector (inputs/outputs for model
monitoring) are switched on. Billing is per instance-hour while it exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from pipeline.azure_utils import DEFAULTS, get_ml_client, resolve_model

BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"


def default_endpoint_name() -> str:
    """Endpoint names must be unique per Azure region, so add a short stable suffix."""
    seed = os.getenv("AZURE_SUBSCRIPTION_ID", "") + os.getenv("AZUREML_WORKSPACE", "")
    return os.getenv("AZUREML_ENDPOINT", f"credit-risk-{hashlib.sha1(seed.encode()).hexdigest()[:6]}")


def build_deployment(name: str, endpoint: str, model_id: str, instance_type: str, instance_count: int, collect: bool):
    from azure.ai.ml.entities import (
        CodeConfiguration,
        DataCollector,
        DeploymentCollection,
        Environment,
        ManagedOnlineDeployment,
        OnlineRequestSettings,
    )

    env = Environment(
        name="credit-default-serving",
        image=BASE_IMAGE,
        conda_file=str(ROOT / "environments" / "serving.yaml"),
        description="Pinned serving environment for the credit default endpoint",
    )
    collector = (
        DataCollector(
            collections={
                "model_inputs": DeploymentCollection(enabled="true"),
                "model_outputs": DeploymentCollection(enabled="true"),
            },
            sampling_rate=1.0,
        )
        if collect
        else None
    )
    return ManagedOnlineDeployment(
        name=name,
        endpoint_name=endpoint,
        model=model_id,
        environment=env,
        code_configuration=CodeConfiguration(code=str(ROOT / "src"), scoring_script="serving/score.py"),
        instance_type=instance_type,
        instance_count=instance_count,
        app_insights_enabled=True,
        data_collector=collector,
        request_settings=OnlineRequestSettings(request_timeout_ms=10_000, max_concurrent_requests_per_instance=4),
        environment_variables={"LOG_LEVEL": "INFO"},
        tags={"model": model_id.split("/models/")[-1] if "/models/" in model_id else model_id},
    )


def smoke_test(ml_client, endpoint: str, deployment: str) -> None:
    raw = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint, deployment_name=deployment, request_file=str(ROOT / "deploy" / "sample_request.json")
    )
    body = json.loads(raw)
    if isinstance(body, str):
        body = json.loads(body)
    preds = body.get("predictions", [])
    expected = len(json.loads((ROOT / "deploy" / "sample_request.json").read_text())["clients"])
    if len(preds) != expected or not all(0 < p["pd"] < 1 for p in preds):
        raise SystemExit(f"Smoke test failed: {body}")
    print("Smoke test passed:", [(p.get("ID"), p["pd"], p["grade"], p["action"]) for p in preds])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint-name", default=default_endpoint_name())
    p.add_argument("--model-name", default=DEFAULTS["model_name"])
    p.add_argument("--model-version", default="latest")
    p.add_argument("--instance-type", default=os.getenv("AZUREML_ENDPOINT_SKU", "Standard_DS2_v2"))
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument(
        "--canary",
        type=int,
        default=0,
        help="Percent of traffic for the new deployment (0 = full switch after smoke test)",
    )
    p.add_argument("--keep-old", action="store_true", help="Keep the previous deployment after switching traffic")
    p.add_argument("--no-data-collection", action="store_true")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--delete", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        dep = build_deployment(
            "blue",
            args.endpoint_name,
            f"azureml:{args.model_name}:1",
            args.instance_type,
            args.instance_count,
            not args.no_data_collection,
        )
        with tempfile.TemporaryDirectory() as tmp:
            dep.dump(Path(tmp) / "deployment.yml")
            print((Path(tmp) / "deployment.yml").read_text())
        print(f"Dry run OK: deployment spec for endpoint '{args.endpoint_name}'")
        return

    from azure.ai.ml.entities import ManagedOnlineEndpoint
    from azure.core.exceptions import ResourceNotFoundError

    ml_client = get_ml_client()
    name = args.endpoint_name

    if args.delete:
        ml_client.online_endpoints.begin_delete(name=name).result()
        print(f"Endpoint {name} deleted")
        return

    try:
        endpoint = ml_client.online_endpoints.get(name)
    except ResourceNotFoundError:
        if args.rollback:
            raise SystemExit("Nothing to roll back") from None
        print(f"Creating endpoint {name} ...")
        endpoint = ml_client.online_endpoints.begin_create_or_update(
            ManagedOnlineEndpoint(
                name=name,
                auth_mode="key",
                description="Credit default PD scoring with reason codes",
                tags={"app": "credit-risk"},
            )
        ).result()

    traffic = dict(endpoint.traffic or {})
    existing = [d.name for d in ml_client.online_deployments.list(endpoint_name=name)]
    live = max(traffic, key=traffic.get) if traffic and max(traffic.values()) > 0 else None

    if args.rollback:
        others = [d for d in existing if d != live]
        if not others:
            raise SystemExit("No previous deployment to roll back to")
        endpoint.traffic = {others[0]: 100, **{d: 0 for d in existing if d != others[0]}}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Rolled back: 100% traffic -> {others[0]}")
        return

    model = resolve_model(ml_client, args.model_name, args.model_version)
    new = "green" if live == "blue" else "blue"
    if new in existing and new != live:
        ml_client.online_deployments.begin_delete(name=new, endpoint_name=name).result()
    print(
        f"Deploying {model.name}:{model.version} as '{new}' on {args.instance_type} x{args.instance_count} (takes ~10-15 min) ..."
    )
    dep = build_deployment(new, name, model.id, args.instance_type, args.instance_count, not args.no_data_collection)
    ml_client.online_deployments.begin_create_or_update(dep).result()

    smoke_test(ml_client, name, new)

    if live and args.canary > 0:
        endpoint.traffic = {live: 100 - args.canary, new: args.canary}
        print(f"Canary: {args.canary}% -> {new}, {100 - args.canary}% -> {live}. Re-run with --canary 0 to complete.")
    else:
        endpoint.traffic = {new: 100, **({live: 0} if live else {})}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    if live and args.canary == 0 and not args.keep_old:
        ml_client.online_deployments.begin_delete(name=live, endpoint_name=name).result()
        print(f"Deleted previous deployment '{live}'")

    endpoint = ml_client.online_endpoints.get(name)
    print(f"Endpoint ready: {endpoint.scoring_uri}\nTraffic: {endpoint.traffic}")
    print(f"Key: az ml online-endpoint get-credentials -n {name} -g $AZURE_RESOURCE_GROUP -w $AZUREML_WORKSPACE")


if __name__ == "__main__":
    main()
