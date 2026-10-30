# Azure setup

Everything below fits in the Azure free account ($200 credit for 30 days) or Azure for Students ($100, no card). A full training run costs well under a dollar; the only thing that bills around the clock is the optional online endpoint, so delete it after the demo.

## 1. Azure ML concepts compared with Databricks and AWS

| Concept | Databricks | AWS | Azure ML (this repo) |
|---|---|---|---|
| Workspace | Workspace | SageMaker domain | Azure ML workspace (`mlw-credit-risk`) |
| Tracking | Managed MLflow | Experiments / MLflow | MLflow built in (no server to run) |
| Model registry | Unity Catalog models | Model Registry | Workspace registry, populated through MLflow |
| Job compute | Job cluster | Training instances | Compute cluster `cpu-cluster` (scales 0 to 2 nodes) |
| Pipelines | Workflows | SageMaker Pipelines | Pipeline jobs built from YAML **components** |
| Data | Volumes / tables | S3 | Datastore (Blob) + versioned **data assets** |
| Real-time serving | Model Serving | Endpoints | Managed online endpoint (blue/green) |
| Monitoring | Lakehouse Monitoring | Model Monitor | Azure ML model monitoring + Application Insights |
| Secrets | Secret scopes | Secrets Manager | Key Vault (created with the workspace) |
| Identity | Service principals | IAM roles | Microsoft Entra ID + RBAC; GitHub logs in with OIDC |
| Logs / metrics | - | CloudWatch | Azure Monitor, Log Analytics, Application Insights |
| Container hosting | - | App Runner / Fargate | Azure Container Apps |

The workspace creates its own storage account, key vault and Application Insights; a container registry is added the first time an environment image is built.

## 2. One-time setup (about 15 minutes)

1. **Create the account**: azure.microsoft.com/free (or /free/students). The free account has a spending limit: when the credit is used up, resources stop instead of billing you.
2. **Push this repo to GitHub** (any name, e.g. `credit-default-azureml`).
3. **Open Azure Cloud Shell** (the `>_` icon in portal.azure.com, choose *Bash*). It already has the Azure CLI and is logged in. Then:
   ```bash
   git clone https://github.com/<you>/credit-default-azureml.git && cd credit-default-azureml
   export GITHUB_REPO=<you>/credit-default-azureml      # enables GitHub Actions (optional)
   export AZURE_LOCATION=germanywestcentral              # or westeurope / swedencentral
   bash infra/setup_azure.sh
   ```
   This registers the resource providers, creates `rg-credit-risk` and the workspace, writes `.env`, and (with `GITHUB_REPO`) creates an Entra app with **federated credentials** so GitHub Actions can log in with no stored password. It prints the three secrets and three variables to add in GitHub (or sets them with `gh`).
4. **Check compute quota** in Azure ML studio (ml.azure.com) -> *Compute* -> *Quota*. The cluster uses `Standard_D2ds_v5` (DDSv5 family) and the endpoint `Standard_D2as_v4` (DASv4 family), both 2 vCPU / 8 GB, set in `src/creditrisk/config.yaml`. They are in different families on purpose: an endpoint holds its vCPUs plus a 20% reserve, and on a 4-vCPU family quota it would leave the cluster no node. If a family shows 0, pick one that has quota (`AZUREML_VM_SIZE`, `AZUREML_ENDPOINT_SKU`) or request an increase (usually approved within minutes for small numbers).

## 3. Run the case study

On your laptop (Python 3.10 or 3.11 recommended, same as Azure) after `az login`, or in Cloud Shell:

```bash
make setup                                        # uv sync + official UCI file -> data/raw/
make train                                        # creates cluster + data asset, submits, streams logs
```

The first run builds the pinned environment image (10-15 minutes, once); the four steps then take about 8 minutes on 2 vCPUs. The CLI route does the same with `az ml job create -f pipeline/training_pipeline.yml` (see the header of that file).

### Screenshots for the deliverable
- **Jobs -> credit-default -> the run**: pipeline graph with four green steps (also open *evaluate* -> *Metrics* and *Images*).
- **Models -> credit-default-model -> version 1**: artifacts (model, MODEL_CARD.md, evaluation/), tags (AUC, Gini, git SHA, data version).
- **Data -> credit-default-raw**: versioned data asset with its sha256 tag.
- **Compute -> cpu-cluster**: 0 nodes when idle.

## 4. Monitoring demo (batch)

```bash
make batches                                      # stable + downturn batches
make batch-monitor                                # downturn batch: score -> drift monitor -> insights
```
The *monitor drift* step ends with `[DRIFT ALERT]`; its outputs hold the PSI table, plots and `alert.json`. Schedules: `--schedule "0 7 * * 1"` here (weekly) and `uv run python pipeline/run_pipeline.py --schedule "0 6 1 * *"` (monthly retraining). They appear under *Jobs -> Schedules*.

## 5. Optional: real-time endpoint, model monitor, alerts

```bash
make deploy                                        # ~15 min; smoke test; prints scoring URI
az ml online-endpoint invoke -n <endpoint> -r deploy/sample_request.json -g rg-credit-risk -w mlw-credit-risk
uv run python monitoring/setup_model_monitor.py --email you@example.com
ALERT_EMAIL=you@example.com AZUREML_ENDPOINT=<endpoint> bash infra/alerts.sh
make delete-endpoint                               # stop billing after the demo
```
An endpoint instance needs its vCPUs plus a 20% reserve that Azure ML holds for upgrades, and it shares the family quota with the cluster, so let the cluster scale to zero first (it does so 3 minutes after the last job). A fresh subscription also needs the Microsoft.Compute, Network, PolicyInsights and Cdn providers, which `infra/setup_azure.sh` registers. Request logs and latency land in the workspace's Application Insights (*Logs* -> `traces`, `requests`).

## 6. The app

Local: `uv run python scripts/fetch_model.py --output app/model && make app`.
Cloud: `bash scripts/deploy_app.sh` builds the root `Dockerfile` and deploys it to Azure Container Apps (consumption plan, scales to zero; the monthly free grant covers a demo). Set `MIN_REPLICAS=1` on demo day to avoid a cold start. To enable the LLM agent, set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` before running the script (new subscriptions may need a quota request for Azure OpenAI; the agent falls back to offline SQL templates). To score through the endpoint, also set `SCORING_URI` and `SCORING_KEY`.

## 7. GitHub Actions

| Workflow | Trigger | What it does |
|---|---|---|
| `ci` | every push / PR | ruff format + lint, 53 tests (end to end on synthetic data), dry-run of every Azure spec |
| `train-and-deploy` | push to `main` touching model code, or manual | CI -> Azure ML pipeline (OIDC) -> model card in the job summary -> optional blue/green deploy behind the `production` environment |
| `batch-scoring-and-monitoring` | manual | scores a demo batch and runs the drift monitor |
| `deploy-app` | manual | deploys the Streamlit app to Container Apps |

Create the `production` environment in GitHub (*Settings -> Environments*) and add yourself as a required reviewer to get an approval gate before anything is deployed.

## 8. Costs and clean-up

| Resource | Billing | Typical cost here |
|---|---|---|
| Compute cluster | per node-minute, scales to 0 | cents per pipeline run |
| Storage, key vault | usage | negligible |
| Container registry (Basic) | daily | about $5 per month once created |
| Application Insights / Log Analytics | GB ingested; first 5 GB/month free | negligible |
| Online endpoint | every hour it exists | a few dollars per day - delete after use |
| Model monitor | serverless Spark per run | small per run |
| Container Apps | per second beyond the monthly free grant | usually $0 for a demo |

Set a budget alert in *Cost Management -> Budgets*. After the interview: `az group delete -n rg-credit-risk` removes everything.

## 9. Troubleshooting

- `MissingSubscriptionRegistration`: rerun the provider registration lines in `infra/setup_azure.sh`.
- Quota or capacity errors: different VM size (`AZUREML_VM_SIZE`) or region (`AZURE_LOCATION`).
- `az ml` errors about the old CLI: `az extension remove -n azure-cli-ml && az extension add -n ml`.
- Local auth errors: `az login` (the scripts use `DefaultAzureCredential`, which picks up the CLI login).
- Environment build failed: open the environment in studio -> *Build log*. Every package is pinned and the set was tested end to end, so the usual cause is a network/proxy issue; rerun the job.
