#!/usr/bin/env bash
# Deploy the Streamlit front end to Azure Container Apps (consumption plan, scales to zero).
#   bash scripts/deploy_app.sh
# Optional env vars passed to the app: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
# AZURE_OPENAI_DEPLOYMENT, APPLICATIONINSIGHTS_CONNECTION_STRING, SCORING_URI, SCORING_KEY.
set -euo pipefail
source .env 2>/dev/null || true
RG="${AZURE_RESOURCE_GROUP:?}"; LOCATION="${AZURE_LOCATION:-germanywestcentral}"
APP="${APP_NAME:-credit-risk-app}"; ENV_NAME="${ACA_ENVIRONMENT:-credit-risk-env}"

uv run python scripts/fetch_model.py --output app/model
az extension add --name containerapp --upgrade --yes --only-show-errors
az provider register --namespace Microsoft.App --only-show-errors
az provider register --namespace Microsoft.OperationalInsights --only-show-errors

az containerapp up --name "$APP" --resource-group "$RG" --location "$LOCATION" --environment "$ENV_NAME" \
  --source . --target-port 8501 --ingress external

SECRETS=(); ENVVARS=()
for v in AZURE_OPENAI_API_KEY APPLICATIONINSIGHTS_CONNECTION_STRING SCORING_KEY; do
  if [[ -n "${!v:-}" ]]; then s=$(echo "$v" | tr '[:upper:]_' '[:lower:]-'); SECRETS+=("$s=${!v}"); ENVVARS+=("$v=secretref:$s"); fi
done
for v in AZURE_OPENAI_ENDPOINT AZURE_OPENAI_DEPLOYMENT AZURE_OPENAI_API_VERSION SCORING_URI; do
  if [[ -n "${!v:-}" ]]; then ENVVARS+=("$v=${!v}"); fi
done
if (( ${#SECRETS[@]} )); then az containerapp secret set -n "$APP" -g "$RG" --secrets "${SECRETS[@]}" -o none; fi
if (( ${#ENVVARS[@]} )); then az containerapp update -n "$APP" -g "$RG" --set-env-vars "${ENVVARS[@]}" -o none; fi
az containerapp update -n "$APP" -g "$RG" --min-replicas "${MIN_REPLICAS:-0}" --max-replicas 1 -o none

echo "App URL: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo "Tip: before the demo run 'MIN_REPLICAS=1 bash scripts/deploy_app.sh' (or set --min-replicas 1) to avoid a cold start."
