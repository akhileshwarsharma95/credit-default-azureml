#!/usr/bin/env bash
# Azure Monitor alerts (e-mail): failed Azure ML jobs and slow/failing endpoint requests.
#   ALERT_EMAIL=you@example.com AZUREML_ENDPOINT=credit-risk-xxxxxx bash infra/alerts.sh
set -euo pipefail
source .env 2>/dev/null || true
RG="${AZURE_RESOURCE_GROUP:?}"; WS="${AZUREML_WORKSPACE:?}"; EMAIL="${ALERT_EMAIL:?set ALERT_EMAIL}"

AG_ID=$(az monitor action-group create -g "$RG" -n ag-credit-risk --short-name crisk \
  --action email risk-team "$EMAIL" --query id -o tsv)
WS_ID=$(az ml workspace show -n "$WS" -g "$RG" --query id -o tsv)

az monitor metrics alert create -g "$RG" -n aml-failed-runs --scopes "$WS_ID" \
  --condition "total Failed Runs > 0" --window-size 1h --evaluation-frequency 15m \
  --action "$AG_ID" --severity 2 --description "An Azure ML job failed (training, retraining, scoring or monitoring) - see RUNBOOK" -o none
echo "Alert: failed Azure ML runs"

if [[ -n "${AZUREML_ENDPOINT:-}" ]]; then
  EP_ID=$(az ml online-endpoint show -n "$AZUREML_ENDPOINT" -g "$RG" -w "$WS" --query id -o tsv)
  az monitor metrics alert create -g "$RG" -n endpoint-latency-p95 --scopes "$EP_ID" \
    --condition "avg RequestLatency_P95 > 1000" --window-size 15m --evaluation-frequency 5m \
    --action "$AG_ID" --severity 3 --description "Scoring latency p95 above 1s" -o none
  echo "Alert: endpoint latency"
fi
echo "Drift alerts come from the Azure ML model monitor (monitoring/setup_model_monitor.py --email ...)."
