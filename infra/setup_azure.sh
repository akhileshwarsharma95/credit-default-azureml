#!/usr/bin/env bash
# One-time Azure setup for the credit default solution.
# Run in Azure Cloud Shell (bash) or anywhere with the Azure CLI, after `az login`.
#
#   export GITHUB_REPO=<github-user>/credit-default-azureml   # optional: enables GitHub Actions via OIDC
#   bash infra/setup_azure.sh
#
# Creates: resource providers, resource group, Azure ML workspace (with its storage,
# key vault, Application Insights), a .env file for the local scripts and, if
# GITHUB_REPO is set, a Microsoft Entra app with federated credentials so GitHub
# Actions can log in without any stored secret.
set -euo pipefail

LOCATION="${AZURE_LOCATION:-germanywestcentral}"
RG="${AZURE_RESOURCE_GROUP:-rg-credit-risk}"
WS="${AZUREML_WORKSPACE:-mlw-credit-risk}"
GITHUB_REPO="${GITHUB_REPO:-}"

SUB=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)
echo "Subscription $SUB | tenant $TENANT | region $LOCATION"

az extension add --name ml --upgrade --yes --only-show-errors
for ns in Microsoft.MachineLearningServices Microsoft.Storage Microsoft.KeyVault Microsoft.ContainerRegistry \
          Microsoft.Insights Microsoft.OperationalInsights Microsoft.App; do
  az provider register --namespace "$ns" --only-show-errors
done

az group create --name "$RG" --location "$LOCATION" --only-show-errors -o none
if ! az ml workspace show --name "$WS" --resource-group "$RG" -o none 2>/dev/null; then
  echo "Creating Azure ML workspace $WS (2-4 minutes) ..."
  az ml workspace create --name "$WS" --resource-group "$RG" --location "$LOCATION" --only-show-errors -o none
fi

cat > .env <<EOF
AZURE_SUBSCRIPTION_ID=$SUB
AZURE_TENANT_ID=$TENANT
AZURE_RESOURCE_GROUP=$RG
AZUREML_WORKSPACE=$WS
AZURE_LOCATION=$LOCATION
EOF
[[ -n "${AZUREML_VM_SIZE:-}" ]] && echo "AZUREML_VM_SIZE=$AZUREML_VM_SIZE" >> .env
echo "Wrote .env"

if [[ -n "$GITHUB_REPO" ]]; then
  echo "Configuring GitHub OIDC for $GITHUB_REPO ..."
  APP_ID=$(az ad app list --display-name "gh-$WS" --query "[0].appId" -o tsv)
  if [[ -z "$APP_ID" ]]; then
    APP_ID=$(az ad app create --display-name "gh-$WS" --query appId -o tsv)
  fi
  az ad sp show --id "$APP_ID" -o none 2>/dev/null || az ad sp create --id "$APP_ID" -o none
  SCOPE=$(az group show --name "$RG" --query id -o tsv)
  for i in 1 2 3 4 5; do  # the new service principal can take a moment to replicate
    az role assignment create --assignee "$APP_ID" --role Contributor --scope "$SCOPE" -o none && break || sleep 15
  done
  # Newer repos send immutable-ID subjects (repo:owner@<id>/repo@<id>:...); register both formats
  PREFIXES="repo:$GITHUB_REPO"
  if command -v gh >/dev/null 2>&1; then
    ID_PREFIX=$(gh api "repos/$GITHUB_REPO/actions/oidc/customization/sub" --jq '.sub_claim_prefix // empty' 2>/dev/null || true)
    [[ -n "$ID_PREFIX" && "$ID_PREFIX" != "repo:$GITHUB_REPO" ]] && PREFIXES="$PREFIXES $ID_PREFIX"
  fi
  for prefix in $PREFIXES; do
    suffix=""; [[ "$prefix" == "repo:$GITHUB_REPO" ]] || suffix="-id"
    for pair in "main:ref:refs/heads/main" "production:environment:production" "pr:pull_request"; do
      name="gh-${pair%%:*}$suffix"; subject="$prefix:${pair#*:}"
      az ad app federated-credential create --id "$APP_ID" --only-show-errors -o none --parameters \
        "{\"name\":\"$name\",\"issuer\":\"https://token.actions.githubusercontent.com\",\"subject\":\"$subject\",\"audiences\":[\"api://AzureADTokenExchange\"]}" \
        2>/dev/null || echo "  federated credential $name already exists"
    done
  done
  echo
  echo "Add these to GitHub -> Settings -> Secrets and variables -> Actions:"
  echo "  secrets:   AZURE_CLIENT_ID=$APP_ID  AZURE_TENANT_ID=$TENANT  AZURE_SUBSCRIPTION_ID=$SUB"
  echo "  variables: AZURE_RESOURCE_GROUP=$RG  AZUREML_WORKSPACE=$WS  AZURE_LOCATION=$LOCATION"
  if command -v gh >/dev/null 2>&1; then
    read -r -p "Set them now with the GitHub CLI? [y/N] " ok
    if [[ "$ok" =~ ^[Yy]$ ]]; then
      gh secret set AZURE_CLIENT_ID -R "$GITHUB_REPO" -b "$APP_ID"
      gh secret set AZURE_TENANT_ID -R "$GITHUB_REPO" -b "$TENANT"
      gh secret set AZURE_SUBSCRIPTION_ID -R "$GITHUB_REPO" -b "$SUB"
      gh variable set AZURE_RESOURCE_GROUP -R "$GITHUB_REPO" -b "$RG"
      gh variable set AZUREML_WORKSPACE -R "$GITHUB_REPO" -b "$WS"
      gh variable set AZURE_LOCATION -R "$GITHUB_REPO" -b "$LOCATION"
    fi
  fi
fi
echo "Done. Next: python pipeline/run_pipeline.py --wait"
