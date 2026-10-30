# One target per task; each calls the real command (see README "Commands").
ENDPOINT      = $(shell uv run python -c 'from deploy.deploy_endpoint import default_endpoint_name; print(default_endpoint_name())' 2>/dev/null)
RG           ?= rg-credit-risk
WS           ?= mlw-credit-risk

.DEFAULT_GOAL := help
.PHONY: help setup lint format test check local analytics app deck azure-setup train batches batch-monitor deploy invoke delete-endpoint lock

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Install the locked environment and download the UCI data
	uv sync && uv run python scripts/download_data.py

lint:  ## Format check and lint
	uv run ruff format --check . && uv run ruff check .

format:  ## Format and apply safe lint fixes
	uv run ruff format . && uv run ruff check . --fix

test:  ## Run the test suite
	uv run pytest -q

check: lint test  ## Lint and test (what CI runs)

local:  ## Full pipeline locally with a SQLite MLflow registry (~3 min)
	uv run python scripts/run_local.py

analytics:  ## Feature analysis + portfolio risk figures for the deck (docs/analytics/)
	uv run python scripts/risk_analytics.py

app:  ## Streamlit cockpit (registered model in app/model, else the local package)
	uv run streamlit run app/streamlit_app.py

deck:  ## Rebuild docs/presentation.html
	uv run python docs/presentation/build.py

azure-setup:  ## Resource group, workspace and GitHub OIDC (set GITHUB_REPO first)
	bash infra/setup_azure.sh

train:  ## Submit the training pipeline to Azure ML and wait
	uv run python pipeline/run_pipeline.py --wait

batches:  ## Build the stable and downturn demo batches
	uv run python scripts/make_batches.py

batch-monitor:  ## Batch scoring + drift monitoring on the downturn batch (Azure)
	uv run python pipeline/run_batch_pipeline.py --batch data/batches/portfolio_downturn.csv --wait

deploy:  ## Blue/green deployment of the latest registered model
	uv run python deploy/deploy_endpoint.py

invoke:  ## Score the sample request on the online endpoint
	az ml online-endpoint invoke -n $(ENDPOINT) -r deploy/sample_request.json -g $(RG) -w $(WS)

delete-endpoint:  ## Delete the online endpoint (stops hourly billing)
	uv run python deploy/deploy_endpoint.py --delete

lock:  ## Relock and regenerate the Azure and container environment files
	uv lock && uv run python scripts/lock_env.py && { echo "# Streamlit app container (Python 3.10): the app group and the creditrisk runtime from uv.lock. Regenerate with make lock."; uv export --no-default-groups --group app --no-hashes --no-emit-project --no-header --no-annotate --quiet; } > app/requirements.txt
