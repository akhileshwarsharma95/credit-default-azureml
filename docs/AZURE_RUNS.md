# Azure run record

What ran in the Azure ML workspace `mlw-credit-risk` (Germany West Central), exported from the workspace with `az ml` and MLflow on 2 November 2026. Times are UTC; step durations are wall time from submission to the last status update. Screenshots of the same views are in `docs/screenshots/`.

## Pipeline runs

| Run | Pipeline | Submitted | Submitted by | Total | Steps |
|---|---|---|---|---|---|
| `cool_rhythm_hx51hfjj1r` | training | 2026-10-29 20:06 | `make train` | 21 min | preprocess 15.7 min (includes the one-time environment image build) · train 2.4 · evaluate 0.6 · register 0.5 → **v1** |
| `ivory_rabbit_9x86flykq4` | batch scoring + monitoring | 2026-10-29 20:30 | `make batch-monitor` | 2.4 min | score 0.6 · insights 0.5 and monitor 1.2 in parallel → **drift alert** |
| `brave_ocean_w3b8k8s0dq` | training | 2026-10-29 21:08 | GitHub Actions (OIDC) | 7.4 min | preprocess and train reused from the previous run (unchanged inputs) · evaluate 5.7 · register 0.6 → **v2** |
| `eager_cord_w2yj7rnz4n` | training | 2026-10-30 00:47 | GitHub Actions (OIDC) | 11.4 min | preprocess 5.7 · train 2.2 · evaluate 0.5 · register 0.6 → **v3**, commit `600cc5b` |

All runs and steps completed. Compute: `cpu-cluster`, Standard_D2ds_v5, 0–2 nodes, scale-down after 180 s idle.

## Registered models

| Version | Registered | Test AUC | Gini | ECE | Threshold | Champion comparison |
|---|---|---|---|---|---|---|
| credit-default-model:1 | 2026-10-29 20:27 | 0.7785 | 0.5570 | 0.0105 | 0.1667 | no champion yet |
| credit-default-model:2 | 2026-10-29 21:16 | 0.7785 | 0.5570 | 0.0105 | 0.1667 | compared with v1, passed |
| credit-default-model:3 | 2026-10-30 00:58 | 0.7785 | 0.5570 | 0.0105 | 0.1667 | compared with v2, passed |

Every version is tagged with `model_family = lightgbm_monotone`, `data_version = credit-default-raw:1` and `exclude_sensitive = True`, plus the git commit it was trained from. The three versions are identical in performance because the data, seed and modelling code did not change between them; v2 and v3 came from engineering changes (CI, uv-locked environment).

## Metrics (evaluate step of `eager_cord_w2yj7rnz4n`, held-out test set of 5,993 clients)

| Group | Metric | Value |
|---|---|---|
| Ranking | AUC (95% CI) | 0.7785 (0.7647–0.7917) |
| | Gini (95% CI) / KS | 0.557 (0.529–0.583) / 0.423 |
| | PR AUC | 0.549 |
| Calibration | ECE / Brier / log loss | 0.0105 / 0.1351 / 0.430 |
| | Calibration slope / intercept | 0.944 / −0.053 |
| | Hosmer-Lemeshow p | 0.359 |
| | Mean PD vs observed default rate | 22.03% vs 22.13% |
| Backtest | Rating grades red / amber | 0 / 1 |
| Decision at PD ≥ 0.1667 | Recall / precision / flagged | 71.0% / 38.9% / 40.4% |
| | Confusion matrix (TP / FP / FN / TN) | 942 / 1,478 / 384 / 3,189 |
| | Expected cost per client vs no model | 0.567 vs 1.106 (−49%) |
| Fairness (not model inputs) | Disparate impact ratio: sex / marital status / age | 0.946 / 0.993 / 0.893 |
| Gate | Champion AUC / register decision | 0.7785 / registered |

Train step (5-fold out-of-fold): monotonic LightGBM AUC 0.7893, unconstrained LightGBM 0.7889, logistic scorecard 0.7713; the monotonicity constraints cost −0.0005 AUC, which is no cost.

## Drift monitoring (monitor step of `ivory_rabbit_9x86flykq4`)

The downturn batch (`credit-default-batch-portfolio-downturn:1`) scored against model v1's reference profile:

| Metric | Value |
|---|---|
| Score PSI | 0.344 (alert ≥ 0.25) |
| Rating-grade PSI | 0.331 |
| Features in alert / warning | 13 / 9 (largest feature PSI 1.70) |
| Mean PD, reference → downturn | 22.1% → 29.7% |
| Drift alert | raised |

## Data assets

| Asset | Version | sha256 |
|---|---|---|
| `credit-default-raw` (UCI, `default_of_credit_card_clients.xls`) | 1 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| `credit-default-batch-portfolio-downturn` (`portfolio_downturn.csv`) | 1 | `3da7e629d3ee471e5a2a257266320a9b37ad8c261fa30961c9dd49ccea8c91ea` |

## Online endpoint

`credit-risk-fb1504`, deployment `blue` serving credit-default-model:2 on Standard_D2as_v4 × 1 with the data collector on inputs and outputs, from 29 October to 2 November 2026, when I deleted it to stop hourly billing. Its response to `deploy/sample_request.json` is saved in `deploy/sample_response.json` (79 ms). Recreate it with `make deploy`.
