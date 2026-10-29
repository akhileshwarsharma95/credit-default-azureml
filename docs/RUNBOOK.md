# Runbook - credit default PD engine

Operating model for the pipeline, the model and the endpoint: what is monitored, what triggers an alert, and what to do.

## Cadence

| Process | Schedule | Mechanism | Output |
|---|---|---|---|
| Retraining (challenger) | monthly, 1st, 06:00 | Azure ML schedule `credit-default-retraining` | new version registered only if the gate passes |
| Batch scoring + drift monitor | weekly, Monday 07:00 | Azure ML schedule `credit-default-weekly-monitoring` | scored portfolio, PSI report, `alert.json`, Power BI tables |
| Endpoint monitor | daily, 06:00 UTC | Azure ML model monitor | drift / data-quality signals, e-mail alert |
| Backtest with outcomes | monthly, once labels mature | batch pipeline on a labelled batch | realised AUC/Gini, Jeffreys test per grade |
| Model review | quarterly | model card + monitoring history | sign-off by credit risk and model owner |

## Signals and thresholds

| Signal | Warn | Alert | Where |
|---|---|---|---|
| Data-quality hard check (types, ranges, nulls) | - | any violation: step fails | preprocess / score step logs |
| Score PSI vs training reference | >= 0.10 | >= 0.25 | monitor step, `drift_summary.json` |
| Feature PSI (any feature) | >= 0.10 | >= 0.25 | `drift_table.csv`, PSI chart |
| Rating-grade PSI | >= 0.10 | >= 0.25 | `drift_summary.json` |
| Backtest per grade (Jeffreys) | amber (p < 5%) | red (p < 0.1%) | `backtest.csv` |
| Realised AUC vs validation AUC | -0.02 | -0.05 | monitor step (labelled batches) |
| Calibration error (ECE) | > 0.03 | > 0.05 | evaluate / monitor |
| Endpoint p95 latency | > 500 ms | > 1 s | Azure Monitor alert `endpoint-latency-p95` |
| Endpoint 4xx/5xx rate | > 1% | > 5% | Application Insights `requests` |
| Failed Azure ML job | - | any | Azure Monitor alert `aml-failed-runs` |

## Playbooks

### 1. Pipeline or job failed
1. Studio -> *Jobs* -> failed run -> failed step -> *Outputs + logs* (`user_logs/std_log.txt`).
2. Environment build error: check *Environments* -> build log; usually transient - rerun.
3. Code error: reproduce locally with `python scripts/run_local.py --n_trials 2`, fix, add a test, merge; CI reruns the pipeline.
4. No model is promoted by a failed run, so production is unaffected; nothing to roll back.

### 2. Data-quality gate failed
1. Read `quality_report.json` / `.md` in the preprocess (or score) outputs: column, check, failure examples.
2. Upstream schema change or extraction bug -> contact the data owner; do **not** relax the hard schema to make the run pass.
3. New legitimate category -> extend the documented domain in `creditrisk/quality.py`, review whether features or the model need retraining, ship via PR.

### 3. Drift alert
1. Open the monitor output: which features drive the PSI, how the PD distribution and grade mix moved, mean PD before/after.
2. Real portfolio shift (economy, new product, new segment) vs. data issue (pipeline bug, unit change)? Compare raw distributions; check DQ soft violations.
3. Data issue -> treat as playbook 2. Real shift -> inform credit management (more clients may move to Watch/Intervene), check the latest labelled backtest, trigger retraining (`train-and-deploy` workflow) and let the champion/challenger gate decide.
4. Until a new model is approved, consider tightening the cut-off in the policy (it is a parameter, not a retrain).

### 4. Calibration or backtest breach (amber/red grade)
1. Amber on one grade: document, watch the next cycle.
2. Red, or amber on two consecutive cycles: recalibrate (rerun with fresh data; calibration is a separate layer) and review the master scale.
3. Systematic underestimation across grades: escalate to model owner; apply a conservative margin of conservatism to PDs until a recalibrated model passes validation.

### 5. Endpoint incident (latency, errors)
1. Application Insights -> *Failures* / *Performance*; endpoint *Logs* in studio (`az ml online-deployment get-logs -e <ep> -n <deployment>`).
2. Capacity: scale out `az ml online-deployment update --name blue --endpoint-name <ep> --set instance_count=2`.
3. Bad release: `python deploy/deploy_endpoint.py --rollback` sends 100% traffic to the previous deployment (keep it with `--keep-old` during canary releases).
4. Batch scoring keeps working independently of the endpoint.

### 6. A bad model reached production
1. Roll back the endpoint (`--rollback`) or redeploy a known-good version: `python deploy/deploy_endpoint.py --model-version <n>`.
2. Batch scoring: pass `--model-version <n>` to `run_batch_pipeline.py`.
3. Root-cause through lineage: the model version tags hold the git SHA and data asset version; the model card holds the evidence the gate saw.

## Promotion policy
- A challenger is registered only if it beats the minimum AUC, the calibration limit, and is not worse than the champion by more than 0.005 AUC on the same test data.
- Registration is not deployment: deployment runs behind the GitHub `production` environment (human approval) and a smoke test, with blue/green or canary traffic.
- Every registered version carries its model card, evaluation evidence, drift reference profile, git SHA and data version.

## Post-incident
Blameless review within a week: timeline, root cause, why monitoring did or did not catch it, and a regression test or new monitor.
