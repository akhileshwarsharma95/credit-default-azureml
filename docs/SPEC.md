# Specification: credit default PD engine on Azure ML

| | |
|---|---|
| Author | Akhileshwar Sharma |
| Context | Senior Data Scientist case study, DKV Mobility |
| Status | Implemented; every requirement below is verified by the named test or command |
| Source of parameters | `src/creditrisk/config.yaml` (values quoted here are the defaults) |

I work spec-first: this document states what the system must do and how each requirement is proven, before and independently of the code. Requirement IDs are stable; the traceability table (section 15) maps each one to code and evidence.

**Requirement keywords:** *must* = mandatory and verified; *should* = default behaviour, deviation needs a reason; *may* = optional.

---

## 1. Purpose and scope

**1.1 Problem.** Predict the probability that a credit-card client misses the next monthly payment, and turn that probability into credit decisions: who to act on, how much limit to grant, what to provision and price.

**1.2 Brief (case study).** Load, clean and preprocess the data; train and log metrics; load the registered model and test data, evaluate and log metrics and plots; build an Azure ML SDK v2 pipeline (or YAML + CLI) on a compute cluster; register the model. Bonus: YAML components and an endpoint.

**1.3 Scope of this build.** The brief plus the operating model a credit-risk team needs: validation as for a credit model, batch scoring with drift monitoring, a real-time endpoint, portfolio-level risk, explainability, a cockpit for credit officers, CI/CD and governance.

**1.4 Users.**

| User | Needs |
|---|---|
| Credit management | Calibrated PDs, grades, expected loss, limit and pricing actions, escalation |
| Collections / account managers | Early-warning watchlist with reasons, client profile |
| Finance | Expected credit loss inputs (stages, lifetime PD vectors), portfolio loss distribution |
| Model owner / validation | Evidence per version: metrics, backtest, fairness, model card, lineage |
| ML / platform engineering | Reproducible pipelines, one command per task, monitoring, rollback |

**1.5 Decisions supported (BR-1).** The engine *must* support four decisions: (a) intervention (act / watch / standard), (b) limit steering, (c) provisioning inputs, (d) risk-based pricing. It is *not* an automated decline engine; decisions stay with credit management.

---

## 2. Data requirements

**DR-1 Source.** The official UCI *Default of Credit Card Clients* file (id 350; 30,000 clients, Taiwan, Apr–Sep 2005) *must* be downloaded from UCI, with a mirror fallback on network or archive errors. *Verified:* `make setup`.

**DR-2 Lineage.** Every training run *must* register the raw file as a versioned Azure ML data asset (`credit-default-raw:v`) tagged with its sha256; re-registering an identical file *must* reuse the existing version. Each model version *must* carry the data version and git SHA as tags.

**DR-3 Schema (hard checks, fail fast).** Types coerced; no nulls; `LIMIT_BAL` 1–10,000,000; `SEX` ∈ {1,2}; `EDUCATION` 0–6; `MARRIAGE` 0–3; `AGE` 18–100; `PAY_1..6` −2–9; bill amounts ±10,000,000; payments 0–10,000,000; target ∈ {0,1}. Any violation *must* stop the step with a report. *Verified:* `test_quality_gate_fails_on_impossible_values`.

**DR-4 Documented domain (soft checks, clean and report).** Undocumented `EDUCATION` codes (0, 5, 6) *must* be remapped to 4 "other" and undocumented `MARRIAGE` 0 to 3 "other", counted in the quality report. `PAY_x` = −2 / −1 / 0 are undocumented but meaningful (no consumption / paid in full / revolving) and *must* be kept. Exact duplicates on all feature columns *must* be dropped. *Verified:* `test_quality_gate_passes_and_cleans`.

**DR-5 Naming.** `PAY_0` *must* be renamed `PAY_1` so status, bill and payment indices align (1 = September, the latest month).

**DR-6 Split before fitting.** A stratified 80/20 train/test split (`random_state` 42) *must* happen before anything is fitted. Only row-wise, stateless cleaning may precede it.

**DR-7 Sensitive attributes.** `SEX`, `MARRIAGE` and `AGE` *must not* be model inputs by default (`exclude_sensitive: true`); they *must* be kept for the fairness check.

**DR-8 Known data characteristics** (documented, not "fixed"):
- the target is next-month payment default, not the regulatory 90-days-past-due definition;
- one cross-sectional snapshot, so there is no out-of-time sample;
- a 1-month delay (`PAY = 1`) is almost never recorded before the latest month (7 observed transitions out of the 30-dpd bucket in the 5,993 test histories);
- amounts are in NT$.

---

## 3. Feature requirements

**FR-1 Features live inside the model pipeline.** Engineered features *must* be computed by a stateless transformer inside the saved model, so batch, endpoint and app score identically (no training/serving skew).

**FR-2 Engineered features.**

| Feature | Definition |
|---|---|
| `util_1`, `util_avg`, `util_max` | bill / limit (latest, mean, max over 6 months), clipped to [−1, 5] |
| `bill_trend` | (bill Sep − bill Apr) / limit, clipped to [−5, 5] |
| `pay_ratio_1` | payment in Sep / bill in Aug (the statement it settles), clipped to [0, 5] |
| `pay_ratio_avg` | 5-month payments / 5-month bills (positive), clipped to [0, 5] |
| `n_delays`, `n_delays_recent3` | months with status ≥ 1 (last 6, last 3) |
| `max_delay` | longest delay in months |
| `delay_trend` | delay in Sep − delay in Jun |
| `n_no_consumption`, `n_paid_full` | months with status −2, −1 |
| `months_since_delay` | months since the latest late payment (7 = never) |

**FR-3 Monotonic risk directions.** LightGBM *must* use monotonic constraints: risk rises with `PAY_1..6`, `n_delays`, `n_delays_recent3`, `max_delay`; falls with `LIMIT_BAL`, `pay_ratio_avg`, `months_since_delay`. Paying later can never lower a PD.

**FR-4 Feature analysis.** For every raw and engineered feature, the default rate per decile (per value for ≤ 10 distinct values), the information value (IV bands 0.02 / 0.1 / 0.3 / 0.5) and the risk direction (Spearman of bin default rates, ±0.8) *must* be reported. *Verified:* `test_feature_report_ranks_the_informative_feature_first`; output `docs/analytics/feature_report.md` via `make analytics`.

**FR-5 Findings that shaped the design** (from FR-4 on the full data): delinquency patterns are very strong (IV 0.78–0.86), bill amounts are not predictive (IV ≤ 0.02), the engineered `n_delays` outranks raw `PAY_1`, raw PAY codes are non-monotonic because of the undocumented codes, a higher limit means lower risk, and sex and marital status carry almost no signal (IV 0.009 / 0.005).

---

## 4. Model requirements

**MR-1 Candidates.** Monotonic LightGBM (selectable), logistic scorecard (selectable) and unconstrained LightGBM (sensitivity check only, never selected).

**MR-2 Tuning.** Optuna, 25 trials (`n_trials`), 3 tuning folds, maximising AUC; 0 trials = default hyperparameters.

**MR-3 Selection.** Out-of-fold metrics on 5 folds; the selectable candidate with the highest OOF AUC wins. The AUC the monotonic constraints cost *must* be reported (`monotonicity_auc_cost`).

**MR-4 Calibration.** Isotonic calibration, fitted out-of-fold on the training set. No resampling of the classes: it would distort the probabilities that pricing and provisioning rely on.

**MR-5 PD floor.** 0.1% (`pd_floor`, the CRR3 input floor for qualifying revolving retail exposures), applied by a wrapper around the calibrated model.

**MR-6 Rating master scale.** Seven grades by PD: A [0, 5%), B [5, 8%), C [8, 12%), D [12, 20%), E [20, 35%), F [35, 55%), G [55%, 100%], chosen so every grade holds a testable population.

**MR-7 Cut-off from economics.** With calibrated PDs the cost-optimal cut-off is cost_fp / (cost_fp + cost_fn) = 1 / (1 + 5) = 0.167 (`costs` in config). *Verified:* `test_cost_optimal_threshold_and_grades`.

**MR-8 Through-the-cycle (TTC) level.** The model provides the ranking; the level is a separate calibration: all PDs are shifted by one log-odds constant so the portfolio mean equals the long-run default rate (`ttc.target_pd`, illustrative 20%); the slope is unchanged. Segment-level TTC (each borrower type to its own average) uses the same function per segment. *Verified:* `test_ttc_shift_hits_the_target_and_keeps_the_slope`.

**MR-9 One model, segment calibration.** Separate models per risk grade *must not* be built (grades derive from the PD, so it is circular); per-segment models on this data volume overfit. Segment differences are handled by segment calibration (MR-8).

**MR-10 Reproducibility.** Fixed seeds (`random_state` 42, `PYTHONHASHSEED` 0), pinned library versions in the model package, identical outputs for identical inputs on the same platform.

---

## 5. Validation requirements

**VR-1 Discrimination.** ROC AUC, Gini and KS on the held-out test set with 500-sample bootstrap confidence intervals.

**VR-2 Calibration.** ECE, Brier score, calibration slope, Hosmer-Lemeshow p-value; calibrated vs uncalibrated comparison.

**VR-3 PD backtest per grade.** One-sided Jeffreys test of observed vs predicted default rate per grade; traffic lights: amber p < 5%, red p < 0.1%. *Verified:* `test_jeffreys_flags_underestimation`.

**VR-4 Fairness.** Per sensitive attribute: disparate impact ratio (flag below 0.8), equal-opportunity gap, calibration gap.

**VR-5 Explainability.** Exact TreeSHAP contributions per client (log-odds); the top 4 risk-raising contributions become reason codes with stable codes (e.g. `RC_PAY_1`) and plain sentences using the client's own values. Reason codes *must* appear in batch output, the endpoint response, the watchlist and the cockpit.

**VR-6 Registration gate (champion/challenger).** A candidate is registered only if test AUC ≥ 0.70, ECE ≤ 0.05, and AUC ≥ champion AUC − 0.005 on the same test set (`registration_gate`). An unreachable registry or an unloadable champion *must not* stop evaluation; the comparison is then recorded as skipped. *Verified:* `test_champion_challenger_second_registration`, `test_quality_gate_blocks_registration`, `test_component_defaults_match_config`.

**VR-7 Evidence per version.** Each registered version *must* carry the model, `MODEL_CARD.md`, the evaluation outputs (metrics, 12 charts, tables), the drift reference profile and tags (test AUC, Gini, ECE, threshold, git SHA, data version). *Verified:* `test_model_package_contents`.

---

## 6. Business layer requirements

**BL-1 Exposure.** EAD = drawn balance + CCF × undrawn limit (CCF 0.50).

**BL-2 Expected loss.** EL = PD × LGD × EAD (LGD 0.60, unsecured revolving; an explicit assumption until internal loss data exists).

**BL-3 Scenario LGD (target model).** LGD = p_cure × LGD_cure + p_recovery × LGD_recovery + p_loss × LGD_loss, where recovery is collateral (deposits, guarantees; vehicle remarketing for leasing) and LGD_recovery = (EAD − net proceeds) / EAD × (1 − R). Status: roadmap; today's flat LGD is documented in the model card.

**BL-4 IFRS 9 staging (illustrative).** Stage 3: 90+ dpd. Stage 2: 30+ dpd backstop or a SICR grade (F, G). Otherwise stage 1. ECL: 12-month horizon in stage 1, lifetime (3 years behavioural life) in stage 2, PD = 1 in stage 3.

**BL-5 Lifetime PD from migration (cohort method).** The mean monthly migration matrix between delinquency buckets Current / 30 / 60 / 90+ dpd *must* be estimated from the 6-month panel, with 90+ dpd absorbing. Cumulative PD(t) = (Mᵗ)[:, default], marginal PD(t) = Mᵗ − Mᵗ⁻¹, delivered as a vector per bucket (the format an impairment engine consumes). Buckets with fewer than 100 observed transitions *must* be reported as not estimable, never as PD 0. Preconditions stated with the output: homogeneous Markov chain (stable migration rates), stable portfolio size and mix. *Verified:* `test_lifetime_pd_is_the_default_column_of_matrix_powers`, `test_lifetime_pd_skips_buckets_without_enough_observed_migrations`.

**BL-6 Monthly default forecast.** Expected new 90+ dpd defaults and loss per month for 12 months = Σ bucket share × marginal PD × clients, with the share of the portfolio covered reported (`markov_coverage`).

**BL-7 Portfolio loss distribution (total risk).** Single-factor (Vasicek / Basel ASRF) model: loss at confidence q = Σ EAD × LGD × Φ((Φ⁻¹(PD) + √ρ Φ⁻¹(q)) / √(1 − ρ)); report expected loss, 50 / 90 / 99 / 99.9% losses and unexpected loss at 99.9%. ρ = 0.04 (Basel QRRE) for revolving retail; 0.12–0.24 for SME and corporate fleet clients. *Verified:* `test_credit_var_exceeds_expected_loss_and_rises_with_correlation`.

**BL-8 Concentration.** Herfindahl index of single-name EAD, top-10 and top-1% EAD share, largest segment share.

**BL-9 Early-warning triggers.** New delinquency this month; delinquency rolling to a worse bucket; utilisation above 80% with a jump of more than 20 points; payment below 10% of the previous bill; PD in grade F/G.

**BL-10 Action levels and escalation** (`escalation` in config):

| Level | Trigger | Action | Owner | Response |
|---|---|---|---|---|
| Standard | PD below half the cut-off and fewer than 2 triggers | Normal limits and pricing; increase if A/B and utilisation above 80% | Automated | Monthly batch |
| Watch | PD ≥ half the cut-off, or ≥ 2 triggers | No limit increase; client contact; payment terms reviewed | Account manager | 10 days |
| Intervene | PD ≥ cut-off | Undrawn limit reduced; deposit or guarantee requested; collections plan | Credit risk | 48 hours |
| Default | 90+ dpd | Card blocked; recovery and legal process; full lifetime provision | Collections / credit committee | Immediately |

**BL-11 Limit steering.** Grades F/G with utilisation below 80%: reduce the limit to 110% of the drawn balance (rounded up to 1,000, minimum 10,000). Grades A/B with utilisation above 80%: increase by 20%. The EAD and EL impact *must* be reported.

**BL-12 Risk-based pricing.** Price per unit of exposure = funding (1.0%) + expected loss rate + capital charge (unexpected loss at 99.9% × 12% cost of capital) + opex (0.5%) + margin (0.5%), per grade. For DKV this maps to card fees, payment terms and deposit requirements. *Verified:* `test_risk_based_price_rises_with_pd`.

**BL-13 Segments.** k-means behavioural segments (5) named by their profile (e.g. "Delinquent revolvers", "Transactors (pay in full)").

**BL-14 Portfolio insights in SQL.** DuckDB SQL files for KPIs, grade, segment and stage summaries, watchlist (top 50 by EL), limit actions, roll rates and delinquency trend; outputs are flat tables ready for Power BI.

---

## 7. Serving requirements

**SR-1 Batch pipeline.** score → monitor drift → portfolio insights, on a versioned batch data asset, pinned or latest model version, weekly schedule option. *Verified:* `test_batch_scoring_monitoring_insights`.

**SR-2 Online endpoint contract.** Request: `{"clients": [{feature: value, ...}]}` or `{"input_data": {"columns": [...], "data": [[...]]}}`. Response: `model_version`, `threshold`, `latency_ms` and per client `pd, grade, action, flag, ead, el, ifrs9_stage, ecl, ews_triggers, limit_action, recommended_limit, ID, reason_codes[{code, text}]`. A non-object body, missing fields or non-numeric values *must* return HTTP 400 with the required fields, never 500. *Verified:* `test_endpoint_scoring_script`.

**SR-3 Release.** Blue/green: deploy to the idle colour, smoke test (3 sample clients, 0 < PD < 1), then switch traffic or split for a canary; previous deployment deleted unless kept; one-command rollback.

**SR-4 Telemetry.** Application Insights enabled; data collector on inputs and outputs (feeds the model monitor); OpenTelemetry span per request.

**SR-5 Instance.** `Standard_D2as_v4` by default (`endpoint_sku`), overridable by `AZUREML_ENDPOINT_SKU`.

---

## 8. Monitoring and observability requirements

**OR-1 Logging.** One logger (`creditrisk.observability.get_logger`), format `time level module message`, INFO by default (`LOG_LEVEL`), on stderr, so stdout carries only machine-readable output (dry-run specs, tables, the `[DRIFT ALERT]` line). Azure ML captures both streams. Azure SDK loggers are held at WARNING (they log every HTTP request at INFO). Telemetry export to Application Insights when `APPLICATIONINSIGHTS_CONNECTION_STRING` is set; a telemetry failure *must never* break scoring.

**OR-2 Drift.** PSI per feature, score and rating grade against the reference profile stored with each model (10 quantile bins, categorical for ≤ 12 values); warn ≥ 0.10, alert ≥ 0.25. *Verified:* `test_psi_identical_is_zero_and_shift_alerts`, `test_psi_is_robust_to_csv_float_roundtrip`.

**OR-3 Performance on labelled batches.** Realised AUC vs validation AUC: warn at a drop of 0.02, alert at 0.05; Jeffreys backtest per grade.

**OR-4 Alerts.** Failed Azure ML job (any); endpoint p95 latency above 500 ms warn, above 1 s alert; 4xx/5xx above 1% warn, above 5% alert.

**OR-5 Runbook.** Six playbooks (job failed, data-quality gate, drift alert, backtest breach, endpoint incident, bad model in production), each with owner actions (`docs/RUNBOOK.md`).

---

## 9. Data retention, compliance and regulation

**CR-1 Lineage (in place).** Data assets with sha256, model versions with git SHA and data version, model card and evaluation evidence per version, MLflow runs, data collector output in Blob storage.

**CR-2 Long-term storage (target).** Immutable (WORM) Blob storage for scored decisions and model evidence; lifecycle tiers (hot → cool → archive); 10-year retention for business records (German commercial and tax law); GDPR data minimisation and purpose limitation for personal data; deletion policy for raw personal data after the retention period.

**CR-3 Retraining on production data (target).** The data collector stores endpoint inputs and outputs; once the outcome window matures (one month for this target, 12 months for a 90-dpd target) outcomes are joined by client and date, backtested as a labelled batch, and used by the monthly retraining; validation out-of-time by vintage; the registration gate decides.

**CR-4 EU AI Act.** Regulation (EU) 2024/1689 lists AI systems that evaluate the creditworthiness or establish the credit score of **natural persons** as high-risk (Annex III, point 5(b); fraud detection is excluded).
- *Timeline:* in force 1 August 2024; prohibitions and AI-literacy duties since 2 February 2025; general-purpose AI obligations since 2 August 2025; transparency obligations (Article 50) since 2 August 2026. The Annex III high-risk obligations were due on 2 August 2026, but the Digital Omnibus on AI, adopted in 2026, moved them to **2 December 2027** (and Annex I product-embedded systems to 2 August 2028). Dates to be confirmed against the Official Journal text.
- *Scope for DKV:* fleet customers are mostly legal persons, so most of the B2B portfolio is outside Annex III 5(b); sole traders and private customers are natural persons and would be in scope. I apply the high-risk requirements to the whole portfolio as good practice.
- *How this build maps to the high-risk requirements:* risk management (runbook, gate, monitoring), data governance (DR-1 to DR-8), technical documentation and record-keeping (model card, lineage, logs), transparency to deployers (reason codes, model card), human oversight (not an automated decline engine; escalation owners), accuracy and robustness (VR-1 to VR-6, drift monitoring), post-market monitoring (OR-2 to OR-4).
- *Article 50:* the cockpit's portfolio agent shows that answers come from an AI model and which mode is active.

**CR-5 Other frameworks referenced.** EBA guidelines on loan origination and monitoring (explainable, monitored automated models); IFRS 9 (staging, lifetime ECL); CRR3 (PD input floor).

---

## 10. Engineering standards

**ES-1 Repository layout.** `components/` (7 YAML components, single source of truth for SDK and CLI), `pipeline/` (submitters, CLI pipelines), `src/` (step scripts), `src/creditrisk/` (library + `config.yaml` + `sql/`), `src/serving/` (endpoint script), `deploy/`, `monitoring/`, `infra/`, `app/`, `scripts/`, `tests/`, `docs/`.

**ES-2 Python and dependencies.** Python 3.10–3.13 (3.10 in CI and Azure); uv-managed: direct dependencies pinned in `pyproject.toml`, groups `azure`, `app`, `dev`; `uv.lock` the single lock for Mac and Linux; Azure conda environments and the container requirements generated from the same versions (`make lock`); no upgrades without a relock and a full verification.

**ES-3 Packages.** `creditrisk`, `pipeline` and `deploy` are installed in editable mode; no `sys.path` edits except in the endpoint script (it runs from the uploaded `src/`, where nothing is installed) and in tests that import step scripts the way Azure runs them. Step scripts *must* import only `creditrisk` and third-party packages. *Verified:* `test_step_scripts_import_only_creditrisk_and_third_party`.

**ES-4 Style.** `ruff format` (line length 120); `ruff check` rules E, F, W, I, B, BLE, UP, SIM, C4, RUF, PTH; E501 ignored with reason (long user-facing strings); every `noqa` names a reason.

**ES-5 Simplicity.** No class, framework or abstraction layer without a second use; functions named after what they do; long entry points split into a few named steps; no dead code.

**ES-6 Comments.** Module docstrings everywhere; comments only for the non-obvious "why" (a regulatory floor, a data quirk, a deliberate exception); no comments that restate the code.

**ES-7 Error handling.** Catch only the exceptions expected at that point. A broad catch is allowed only where a failure must not stop the main path (telemetry, endpoint data collection, an optional plot, the advisory champion comparison, model-version metadata, the agent passing tool errors back to the model) and *must* carry a reason and a log line. At the boundaries (missing data file, missing setting, failed Azure call, missing step output) fail fast with a message that says what to do.

**ES-8 Configuration.** Business and technical parameters in `src/creditrisk/config.yaml`, inside the package so it ships with every Azure job, the endpoint and each registered model. Precedence: command-line flag, then environment / `.env` (Azure settings only), then `config.yaml`. Values repeated in Azure component YAMLs *must* equal the config. *Verified:* `test_config_drives_policy_and_monitoring_constants`, `test_component_defaults_match_config`, `test_cli_pipeline_defaults_match_config`.

**ES-9 Testing.** Unit tests for logic (quality, features, metrics, Jeffreys, PSI, policy, SQL guard, migration, portfolio risk, pricing, config consistency, import boundaries); one end-to-end test that runs every pipeline step on synthetic data in the official UCI layout with a local MLflow registry; dry-runs of every Azure spec. Every test protects a behaviour; no duplicate tests. 53 tests, about 15 seconds. *Verified:* `make check`.

**ES-10 Commands.** One `make` target per task, each a single line calling the real command, mirrored in the README "Commands" table so everything also runs without make (`uv run …`).

**ES-11 CI/CD.** GitHub Actions: `ci` on every push/PR (format check, lint, tests, spec dry-runs, `uv sync --locked`); `train-and-deploy` on model-code pushes or manual dispatch (Azure ML pipeline, model card in the job summary, deployment only by manual dispatch behind the `production` environment with a required reviewer); `batch-scoring-and-monitoring` and `deploy-app` on manual dispatch.

**ES-12 Security.** No secrets in code or GitHub: GitHub logs in through OIDC federated credentials (both subject-claim formats), Contributor scoped to one resource group; workspace secrets in Key Vault; endpoint key auth; the agent's SQL tool is SELECT-only, single statement, keyword deny-list, no file or network functions, 200-row cap, on a locked-down DuckDB connection. *Verified:* `test_sql_guard_blocks_unsafe`, `test_sql_guard_allows_select_and_caps_rows`, `test_duckdb_external_access_locked`.

**ES-13 Infrastructure.** `infra/setup_azure.sh` is idempotent: resource providers (including Compute, Network, PolicyInsights and Cdn for managed endpoints), resource group, workspace, `.env`, Entra app with federated credentials, GitHub secrets and variables. The compute cluster (`Standard_D2ds_v5`, DDSv5) scales 0–2 nodes and to zero after 3 minutes idle; the endpoint (`Standard_D2as_v4`, DASv4) uses a different VM family so it can never take the cluster's quota.

---

## 11. Front-end requirements

**FE-1 Cockpit (Streamlit).** Tabs: Portfolio, Client scoring, Cut-off & limits, Model & monitoring, Ask the portfolio, Model card. Sidebar: cost assumptions, cut-off, LGD and CCF, all live.

**FE-2 Client profile.** For a selected client: segment, PD and grade with its percentile in the portfolio and within the segment, action level with response time and escalation owner, IFRS 9 stage, exposure and limit, expected loss, 6-month bill vs payment history with the delinquency status per month, early-warning triggers, limit advice; then a what-if form with re-scoring and reason codes. *Verified:* `test_streamlit_app_renders`.

**FE-3 Portfolio agent.** Tool-calling LLM (Azure OpenAI) with one read-only SQL tool (ES-12); offline template mode without an LLM; every step traced; numbers in answers come from query results.

**FE-4 Presentation.** One offline HTML file (React, React Flow, official Azure architecture icons, the client's logo), rebuilt with `make deck` from code-generated figures (`make analytics`).

**FE-5 Production front end (target).** One React application on top of the online endpoint plus a small API for portfolio queries and the agent.

---

## 12. Non-functional requirements

| ID | Requirement | Current |
|---|---|---|
| NF-1 | Local end-to-end run | about 3 minutes (`make local`) |
| NF-2 | Azure training run once the environment image exists | about 7 minutes |
| NF-3 | Batch scoring and monitoring on Azure | about 3 minutes |
| NF-4 | Endpoint latency | about 86 ms per request; alert above 1 s p95 |
| NF-5 | Idle cost | cluster at 0 nodes; endpoint is the only hourly cost and is deleted after use |
| NF-6 | Behaviour freeze on refactors | metrics, backtest, deciles and every client's PD byte-identical to a saved baseline |

---

## 13. Acceptance

The build is accepted when, from a fresh clone: `make setup` installs and downloads; `make check` passes (format, lint, 53 tests); `make local` reproduces test AUC 0.778 (Mac) / 0.779 (Linux) with cut-off 0.167; the four dry-runs build; CI is green; a pushed model change runs on Azure and registers or gates a new version with a valid git SHA.

**Reference results (held-out test set, 5,993 clients):** AUC 0.779 (95% CI 0.765–0.792), Gini 0.557, KS 0.423, ECE 0.0105, Brier 0.135, HL p 0.36; 6 of 7 grades green, grade B amber; at PD ≥ 0.167, 71% of defaults caught while flagging 40% of clients, expected cost 49% below acting on nobody.

---

## 14. Out of scope and roadmap

**Next, with DKV data:** CR-2 and CR-3; bureau data (Creditreform, D&B, Creditsafe, insolvency notices); group view (parent-company exposure, group limits, contagion); industry and macro overlay (default rates by NACE code, freight index, diesel price scenarios); BL-3 scenario LGD; segment-level TTC for SMB vs corporate; lag and velocity features from daily fuel, toll and charging transactions; FE-5.

**Deliberately not:** separate models per risk grade (MR-9); per-segment models on this data volume; resampling the defaults (MR-4); model changes outside the registration gate.

---

## 15. Traceability

| Requirement | Code | Evidence |
|---|---|---|
| DR-3, DR-4 | `creditrisk/quality.py` | `test_quality_gate_*` |
| FR-1–FR-3 | `creditrisk/features.py`, `creditrisk/modeling.py` | `test_feature_engineering_is_finite_and_stateless`, e2e |
| FR-4 | `creditrisk/feature_report.py`, `scripts/risk_analytics.py` | `test_feature_report_*`, `make analytics` |
| MR-1–MR-5, MR-10 | `src/train.py`, `creditrisk/modeling.py` | e2e, `test_pd_floor_wrapper` |
| MR-6, MR-7 | `creditrisk/policy.py` | `test_cost_optimal_threshold_and_grades` |
| MR-8 | `policy.ttc_shift` | `test_ttc_shift_*` |
| VR-1–VR-5 | `creditrisk/metrics.py`, `creditrisk/explain.py`, `src/evaluate.py` | `test_discrimination_and_ks_known_values`, `test_jeffreys_*`, `test_evaluation_quality` |
| VR-6, VR-7 | `evaluate.gate_decision`, `src/register.py` | `test_champion_challenger_*`, `test_quality_gate_blocks_registration`, `test_model_package_contents` |
| BL-1–BL-4, BL-9–BL-11 | `creditrisk/policy.py`, `creditrisk/scoring.py` | e2e batch and endpoint tests |
| BL-5, BL-6 | `creditrisk/migration.py`, `insights.default_forecast` | `test_lifetime_pd_*` |
| BL-7, BL-8 | `creditrisk/portfolio_risk.py` | `test_credit_var_*` |
| BL-12 | `policy.risk_based_price` | `test_risk_based_price_rises_with_pd` |
| BL-13, BL-14 | `creditrisk/insights.py`, `creditrisk/sql/` | `test_batch_scoring_monitoring_insights` |
| SR-1–SR-5 | `pipeline/run_batch_pipeline.py`, `src/serving/score.py`, `deploy/deploy_endpoint.py` | `test_endpoint_scoring_script`, `test_azure_specs_build` |
| OR-1–OR-4 | `creditrisk/observability.py`, `creditrisk/drift.py`, `src/monitor_drift.py`, `infra/alerts.sh` | `test_psi_*` |
| ES-3 | `pyproject.toml` | `test_step_scripts_import_only_creditrisk_and_third_party` |
| ES-8 | `creditrisk/config.py`, `config.yaml` | `test_config_*`, `test_component_defaults_match_config`, `test_cli_pipeline_defaults_match_config` |
| OR-1 | `creditrisk/observability.py` | `test_logging_setup_keeps_azure_sdk_http_logs_quiet` |
| ES-12 | `creditrisk/agent.py`, `infra/setup_azure.sh` | `test_sql_guard_*`, `test_duckdb_external_access_locked`, `test_agent_returns_bad_tool_arguments_to_the_model` |
| FE-2 | `app/streamlit_app.py` | `test_streamlit_app_renders` |
