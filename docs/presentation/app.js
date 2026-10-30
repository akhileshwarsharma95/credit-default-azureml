/* Credit default PD engine - presentation. React 18 + htm (no build step); assets come from build.py. */
const { useState, useEffect, useMemo } = React;
const html = htm.bind(React.createElement);
const I = (k) => window.ICONS[k];
const IMG = (k) => window.IMAGES[k];

const WS = {
  sub: "ff629f6f-e7e7-4106-9364-879520fc6cd7",
  tenant: "99a0cba1-a60c-4d39-bb12-bed50cb310c3",
  rg: "rg-credit-risk",
  name: "mlw-credit-risk",
};
const WSID = `wsid=/subscriptions/${WS.sub}/resourcegroups/${WS.rg}/workspaces/${WS.name}&tid=${WS.tenant}`;
const studio = (run) => `https://ml.azure.com/runs/${run}?${WSID}`;
const STUDIO = {
  model: (v) => `https://ml.azure.com/model/credit-default-model:${v}/details?${WSID}`,
  endpoint: `https://ml.azure.com/endpoints/realtime/credit-risk-fb1504/detail?${WSID}`,
  data: `https://ml.azure.com/data/credit-default-raw/1/details?${WSID}`,
  jobs: `https://ml.azure.com/experiments?${WSID}`,
};
const LinkBtn = ({ href, children, primary }) => html`<a class=${"iconbtn link-btn" + (primary ? " on" : "")} href=${href} target="_blank" rel="noopener">${children} ↗</a>`;

/* ------------------------------------------------------------------ shared bits */
const Kpi = ({ label, value, note, accent }) => html`
  <div class=${"card kpi" + (accent ? " accent" : "")}>
    <div class="label">${label}</div><div class="value">${value}</div>${note && html`<div class="note">${note}</div>`}
  </div>`;
const Svc = ({ icon, name, children }) => html`
  <div class="svc"><img src=${I(icon)} alt="" /><div><b>${name}</b><span>${children}</span></div></div>`;
const Chip = ({ icon, name }) => html`<span class="logo-chip">${icon && html`<img src=${I(icon)} alt="" />`}${name}</span>`;
const Pill = ({ kind, children }) => html`<span class=${"pill " + (kind || "")}>${children}</span>`;
const Seg = ({ options, value, onChange }) => html`
  <div class="seg">${options.map(([k, label]) => html`
    <button key=${k} class=${value === k ? "on" : ""} onClick=${() => onChange(k)}>${label}</button>`)}</div>`;
const Head = ({ title, lead }) => html`<h2>${title}</h2><p class="lead">${lead}</p>`;

function Gallery({ items }) {
  const [cur, setCur] = useState(items[0][0]);
  const item = items.find((i) => i[0] === cur);
  return html`<div>
    <div class="thumbs"><${Seg} options=${items.map(([k, l]) => [k, l])} value=${cur} onChange=${setCur} /></div>
    <figure><img src=${IMG(item[0])} alt=${item[1]} /><figcaption>${item[2]}</figcaption></figure>
  </div>`;
}

/* ------------------------------------------------------------------ architecture diagram (React Flow) */
const RF = window.ReactFlow;
// centre coordinates on a 1280 x 690 canvas; svc nodes are 134 x 72 (w overrides), steps 96 x 54
const NODES = {
  repo: { x: 112, y: 140, icon: "github", t: "GitHub repo", s: "private · main" },
  ci: { x: 112, y: 290, icon: "githubactions", t: "CI workflow", s: "ruff · 53 tests · dry-runs" },
  cd: { x: 112, y: 440, icon: "githubactions", t: "train-and-deploy", s: "OIDC · approval gate" },
  storage: { x: 315, y: 140, icon: "az/storage", t: "Blob storage", s: "versioned data assets" },
  acr: { x: 315, y: 290, icon: "az/container-registry", t: "Container Registry", s: "pinned env image" },
  entra: { x: 315, y: 440, icon: "az/app-registration", t: "Entra ID app", s: "federated OIDC" },
  kv: { x: 315, y: 590, icon: "az/key-vault", t: "Key Vault", s: "workspace secrets" },
  pre: { x: 478, y: 140, step: "1", t: "preprocess", s: "pandera DQ gate" },
  trn: { x: 598, y: 140, step: "2", t: "train", s: "Optuna · calibrate" },
  evl: { x: 718, y: 140, step: "3", t: "evaluate", s: "backtest · SHAP" },
  reg: { x: 838, y: 140, step: "4", t: "register", s: "gated" },
  cluster: { x: 598, y: 290, icon: "az/vm", t: "cpu-cluster", s: "D2ds_v5 · 0–2 nodes" },
  registry: { x: 838, y: 290, icon: "az/machine-learning", t: "Model registry", s: "model + card + baseline" },
  score: { x: 598, y: 420, step: "a", t: "score", s: "PD · grade · EL" },
  drift: { x: 718, y: 420, step: "b", t: "monitor", s: "PSI drift" },
  insights: { x: 838, y: 420, step: "c", t: "insights", s: "SQL · DuckDB" },
  modelmon: { x: 598, y: 590, icon: "az/monitor", t: "Model monitor", s: "daily drift + DQ" },
  endpoint: { x: 838, y: 590, icon: "az/machine-learning", t: "Online endpoint", s: "blue/green · collector" },
  appins: { x: 1075, y: 150, icon: "az/app-insights", t: "App Insights", s: "endpoint · app", w: 104 },
  logs: { x: 1205, y: 150, icon: "az/log-analytics", t: "Log Analytics", s: "KQL", w: 104 },
  alerts: { x: 1140, y: 270, icon: "az/alerts", t: "Monitor alerts", s: "failed runs · p95", w: 104 },
  powerbi: { x: 1205, y: 420, icon: "az/power-bi", t: "Power BI", s: "insight tables", w: 104 },
  app: { x: 1075, y: 590, icon: "az/container-apps", t: "Container Apps", s: "Streamlit cockpit", w: 104 },
  openai: { x: 1205, y: 590, icon: "az/azure-openai", t: "Azure OpenAI", s: "text-to-SQL agent", w: 104 },
};
const ZONES = [
  ["z-gh", 14, 40, 196, 640, "GitHub", "github", "gh"],
  ["z-az", 226, 40, 1044, 640, "Azure subscription › rg-credit-risk · Germany West Central", "az/resource-group", "az"],
  ["z-aml", 414, 78, 586, 590, "Azure ML workspace · mlw-credit-risk", "az/machine-learning", "aml"],
  ["z-obs", 1020, 78, 240, 250, "Observability", null, "sub"],
  ["z-apps", 1020, 345, 240, 323, "Apps & reporting", null, "sub"],
];
// [source, side, target, side, flows, label, dashed]
const EDGES = [
  ["repo", "b", "ci", "t", ["cicd"], "push / PR"],
  ["ci", "b", "cd", "t", ["cicd"], "on green"],
  ["cd", "r", "entra", "l", ["cicd"], "OIDC"],
  ["entra", "r", "pre", "l", ["cicd", "train"]],
  ["storage", "r", "pre", "l", ["train"], "data v1"],
  ["pre", "r", "trn", "l", ["train"]],
  ["trn", "r", "evl", "l", ["train"]],
  ["evl", "r", "reg", "l", ["train"]],
  ["acr", "r", "cluster", "l", ["train"], "env image"],
  ["cluster", "t", "trn", "b", ["train"], "runs steps", true],
  ["reg", "b", "registry", "t", ["train"], "gate passed"],
  ["registry", "l", "evl", "b", ["train"], "champion", true],
  ["registry", "b", "score", "t", ["monitor"], "latest version"],
  ["score", "r", "drift", "l", ["monitor"]],
  ["drift", "r", "insights", "l", ["monitor"]],
  ["insights", "r", "powerbi", "l", ["monitor"], "tables"],
  ["registry", "r", "endpoint", "t", ["serve"], "blue/green"],
  ["endpoint", "l", "modelmon", "r", ["serve", "monitor"], "collector"],
  ["endpoint", "r", "app", "l", ["serve"], "REST"],
  ["app", "r", "openai", "l", ["serve"]],
  ["appins", "r", "logs", "l", ["monitor"]],
  ["logs", "b", "alerts", "r", ["monitor"]],
];
const FLOWS = [["all", "Everything"], ["cicd", "CI/CD"], ["train", "Training"], ["monitor", "Batch & monitoring"], ["serve", "Real-time & app"]];
const FLOW_NODES = {};
for (const [s, , t, , fl] of EDGES) for (const f of fl) (FLOW_NODES[f] = FLOW_NODES[f] || new Set()).add(s).add(t);
const DETAILS = {
  repo: ["GitHub repository", "Private repo, everything as code: 7 YAML components, SDK v2 submitters, locked environments, infra scripts, tests, runbook."],
  ci: ["CI workflow (every push / PR)", "ruff format + lint, 53 tests (end to end on synthetic data) and a dry-run of every Azure ML spec, so a broken component never reaches Azure. Dependencies install from uv.lock."],
  cd: ["train-and-deploy workflow", "On push to main (model code) or manual dispatch: CI, then submit the Azure ML pipeline, wait, publish the model card to the job summary. Deployment only on manual dispatch, behind the GitHub production environment."],
  entra: ["Microsoft Entra ID app registration", "Workload identity federation: GitHub exchanges its OIDC token for an Azure token. No stored client secret. Contributor scoped to rg-credit-risk only."],
  storage: ["Storage account (workspace datastore)", "Raw UCI file registered as data asset credit-default-raw with a sha256 tag; batches registered as their own assets. Every model version points to its data version."],
  acr: ["Azure Container Registry", "Built once from the locked conda environment (environments/), reused by every job and the endpoint. Same image for training and scoring = no train/serve skew."],
  kv: ["Key Vault", "Created with the workspace; holds workspace and connection secrets. Nothing secret lives in the repo or in GitHub."],
  pre: ["Step 1 · preprocess", "pandera hard checks (types, ranges, nulls) fail the run; soft checks are reported. Documented remap of undocumented codes, de-duplication, stratified split before anything is fitted."],
  trn: ["Step 2 · train", "Optuna search; out-of-fold comparison of monotonic LightGBM vs unconstrained LightGBM vs logistic scorecard; isotonic calibration + 0.1% PD floor; cost-based cut-off. All logged to MLflow."],
  evl: ["Step 3 · evaluate", "Test-set AUC/Gini/KS with bootstrap CIs, calibration, Jeffreys backtest per grade, fairness, SHAP. Loads the registered champion and scores it on the same test set."],
  reg: ["Step 4 · register", "Champion/challenger gate: minimum AUC, calibration limit, not worse than the champion by more than 0.005 AUC. Registers model + model card + evaluation evidence + drift baseline."],
  cluster: ["Compute cluster cpu-cluster", "Standard_D2ds_v5 (2 vCPU, 8 GB, DDSv5), 0–2 nodes, scales to zero after 3 minutes idle: cents per run, nothing billed when idle."],
  registry: ["Model registry", "credit-default-model, versioned through MLflow. Tags: test AUC, Gini, ECE, threshold, git SHA, data version. Artifacts: model, MODEL_CARD.md, evaluation/, drift reference profile."],
  score: ["Batch step a · score", "Scores a portfolio batch with the registered model: PD, grade, EL, IFRS 9 stage, early-warning triggers, limit advice, reason codes."],
  drift: ["Batch step b · monitor drift", "PSI per feature, score and rating grade vs the baseline stored with the model; writes drift_summary.json and alert.json. Warn ≥ 0.10, alert ≥ 0.25."],
  insights: ["Batch step c · portfolio insights", "DuckDB SQL (creditrisk/sql/): KPIs, grade and segment summaries, IFRS 9 staging, watchlist, limit actions, roll rates and a 3-month Markov projection."],
  modelmon: ["Azure ML model monitor", "Daily signal on the endpoint's collected inputs/outputs: data drift and data quality vs training data, e-mail alert."],
  endpoint: ["Managed online endpoint", "Blue/green deployments with smoke test, optional canary %, rollback. Data collector on inputs/outputs. Returns PD, grade, action, EL, stage and reason codes."],
  appins: ["Application Insights", "Request logs, latency and failures from the endpoint and the app; OpenTelemetry spans from scoring and the agent."],
  logs: ["Log Analytics", "Workspace behind App Insights; KQL over traces/requests for incidents and dashboards."],
  alerts: ["Azure Monitor alerts", "aml-failed-runs (any failed job) and endpoint-latency-p95 (> 500 ms warn, > 1 s alert), wired by infra/alerts.sh."],
  powerbi: ["Power BI-ready tables", "Insight outputs are flat tables (parquet/CSV) designed for a Power BI model: grade, segment, stage, watchlist, limit actions."],
  app: ["Azure Container Apps", "Streamlit credit-risk cockpit, consumption plan (scales to zero). Uses the registered model package or the endpoint."],
  openai: ["Azure OpenAI", "Tool-calling agent that answers portfolio questions with one SELECT-only, read-only DuckDB query; traced; offline template fallback."],
};


const NODE_INFO = {
  repo: [[".github/", "pyproject.toml", "uv.lock", "Makefile"], "Everything is code, including the Azure setup and the environments."],
  ci: [[".github/workflows/ci.yml", "tests/"], "Nothing reaches Azure without passing format, lint, 53 tests and spec dry-runs."],
  cd: [[".github/workflows/train-and-deploy.yml"], "Training runs on every model-code push; deployment needs a human approval."],
  storage: [["pipeline/azure_utils.py (ensure_data_asset)"], "Same file, same version: the sha256 tag makes data lineage provable."],
  acr: [["environments/conda.yaml", "scripts/lock_env.py"], "Built once from the lock; every step and the endpoint run the tested versions."],
  entra: [["infra/setup_azure.sh"], "No stored secrets: GitHub gets short-lived tokens, Contributor on one resource group."],
  kv: [["(created with the workspace)"], "Secrets never live in the repo or in GitHub."],
  pre: [["src/preprocess.py", "src/creditrisk/quality.py", "components/preprocess.yml"], "Bad data stops the run; undocumented codes are cleaned and reported."],
  trn: [["src/train.py", "src/creditrisk/modeling.py", "src/creditrisk/features.py"], "Monotonic LightGBM: same AUC as unconstrained, explanations that always make sense."],
  evl: [["src/evaluate.py", "src/creditrisk/metrics.py", "src/creditrisk/explain.py"], "Scores the registered champion on the same test set: that is the gate's evidence."],
  reg: [["src/register.py", "src/creditrisk/model_card.py"], "Registered only if better than the champion; the model card travels with it."],
  cluster: [["pipeline/azure_utils.py (ensure_compute)", "config.yaml technical.azure"], "Scales to zero; on its own quota family so the endpoint can never block it."],
  registry: [["src/register.py", "scripts/fetch_model.py"], "Every version carries its model card, evidence, drift baseline, git SHA and data version."],
  score: [["src/score_batch.py", "src/creditrisk/scoring.py"], "One scoring function for batch, endpoint and app: no training/serving skew."],
  drift: [["src/monitor_drift.py", "src/creditrisk/drift.py"], "PSI against the baseline stored with the model: alert at 0.25."],
  insights: [["src/portfolio_insights.py", "src/creditrisk/insights.py", "src/creditrisk/sql/"], "Turns PDs into EL, stages, watchlist, limits, Markov lifetime PDs and the loss distribution."],
  modelmon: [["monitoring/setup_model_monitor.py"], "Daily drift and data-quality checks on what the endpoint actually receives."],
  endpoint: [["deploy/deploy_endpoint.py", "src/serving/score.py"], "Blue/green with a smoke test; bad input gets a 400 with the required fields."],
  appins: [["src/creditrisk/observability.py"], "Requests, latency and OpenTelemetry spans from the endpoint and the app."],
  logs: [["(workspace behind App Insights)"], "KQL over traces and requests for incidents."],
  alerts: [["infra/alerts.sh", "docs/RUNBOOK.md"], "Every alert points to a runbook playbook with an owner."],
  powerbi: [["src/creditrisk/sql/"], "Flat tables per batch, ready for a Power BI model."],
  app: [["app/streamlit_app.py", "Dockerfile", "scripts/deploy_app.sh"], "The credit officer's view: client profile, what-if, cut-off economics, drift simulator."],
  openai: [["src/creditrisk/agent.py"], "One SELECT-only tool on a locked-down DuckDB; every call traced; labelled as AI (Art. 50)."],
};

const SIDES = { t: "Top", r: "Right", b: "Bottom", l: "Left" };
const Handles = () => Object.entries(SIDES).map(([k, p]) => html`
  <${RF.Handle} key=${"s" + k} id=${"s-" + k} type="source" position=${RF.Position[p]} isConnectable=${false} />
  <${RF.Handle} key=${"t" + k} id=${"t-" + k} type="target" position=${RF.Position[p]} isConnectable=${false} />`);

const SvcNode = ({ data }) => html`<div class=${"rf-svc" + (data.sel ? " sel" : "") + (data.dim ? " dim" : "")} style=${{ width: data.w + "px" }}>
  <${Handles} /><img src=${I(data.icon)} alt="" /><b>${data.t}</b><span>${data.s}</span></div>`;
const StepNode = ({ data }) => html`<div class=${"rf-step" + (data.sel ? " sel" : "") + (data.dim ? " dim" : "")}>
  <${Handles} /><b>${data.step}  ${data.t}</b><span>${data.s}</span></div>`;
const ZoneNode = ({ data }) => html`<div class=${"rf-zone " + data.cls} style=${{ width: data.w + "px", height: data.h + "px" }}>
  <div class="rf-zone-label">${data.icon && html`<img src=${I(data.icon)} alt="" />`}${data.label}</div></div>`;

function FlowEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, label }) {
  const [path, lx, ly] = RF.getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 10 });
  const style = {
    stroke: data.hot ? "var(--accent)" : "var(--edge)", strokeWidth: data.hot ? 2.4 : 1.6,
    strokeDasharray: data.dash ? "5 4" : undefined, opacity: data.dim ? 0.15 : 1, transition: "opacity .25s, stroke .25s",
  };
  return html`<${React.Fragment}>
    <${RF.BaseEdge} id=${id} path=${path} markerEnd=${markerEnd} style=${style} />
    ${!data.dim && html`<circle r=${data.hot ? 4.5 : 3} fill=${data.hot ? "var(--accent)" : "var(--edge)"} opacity=${data.hot ? 1 : 0.7}>
      <animateMotion dur=${data.hot ? "1.8s" : "4s"} repeatCount="indefinite" path=${path} begin=${data.delay} /></circle>`}
    ${label && html`<${RF.EdgeLabelRenderer}>
      <div class=${"rf-elabel" + (data.dim ? " dim" : "")} style=${{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}>${label}</div>
    </${RF.EdgeLabelRenderer}>`}
  </${React.Fragment}>`;
}
const NODE_TYPES = { svc: SvcNode, step: StepNode, zone: ZoneNode };
const EDGE_TYPES = { flow: FlowEdge };

/* ------------------------------------------------------------------ business overview */
// Follows one real client (7926, scored live by the endpoint) through every stage.
const BIZ = {
  data: { x: 110, y: 250, n: "1", t: "Client behaviour", f: "6 months of history",
    q: "What do we know about the client?",
    formula: "limit · 6 bills · 6 payments · 6 payment statuses",
    how: ["Last 6 monthly statements: balance, amount paid, months late", "Turned into behaviour signals: how often late, how recently, repayment ratio, utilisation"],
    why: "How a client pays predicts default far better than how much they owe (delinquency signals are 40× more informative than bill amounts).",
    meaning: "The model never sees sex, age or marital status; only behaviour.",
    ex: "2 months late in September, 3 late payments in the last 3 months, last late payment 1 month ago." },
  pd: { x: 340, y: 110, n: "2", t: "Probability of default", f: "PD = P(default next month)",
    q: "How likely is the client not to pay?",
    formula: "PD = calibrated model output, floored at 0.1%",
    how: ["Monotonic LightGBM ranks clients by risk", "Isotonic calibration turns scores into true probabilities", "Floor 0.1%: nobody is risk-free (CRR3)"],
    why: "Every money decision downstream multiplies by the PD, so it must be right in level, not just in order. Calibration is checked per grade.",
    meaning: "PD 20% = out of 100 similar clients, about 20 miss next month's payment.",
    ex: "PD 83.3%: out of 100 clients like 7926, about 83 would miss the next payment." },
  ead: { x: 340, y: 250, n: "3", t: "Exposure at default", f: "EAD = drawn + CCF × undrawn",
    q: "How much would they owe if they default?",
    formula: "EAD = drawn balance + 50% × unused limit",
    how: ["Drawn = current statement balance", "CCF 50%: share of the unused limit typically drawn before default"],
    why: "Clients in trouble tend to use their remaining limit before they stop paying, so today's balance understates the exposure.",
    meaning: "The unused limit is risk too, which is why reducing limits reduces loss.",
    ex: "EAD NT$ 41,450." },
  lgd: { x: 340, y: 390, n: "4", t: "Loss given default", f: "LGD = 1 − recovery",
    q: "If they default, how much do we lose?",
    formula: "LGD = 60% (unsecured card)",
    how: ["Today a documented assumption (no loss data in UCI)", "Target: scenario-weighted cure / collateral recovery / total loss"],
    why: "Recoveries (collections, deposits, guarantees) reduce the loss; without collateral most of the exposure is lost.",
    meaning: "Deposits and guarantees lower LGD, which is why 'Intervene' asks for them.",
    ex: "LGD 60%." },
  grade: { x: 590, y: 70, n: "5", t: "Rating grade", f: "PD → A … G",
    q: "Which risk class is the client in?",
    formula: "A 0–5% · B 5–8% · C 8–12% · D 12–20% · E 20–35% · F 35–55% · G 55%+",
    how: ["Fixed PD bands (the master scale)", "Each grade is backtested: predicted vs observed default rate"],
    why: "Policies, reports and reviews need stable classes, not raw probabilities.",
    meaning: "'No limit increase above D' is a policy a credit team can run.",
    ex: "Grade G (PD above 55%)." },
  el: { x: 590, y: 280, n: "6", t: "Expected loss", f: "EL = PD × LGD × EAD",
    q: "What does this client cost us on average?",
    formula: "EL = PD × LGD × EAD",
    how: ["Multiply the three building blocks per client", "Sum over clients for the portfolio"],
    why: "It turns risk into money: comparable across clients, segments and products.",
    meaning: "EL is a cost of doing business: it has to be covered by price and by provisions.",
    ex: "0.833 × 0.60 × 41,450 = NT$ 20,725." },
  action: { x: 840, y: 90, n: "7", t: "Decision & escalation", f: "act if PD ≥ 1 / (1 + 5)",
    q: "Do we act, and who acts by when?",
    formula: "cut-off = cost of false alarm / (false alarm + missed default) = 1 / 6 = 0.167",
    how: ["Missing a defaulter costs about 5× a needless intervention", "Plus early-warning triggers (new delinquency, payment shortfall …)", "Levels: Standard → Watch → Intervene → Default, each with an owner and a deadline"],
    why: "The cut-off comes from business costs, not from the model; change the costs and the policy follows.",
    meaning: "At 0.167 we catch 71% of defaulters while contacting 40% of clients, 49% cheaper than doing nothing.",
    ex: "Intervene (PD ≥ 0.167; triggers: paid less than 10% of the last bill, grade F/G): credit risk, within 48 hours." },
  limits: { x: 840, y: 250, n: "8", t: "Limits & pricing", f: "price = funding + EL + capital + costs",
    q: "How much credit, at what price?",
    formula: "price = funding + EL rate + capital × cost of capital + opex + margin",
    how: ["Reduce undrawn limits for F/G, raise limits for A/B at high utilisation", "Price per grade covers expected and unexpected loss"],
    why: "Lower limits cut EAD and so EL; risk-based prices keep good clients and charge for risk.",
    meaning: "For DKV: card limits, payment terms (14 vs 30 days), fees and deposits by grade.",
    ex: "Limit reduced to NT$ 37,000; grade G price about 46% per unit of exposure." },
  provision: { x: 840, y: 400, n: "9", t: "Provision (IFRS 9)", f: "stage 1: 12 m · stage 2: lifetime",
    q: "How much do we set aside in the balance sheet?",
    formula: "ECL = horizon PD × LGD × EAD (12 months in stage 1, lifetime in stage 2, PD = 1 in stage 3)",
    how: ["Stage 2 if 30+ days late or grade F/G (significant increase in risk)", "Stage 3 at 90+ days late", "Lifetime PDs from monthly migration between delinquency buckets (Markov)"],
    why: "Accounting must recognise expected losses early, and more of them once risk has risen.",
    meaning: "Moving a client to stage 2 increases the provision and hits profit immediately.",
    ex: "Stage 2: lifetime ECL NT$ 24,755." },
  portfolio: { x: 1080, y: 250, n: "10", t: "Portfolio & capital", f: "loss at 99.9% − EL = capital",
    q: "How much risk does the whole book carry?",
    formula: "loss at 99.9% = Σ EAD × LGD × PD in a severe downturn (single-factor model)",
    how: ["Clients default together in a recession (correlation)", "Expected loss vs loss in a 1-in-1000 year", "Concentration by client and segment"],
    why: "Average losses are priced in; bad years are covered by capital. The total matters more than any one client.",
    meaning: "Test portfolio: EL NT$ 70.6 M, 99.9% loss NT$ 132.3 M, so NT$ 61.6 M of capital.",
    ex: "Client 7926 contributes NT$ 20,725 to the portfolio's expected loss." },
  monitor: { x: 1080, y: 410, n: "11", t: "Monitoring & feedback", f: "PSI ≥ 0.25 → alert",
    q: "Is the model still right?",
    formula: "PSI = Σ (actual − expected) × ln(actual / expected), per feature and score",
    how: ["Weekly: compare new clients with the training profile", "Monthly: backtest predicted vs actual defaults per grade", "Alert → runbook → retrain → the champion/challenger gate decides"],
    why: "Economies and portfolios change; a model is only as good as its last check.",
    meaning: "The simulated downturn raised a drift alert (score PSI 0.34): credit management is told before losses show.",
    ex: "Next month, 7926's actual outcome feeds the backtest of grade G." },
};
const BIZ_EDGES = [
  ["data", "r", "pd", "l", "signals"], ["data", "r", "ead", "l"], ["data", "r", "lgd", "l", "loss history"],
  ["pd", "t", "grade", "l"], ["pd", "r", "el", "l"], ["ead", "r", "el", "l"], ["lgd", "r", "el", "l"],
  ["grade", "r", "action", "l"], ["pd", "r", "action", "l", "cut-off"],
  ["el", "r", "limits", "l"], ["grade", "r", "limits", "t"], ["el", "r", "provision", "l"],
  ["el", "r", "portfolio", "l"], ["provision", "r", "portfolio", "l"], ["limits", "r", "portfolio", "l"],
  ["portfolio", "b", "monitor", "t"], ["action", "r", "monitor", "t", "outcomes"],
  ["monitor", "b", "pd", "b", "retrain", true],
];
const BIZ_ORDER = ["data", "pd", "ead", "lgd", "grade", "el", "action", "limits", "provision", "portfolio", "monitor"];

const BizNode = ({ data }) => html`<div class=${"rf-biz" + (data.sel ? " sel" : "") + (data.dim ? " dim" : "")}>
  <${Handles} /><div class="rf-biz-n">${data.n}</div><b>${data.t}</b><span>${data.f}</span></div>`;
const BIZ_NODE_TYPES = { biz: BizNode };

function BusinessOverview({ dark }) {
  const [sel, setSel] = useState("data");
  const [play, setPlay] = useState(false);
  useEffect(() => {
    if (!play) return undefined;
    let i = BIZ_ORDER.indexOf(sel);
    const t = setInterval(() => { i = (i + 1) % BIZ_ORDER.length; setSel(BIZ_ORDER[i]); }, 4500);
    return () => clearInterval(t);
  }, [play]);
  const step = BIZ_ORDER.indexOf(sel);
  const nodes = useMemo(() => Object.entries(BIZ).map(([k, b]) => ({
    id: k, type: "biz", position: { x: b.x - 90, y: b.y - 40 },
    data: { ...b, sel: sel === k, dim: play && BIZ_ORDER.indexOf(k) > step },
  })), [sel, play]);
  const edges = useMemo(() => BIZ_EDGES.map(([s, ss, t, ts, label, dash], i) => {
    const hot = t === sel;
    return {
      id: "b" + i, source: s, sourceHandle: "s-" + ss, target: t, targetHandle: "t-" + ts, type: "flow", label,
      data: { hot, dim: play && BIZ_ORDER.indexOf(t) > step, dash, delay: (i % 5) * 0.35 + "s" },
      markerEnd: { type: RF.MarkerType.ArrowClosed, width: 16, height: 16, color: hot ? (dark ? "#4aa3ff" : "#0078d4") : (dark ? "#6d7d90" : "#8a99ab") },
    };
  }), [sel, play, dark]);
  const b = BIZ[sel];
  return html`<div>
    <${Head} title="Business overview: from behaviour to decisions"
      lead="What we compute at each stage, how, why this way, and what it means for the business. Follow one real client (7926, scored live by the endpoint) through the whole chain." />
    <div class="row" style=${{ marginBottom: "12px" }}>
      <button class=${"iconbtn" + (play ? " on" : "")} onClick=${() => setPlay(!play)}>${play ? "❚❚ Pause" : "▶ Play the story"}</button>
      <button class="iconbtn" onClick=${() => { setPlay(false); setSel(BIZ_ORDER[Math.max(0, step - 1)]); }}>← Previous</button>
      <button class="iconbtn" onClick=${() => { setPlay(false); setSel(BIZ_ORDER[Math.min(BIZ_ORDER.length - 1, step + 1)]); }}>Next →</button>
      <span class="muted small">Stage ${step + 1} of ${BIZ_ORDER.length} · click any stage</span>
    </div>
    <div class="arch-layout">
      <div class="card rf-wrap biz-wrap">
        <${RF.ReactFlow} nodes=${nodes} edges=${edges} nodeTypes=${BIZ_NODE_TYPES} edgeTypes=${EDGE_TYPES}
          colorMode=${dark ? "dark" : "light"} fitView fitViewOptions=${{ padding: 0.06 }} minZoom=${0.2} maxZoom=${2.5}
          nodesDraggable=${false} nodesConnectable=${false} elementsSelectable=${false}
          zoomOnScroll=${false} zoomOnDoubleClick=${false} preventScrolling=${false}
          onNodeClick=${(e, n) => { setPlay(false); setSel(n.id); }}>
          <${RF.Controls} showInteractive=${false} position="bottom-right" />
        </${RF.ReactFlow}>
      </div>
      <aside class="card side-panel">
        <div class="side-head"><span class="n">${b.n}</span><div><div class="muted small">${b.q}</div><h3>${b.t}</h3></div></div>
        <div class="formula">${b.formula}</div>
        <div class="side-label">How we compute it</div>
        <ul class="clean small">${b.how.map((h) => html`<li key=${h}>${h}</li>`)}</ul>
        <div class="side-label">Why this way</div><p class="small" style=${{ margin: 0 }}>${b.why}</p>
        <div class="side-label">What it means for the business</div><p class="small" style=${{ margin: 0 }}>${b.meaning}</p>
        <div class="callout small section"><b>Client 7926:</b> ${b.ex}</div>
      </aside>
    </div>
  </div>`;
}

function Architecture({ dark }) {
  const [flow, setFlow] = useState("all");
  const [sel, setSel] = useState("registry");
  const [play, setPlay] = useState(false);
  useEffect(() => {
    if (!play) return undefined;
    const order = FLOWS.map((f) => f[0]).slice(1);
    let i = Math.max(0, order.indexOf(flow));
    setFlow(order[i]);
    const t = setInterval(() => { i = (i + 1) % order.length; setFlow(order[i]); }, 5000);
    return () => clearInterval(t);
  }, [play]);
  const nodeOn = (k) => flow === "all" || FLOW_NODES[flow].has(k);
  const nodes = useMemo(() => [
    ...ZONES.map(([id, x, y, w, h, label, icon, cls]) => ({
      id, type: "zone", position: { x, y }, data: { w, h, label, icon, cls }, selectable: false, draggable: false, zIndex: -1,
    })),
    ...Object.entries(NODES).map(([k, n]) => {
      const w = n.step ? 96 : n.w || 134;
      const h = n.step ? 54 : 72;
      return {
        id: k, type: n.step ? "step" : "svc", position: { x: n.x - w / 2, y: n.y - h / 2 },
        data: { ...n, w, sel: sel === k, dim: !nodeOn(k) },
      };
    }),
  ], [flow, sel]);
  const edges = useMemo(() => EDGES.map(([s, ss, t, ts, fl, label, dash], i) => {
    const on = flow === "all" || fl.includes(flow);
    const hot = flow !== "all" && on;
    return {
      id: "e" + i, source: s, sourceHandle: "s-" + ss, target: t, targetHandle: "t-" + ts, type: "flow", label,
      data: { hot, dim: !on, dash, delay: (i % 5) * 0.35 + "s" },
      markerEnd: { type: RF.MarkerType.ArrowClosed, width: 16, height: 16, color: hot ? (dark ? "#4aa3ff" : "#0078d4") : (dark ? "#6d7d90" : "#8a99ab") },
    };
  }), [flow, dark]);
  const [title, body] = DETAILS[sel];
  return html`<div>
    <${Head} title="Architecture"
      lead="Everything runs inside one resource group: GitHub drives it through OIDC, Azure ML runs the pipelines and hosts the registry and endpoint, Azure Monitor watches it." />
    <div class="row" style=${{ marginBottom: "12px" }}>
      <${Seg} value=${flow} onChange=${(f) => { setPlay(false); setFlow(f); }} options=${FLOWS} />
      <button class=${"iconbtn" + (play ? " on" : "")} onClick=${() => setPlay(!play)}>${play ? "❚❚ Pause tour" : "▶ Play tour"}</button>
      <span class="muted small">Click any component for details · drag to pan</span>
    </div>
    <div class="arch-layout">
      <div class="card rf-wrap">
        <${RF.ReactFlow} nodes=${nodes} edges=${edges} nodeTypes=${NODE_TYPES} edgeTypes=${EDGE_TYPES}
          colorMode=${dark ? "dark" : "light"} fitView fitViewOptions=${{ padding: 0.02 }} minZoom=${0.2} maxZoom=${2.5}
          nodesDraggable=${false} nodesConnectable=${false} elementsSelectable=${false}
          zoomOnScroll=${false} zoomOnDoubleClick=${false} preventScrolling=${false}
          onNodeClick=${(e, n) => n.type !== "zone" && setSel(n.id)}>
          <${RF.Controls} showInteractive=${false} position="bottom-right" />
        </${RF.ReactFlow}>
      </div>
      <aside class="card side-panel">
        <div class="side-head">
          ${NODES[sel].icon ? html`<img src=${I(NODES[sel].icon)} alt="" />` : html`<span class="n">${NODES[sel].step}</span>`}
          <div><div class="muted small">Selected component</div><h3>${title}</h3></div>
        </div>
        <p class="small">${body}</p>
        <div class="callout small">${NODE_INFO[sel][1]}</div>
        <div class="side-label">In the repo</div>
        <div class="file-list">${NODE_INFO[sel][0].map((f) => html`<code key=${f}>${f}</code>`)}</div>
        <div class="side-label">Part of</div>
        <div class="row">${FLOWS.slice(1).filter(([f]) => FLOW_NODES[f].has(sel)).map(([f, l]) => html`<button key=${f} class="pill accent pill-btn" onClick=${() => { setPlay(false); setFlow(f); }}>${l}</button>`)}</div>
      </aside>
    </div>
  </div>`;
}

/* ------------------------------------------------------------------ tabs */
function Overview() {
  return html`<div>
    <div class="prepared"><img src=${I("dkv-mobility")} alt="DKV Mobility" /><span>Prepared for <b>DKV Mobility</b> · Senior Data Scientist case study</span></div>
    <${Head} title="Credit default PD engine on Azure ML"
      lead="I built a model that predicts whether a credit-card client misses next month's payment (UCI, 30,000 clients) and runs as a governed Azure ML SDK v2 pipeline: preprocess → train → evaluate → register, on a compute cluster, with the model in the workspace registry." />
    <div class="grid g4">
      <${Kpi} accent label="Test ROC AUC" value="0.779" note="95% CI 0.765 – 0.792 · 5,993 clients" />
      <${Kpi} accent label="Gini / KS" value="0.557 / 0.423" note="monotonic LightGBM, calibrated" />
      <${Kpi} accent label="Calibration error" value="0.0105" note="ECE · Brier 0.135 · HL p 0.36" />
      <${Kpi} accent label="Defaults caught" value="71%" note="flagging 40% of clients at PD ≥ 0.167" />
    </div>
    <div class="grid g3 section">
      <div class="card"><h3>The brief</h3><ul class="clean">
        <li>Load, clean and preprocess with best practices</li>
        <li>Train and log training metrics</li>
        <li>Load the registered model + test data, evaluate, log metrics and plots</li>
        <li>Pipeline with SDK v2 (or YAML + CLI) on a compute cluster</li>
        <li>Register the model · bonus: YAML components, endpoint</li></ul></div>
      <div class="card"><h3>What I delivered on top</h3><ul class="clean">
        <li>Champion/challenger gate with a model card per version</li>
        <li>PD backtest per grade, fairness, SHAP reason codes</li>
        <li>Batch scoring + PSI drift monitor, blue/green endpoint</li>
        <li>Expected loss, IFRS 9 staging, early warning, limit steering</li>
        <li>Streamlit cockpit with a guarded "talk to your portfolio" agent</li></ul></div>
      <div class="card"><h3>Design principles</h3><ul class="clean">
        <li><b>One source of truth:</b> 7 YAML components shared by SDK and CLI</li>
        <li><b>No train/serve skew:</b> features live inside the model pipeline</li>
        <li><b>Reproducible:</b> locked env, versioned data, git SHA on every model</li>
        <li><b>Decision-first:</b> calibrated PDs, cut-off from misclassification costs</li>
        <li><b>Operable:</b> alerts, runbook, rollback in one command</li></ul></div>
    </div>
    <div class="card section"><h3>Stack</h3><div class="row">
      <${Chip} icon="az/machine-learning" name="Azure Machine Learning" />
      <${Chip} icon="az/app-insights" name="Application Insights" />
      <${Chip} icon="az/container-apps" name="Container Apps" />
      <${Chip} icon="az/azure-openai" name="Azure OpenAI" />
      <${Chip} icon="github" name="GitHub" />
      <${Chip} icon="githubactions" name="GitHub Actions" />
      <${Chip} icon="python" name="Python 3.10" />
      <${Chip} icon="lightgbm" />
      <${Chip} icon="scikitlearn" name="scikit-learn" />
      <${Chip} icon="optuna" />
      <${Chip} icon="mlflow" name="MLflow" />
      <${Chip} icon="pandas" name="pandas · pandera" />
      <${Chip} icon="duckdb" name="DuckDB SQL" />
      <${Chip} icon="streamlit" name="Streamlit" />
      <${Chip} icon="opentelemetry" name="OpenTelemetry" />
      <${Chip} icon="docker" name="Docker" />
      <${Chip} name="SHAP" />
    </div></div>
  </div>`;
}

const STEPS = [
  ["1", "preprocess", "Data quality gate", ["pandera hard checks fail the run; soft checks reported", "Documented remap of undocumented EDUCATION / MARRIAGE / PAY codes", "De-duplication; PAY_0 → PAY_1", "Stratified 80/20 split before anything is fitted"], "out: train / test parquet, quality_report"],
  ["2", "train", "Model selection + calibration", ["Optuna (25 trials) on monotonic LightGBM", "5-fold OOF vs unconstrained LightGBM and a logistic scorecard", "Isotonic calibration, 0.1% PD floor", "Cut-off = cost_fp / (cost_fp + cost_fn) = 0.167"], "out: MLflow model, CV metrics"],
  ["3", "evaluate", "Validation evidence", ["AUC / Gini / KS with bootstrap CIs", "Calibration: ECE, slope, Hosmer-Lemeshow", "Jeffreys backtest per rating grade", "Fairness check · SHAP · 12 charts", "Scores the registered champion on the same test set"], "out: metrics, plots, gate decision"],
  ["4", "register", "Champion / challenger gate", ["Min AUC + calibration limit", "Not worse than champion by > 0.005 AUC", "Registers model + MODEL_CARD.md + evidence + drift baseline", "Tags: AUC, Gini, ECE, threshold, git SHA, data version"], "out: credit-default-model:v"],
];

function Training() {
  return html`<div>
    <${Head} title="Core: the training pipeline"
      lead="Four YAML components wired with the Python SDK v2 (pipeline/run_pipeline.py) or the CLI (pipeline/training_pipeline.yml). The submitter creates the cluster and the versioned data asset, submits, streams logs and fetches the registration result." />
    <div class="flow">
      ${STEPS.map(([n, name, title, items, meta]) => html`<div class="stage" key=${n}>
        <h3><span class="n">${n}</span>${name}</h3>
        <div class="small" style=${{ fontWeight: 600, marginBottom: "6px" }}>${title}</div>
        <ul class="clean small">${items.map((i) => html`<li key=${i}>${i}</li>`)}</ul>
        <div class="meta">${meta}</div></div>`)}
    </div>
    <div class="grid g2 section">
      <div class="card"><h3>Out-of-fold model selection (5-fold, training set)</h3>
        <table><thead><tr><th>Candidate</th><th class="num">CV AUC</th><th class="num">CV Gini</th><th class="num">Brier</th></tr></thead>
        <tbody>
          <tr class="hl"><td>LightGBM, monotonic <${Pill} kind="accent">selected</${Pill}></td><td class="num">0.789</td><td class="num">0.579</td><td class="num">0.133</td></tr>
          <tr><td>LightGBM, unconstrained</td><td class="num">0.789</td><td class="num">0.578</td><td class="num">0.133</td></tr>
          <tr><td>Logistic scorecard</td><td class="num">0.771</td><td class="num">0.543</td><td class="num">0.137</td></tr>
        </tbody></table>
        <p class="small muted" style=${{ marginBottom: 0 }}>Monotonic constraints cost no accuracy and make the model explainable to credit officers and validators. SEX, MARRIAGE and AGE are not inputs; they are used only for the fairness check.</p>
      </div>
      <div class="card"><h3>Engineering choices</h3><ul class="clean small">
        <li><b>Features inside the model pipeline</b> (utilisation, repayment ratios, delinquency patterns) so batch, endpoint and app score identically</li>
        <li><b>Locked environment</b> (environments/train-lock.txt) built once into ACR; same image for every step</li>
        <li><b>Data lineage:</b> raw file → data asset with sha256 tag → model tag <code>credit-default-raw:1</code></li>
        <li><b>MLflow everywhere:</b> params, metrics, plots and the model, natively in the workspace</li>
        <li><b>Same code locally:</b> <code>make local</code> runs every step with a SQLite MLflow registry (~3 min)</li>
        <li><b>One config:</b> business and technical parameters in <code>src/creditrisk/config.yaml</code></li>
        <li><b>Retraining:</b> monthly schedule (<code>--schedule "0 6 1 * *"</code>), drift-triggered runs; the gate decides</li>
      </ul></div>
    </div>
  </div>`;
}

const GRADES = [
  ["A", "0–5%", 305, "2.0%", "2.7%", "good"], ["B", "5–8%", 1238, "7.5%", "5.8%", "warn"],
  ["C", "8–12%", 798, "9.6%", "10.6%", "good"], ["D", "12–20%", 1356, "16.9%", "15.4%", "good"],
  ["E", "20–35%", 1122, "23.0%", "23.9%", "good"], ["F", "35–55%", 589, "42.1%", "44.1%", "good"],
  ["G", "55–100%", 585, "70.9%", "71.5%", "good"],
];

function Model() {
  return html`<div>
    <div style=${{ float: "right" }}><${LinkBtn} primary href=${STUDIO.model(3)}>Open registered model in Azure ML</${LinkBtn}></div>
    <${Head} title="Model & validation"
      lead="Held-out test set (5,993 clients). The question is not only 'does it rank' but 'are the PDs right per grade, is it fair, can we explain each decision'." />
    <div class="grid g6">
      <${Kpi} label="ROC AUC" value="0.779" note="CI 0.765–0.792" />
      <${Kpi} label="Gini" value="0.557" />
      <${Kpi} label="KS" value="0.423" />
      <${Kpi} label="ECE" value="0.0105" note="slope 0.94" />
      <${Kpi} label="Mean PD" value="22.0%" note="observed 22.1%" />
      <${Kpi} label="Top 2 deciles" value="50%" note="of all defaults" />
    </div>
    <div class="grid g2 section">
      <div class="card"><${Gallery} items=${[
        ["calibration", "Calibration", "Isotonic-calibrated PDs track observed default rates; ECE 0.0105, Hosmer-Lemeshow p 0.36."],
        ["backtest", "Backtest", "Jeffreys test per rating grade: 6 green, 1 amber (grade B), 0 red."],
        ["ks_chart", "KS", "Separation of defaulters vs non-defaulters: KS 0.423."],
        ["deciles", "Deciles", "Risk deciles: the top two capture 50% of all defaults."],
        ["cost_curve", "Cut-off", "Expected cost vs cut-off; the optimum PD ≥ 0.167 is 49% cheaper than acting on nobody."],
        ["shap_beeswarm", "SHAP", "Main drivers: months since last delay, latest repayment status, longest delay."],
        ["waterfall_high_risk", "Reason codes", "Per-client SHAP waterfall, the basis for the reason codes returned by batch, endpoint and app."],
        ["fairness", "Fairness", "Outcome rates by sex, marital status and age band; none are model inputs."],
      ]} /></div>
      <div class="card"><h3>PD backtest per grade (Jeffreys test)</h3>
        <table><thead><tr><th>Grade</th><th>PD range</th><th class="num">Clients</th><th class="num">Observed</th><th class="num">Mean PD</th><th></th></tr></thead>
        <tbody>${GRADES.map(([g, r, n, o, p, k]) => html`<tr key=${g}><td><b>${g}</b></td><td>${r}</td><td class="num">${n.toLocaleString("en-US")}</td>
          <td class="num">${o}</td><td class="num">${p}</td><td><${Pill} kind=${k}>${k === "good" ? "green" : "amber"}</${Pill}></td></tr>`)}</tbody></table>
        <div class="callout section small">Grade B is amber (observed 7.5% vs 5.8% predicted, p = 0.007). Runbook: document, watch the next cycle; red or two consecutive ambers trigger recalibration.</div>
        <h3 class="section">Decision layer</h3>
        <ul class="clean small">
          <li>Cut-off from costs: missing a defaulter costs 5× a needless intervention → PD ≥ 0.167</li>
          <li>At that cut-off: recall 71%, precision 39%, 40% of clients flagged</li>
          <li>Not an automated decline engine: decisions stay with credit management</li>
        </ul>
      </div>
    </div>
  </div>`;
}

const SAMPLE_RESPONSE = `{
  "model_version": "credit-default-model:2",
  "threshold": 0.1667,
  "latency_ms": 86.4,
  "predictions": [
    {
      "pd": 0.83333,
      "grade": "G",
      "action": "Intervene",
      "flag": 1,
      "ead": 41449.5,
      "el": 20724.75,
      "ifrs9_stage": 2,
      "ecl": 24754.5625,
      "ews_triggers": "Paid less than 10% of the previous bill; PD in grade F/G",
      "limit_action": "Reduce limit",
      "recommended_limit": 37000.0,
      "ID": 7926,
      "reason_codes": [
        {
          "code": "RC_PAY_1",
          "text": "Payment 2 months late in Sep"
        },
        {
          "code": "RC_MONTHS_SINCE_DELAY",
          "text": "Most recent late payment 1 month ago"
        },
        {
          "code": "RC_MAX_DELAY",
          "text": "Longest payment delay: 2 months"
        },
        {
          "code": "RC_N_DELAYS_RECENT3",
          "text": "3 late payments in the last 3 months"
        }
      ]
    }
  ]
}`;

function Serving() {
  return html`<div>
    <${Head} title="Serving: batch first, real-time when needed"
      lead="Credit portfolios are scored in batch (monthly/weekly); the online endpoint serves origination or limit requests. Both use the same registered model and the same feature code." />
    <div class="grid g2">
      <div class="card"><h3><img src=${I("az/batch")} alt="" style=${{ width: "22px", verticalAlign: "-5px", marginRight: "8px" }} />Batch scoring & monitoring pipeline</h3>
        <div class="flow" style=${{ margin: "8px 0 14px" }}>
          <div class="stage"><h3><span class="n">a</span>score</h3><div class="small muted">PD, grade, EL, stage, EWS, limit advice, reason codes</div></div>
          <div class="stage"><h3><span class="n">b</span>monitor</h3><div class="small muted">PSI vs the model's own baseline → alert.json</div></div>
          <div class="stage"><h3><span class="n">c</span>insights</h3><div class="small muted">DuckDB SQL → Power BI tables</div></div>
        </div>
        <ul class="clean small">
          <li><code>python pipeline/run_batch_pipeline.py --batch data/batches/portfolio_downturn.csv --wait</code></li>
          <li>Each batch becomes a versioned data asset; <code>--model-version</code> pins a model for rollback</li>
          <li>Weekly schedule: <code>--schedule "0 7 * * 1"</code> (Jobs → Schedules)</li>
          <li>Labelled batches add a realised AUC / Jeffreys backtest</li>
        </ul>
      </div>
      <div class="card"><h3><img src=${I("az/machine-learning")} alt="" style=${{ width: "22px", verticalAlign: "-5px", marginRight: "8px" }} />Managed online endpoint</h3>
        <ul class="clean small">
          <li><b>Blue/green:</b> new deployment, smoke test, then switch (or <code>--canary 10</code>)</li>
          <li><b>Rollback:</b> <code>python deploy/deploy_endpoint.py --rollback</code></li>
          <li><b>Data collector</b> on inputs/outputs feeds the Azure ML model monitor</li>
          <li><b>Validation:</b> missing fields → 400 with the list of required fields</li>
          <li><b>Key auth</b>, telemetry to Application Insights, OpenTelemetry span per request</li>
        </ul>
        <pre class="json section">${SAMPLE_RESPONSE}</pre>
        <div class="small muted" style=${{ marginTop: "6px" }}>Live response from <code>credit-risk-fb1504</code> (model v2, client 7926 of deploy/sample_request.json). Smoke test: 3 clients → A / Standard, D / Watch, G / Intervene.</div>
      </div>
    </div>
  </div>`;
}

function Observability() {
  return html`<div>
    <${Head} title="Observability & monitoring"
      lead="Four layers: data quality at every step, model drift per batch, service health on the endpoint, and alerts that point to a runbook playbook." />
    <div class="grid g4">
      <${Kpi} accent label="Score PSI · downturn batch" value="0.344" note="alert ≥ 0.25 · Azure run, 29 Oct 2026" />
      <${Kpi} accent label="Rating-grade PSI" value="0.331" note="vs the baseline stored with the model" />
      <${Kpi} accent label="Features in alert / warn" value="13 / 9" note="max feature PSI 1.70 (pay_ratio_avg)" />
      <${Kpi} accent label="Mean PD" value="22.1% → 29.7%" note="reference vs current batch" />
    </div>
    <div class="grid g2 section">
      <div class="card"><figure><img src=${IMG("psi_downturn")} alt="PSI per feature" />
        <figcaption>PSI per feature for the simulated downturn batch. The monitor step ends with [DRIFT ALERT] and writes alert.json → runbook playbook 3.</figcaption></figure></div>
      <div class="card"><h3>Signals and thresholds</h3>
        <table><thead><tr><th>Signal</th><th>Warn</th><th>Alert</th></tr></thead><tbody>
          <tr><td>Data-quality hard check</td><td>–</td><td>any violation fails the step</td></tr>
          <tr><td>Score / feature / grade PSI</td><td>≥ 0.10</td><td>≥ 0.25</td></tr>
          <tr><td>Backtest per grade (Jeffreys)</td><td>${"p < 5%"}</td><td>${"p < 0.1%"}</td></tr>
          <tr><td>Realised vs validation AUC</td><td>−0.02</td><td>−0.05</td></tr>
          <tr><td>Calibration error (ECE)</td><td>${"> 0.03"}</td><td>${"> 0.05"}</td></tr>
          <tr><td>Endpoint p95 latency</td><td>${"> 500 ms"}</td><td>${"> 1 s"}</td></tr>
          <tr><td>Endpoint 4xx/5xx rate</td><td>${"> 1%"}</td><td>${"> 5%"}</td></tr>
          <tr><td>Failed Azure ML job</td><td>–</td><td>any</td></tr>
        </tbody></table></div>
    </div>
    <div class="grid g4 section">
      <div class="card"><${Svc} icon="az/app-insights" name="Application Insights">requests, latency, failures from endpoint and app; OpenTelemetry spans (endpoint.score, agent tool calls)</${Svc}></div>
      <div class="card"><${Svc} icon="az/monitor" name="Azure ML model monitor">daily drift + data-quality signals on the endpoint's collected data, e-mail alert</${Svc}></div>
      <div class="card"><${Svc} icon="az/alerts" name="Azure Monitor alerts">aml-failed-runs and endpoint-latency-p95 (infra/alerts.sh)</${Svc}></div>
      <div class="card"><${Svc} icon="mlflow" name="MLflow in the workspace">every metric, plot and model version, with lineage back to code and data</${Svc}></div>
    </div>
    <div class="card section"><h3>Runbook playbooks (docs/RUNBOOK.md)</h3>
      <div class="grid g3 small">
        <div><b>1 · Job failed</b><br /><span class="muted">step logs → reproduce locally → fix + test; no model promoted, nothing to roll back</span></div>
        <div><b>2 · DQ gate failed</b><br /><span class="muted">quality_report → data owner; never relax the hard schema to pass</span></div>
        <div><b>3 · Drift alert</b><br /><span class="muted">real shift vs data issue → inform credit mgmt, retrain, gate decides; tighten cut-off meanwhile</span></div>
        <div><b>4 · Backtest breach</b><br /><span class="muted">amber: watch · red / 2× amber: recalibrate, review master scale</span></div>
        <div><b>5 · Endpoint incident</b><br /><span class="muted">App Insights → scale out or --rollback; batch keeps working</span></div>
        <div><b>6 · Bad model in prod</b><br /><span class="muted">rollback; root cause through git SHA + data version tags</span></div>
      </div></div>
  </div>`;
}

function Reporting() {
  return html`<div>
    <${Head} title="Reporting & business impact"
      lead="The PD is an input, not the product. The insights step turns scores into expected loss, staging, early warning and limit actions in SQL, ready for Power BI. Figures: simulated downturn batch, scored on Azure on 29 Oct 2026." />
    <div class="grid g4">
      <${Kpi} label="Exposure (EAD)" value="NT$ 703 M" note="5,993 clients" />
      <${Kpi} label="Expected loss" value="NT$ 110 M" note="15.7% of EAD (12-month)" />
      <${Kpi} label="IFRS 9 stage 2 / 3" value="40.9% / 3.3%" note="share of EAD · lifetime ECL NT$ 171 M" />
      <${Kpi} label="Limit steering" value="−NT$ 18.4 M EL" note="1,119 high-risk clients, EAD −NT$ 59.9 M" />
    </div>
    <div class="grid g2 section">
      <div class="card"><h3>What the business reads</h3><ul class="clean small">
        <li>Grade E carries 30% of expected loss with 31% of clients</li>
        <li>Riskiest segment (k-means): <b>delinquent revolvers, lower limit</b>, mean PD 49%, 24% of clients</li>
        <li>Delinquency 44.9% today → 38.8% projected in 3 months (Markov roll-rate projection)</li>
        <li>Actions: intervene 66.8% · watch 23.1% · 35.3% with multiple early-warning triggers</li>
      </ul>
      <h3 class="section">SQL layer (src/creditrisk/sql/, DuckDB)</h3>
      <div class="row small">${["kpis", "grade_summary", "segment_summary", "stage_summary", "watchlist", "limit_actions", "roll_rates", "delinquency_trend"]
        .map((q) => html`<code key=${q}>${q}.sql</code>`)}</div>
      <h3 class="section">Power BI-ready outputs</h3>
      <div class="row"><img src=${I("az/power-bi")} alt="" style=${{ width: "28px" }} /><span class="small muted">Flat tables per batch (grade, segment, stage, watchlist, limit actions, roll rates) land in the pipeline outputs, ready for a Power BI model. LGD, CCF and cost ratios are explicit, documented assumptions.</span></div>
      <div class="callout warn section small">UCI's target is next-month default, not 90 days past due: PDs are recalibrated before any IFRS 9 / IRB use. Staging here is illustrative.</div>
      </div>
      <div class="card"><figure><img src=${IMG("roll_rates")} alt="Roll-rate matrix" />
        <figcaption>Roll-rate matrix between delinquency buckets, the basis for the 3-month projection.</figcaption></figure></div>
    </div>
  </div>`;
}

function Cockpit() {
  return html`<div>
    <${Head} title="Credit risk cockpit"
      lead="Streamlit app on Azure Container Apps (scales to zero), driven by the registered model package or the endpoint. One screen per question a credit-risk team asks." />
    <div class="row" style=${{ marginBottom: "12px" }}>
      <a class="iconbtn on" href="http://localhost:8501" target="_blank" rel="noopener" style=${{ textDecoration: "none", padding: "6px 14px", fontSize: "13px" }}>Open live cockpit ↗</a>
      <span class="small muted">start it first with <code>make app</code> (runs on your laptop, registered model v2 in <code>app/model</code>)</span>
    </div>
    <div class="card"><${Gallery} items=${[
      ["app_portfolio", "Portfolio", "Portfolio view: exposure, expected loss, grade mix, segments."],
      ["app_client", "Client what-if", "Score a client, change inputs, see PD, grade and SHAP reason codes move."],
      ["app_policy", "Cut-off & limits", "Cut-off economics and limit steering: cost of errors vs intervention volume."],
      ["app_monitoring", "Model & monitoring", "Drift simulator and monitoring view against the model's baseline."],
      ["app_agent", "Ask the portfolio", "LLM agent: one SELECT-only, read-only DuckDB query per tool call; deny-list, row cap, tracing; offline fallback."],
    ]} /></div>
    <div class="grid g3 section">
      <div class="card"><${Svc} icon="az/container-apps" name="Azure Container Apps">root Dockerfile, scripts/deploy_app.sh; MIN_REPLICAS=1 on demo day</${Svc}></div>
      <div class="card"><${Svc} icon="az/azure-openai" name="Azure OpenAI agent">tool-calling, guarded text-to-SQL; answers cite the query it ran</${Svc}></div>
      <div class="card"><${Svc} icon="streamlit" name="Streamlit">6 tabs: portfolio, client scoring, cut-off & limits, model & monitoring, agent, model card</${Svc}></div>
    </div>
  </div>`;
}

function Governance() {
  return html`<div>
    <${Head} title="CI/CD & model governance"
      lead="Registration is not deployment. A model reaches production only through a passing gate, a human approval and a smoke test, and every version can be traced back to code and data." />
    <div class="grid g2">
      <div class="card"><h3><img src=${I("githubactions")} alt="" style=${{ width: "20px", verticalAlign: "-4px", marginRight: "8px" }} />GitHub Actions workflows</h3>
        <table><thead><tr><th>Workflow</th><th>Trigger</th><th>Does</th></tr></thead><tbody>
          <tr><td><code>ci</code></td><td>every push / PR</td><td>ruff, 53 tests, dry-run of every Azure spec</td></tr>
          <tr><td><code>train-and-deploy</code></td><td>push to main (model code), manual</td><td>CI → Azure ML pipeline → model card in job summary → optional blue/green deploy</td></tr>
          <tr><td><code>batch-scoring-and-monitoring</code></td><td>manual</td><td>scores a demo batch, runs the drift monitor</td></tr>
          <tr><td><code>deploy-app</code></td><td>manual</td><td>Streamlit app to Container Apps</td></tr>
        </tbody></table>
        <div class="row section">
          <img src=${I("az/app-registration")} alt="" style=${{ width: "30px" }} />
          <span class="small"><b>OIDC, no secrets:</b> federated credentials for main, PRs and the <code>production</code> environment (required reviewer = approval gate). Contributor scoped to the resource group.</span>
        </div>
      </div>
      <div class="card"><h3>Promotion policy</h3><ul class="clean small">
        <li>Challenger registered only if it passes min AUC + calibration limit and is within 0.005 AUC of the champion on the same test data</li>
        <li>Deployment runs behind GitHub <code>production</code> (human approval) + smoke test, blue/green or canary</li>
        <li>Each version carries its model card, evaluation evidence, drift baseline, git SHA and data version</li>
        <li>Quarterly model review: model card + monitoring history, signed off by credit risk and the model owner</li>
      </ul>
      <h3 class="section">Testing & reproducibility</h3><ul class="clean small">
        <li>53 tests: units + end to end on synthetic data (~15 s); same suite in CI (<code>make check</code>)</li>
        <li>uv-locked dependencies; Azure environments generated from the same lock (<code>make lock</code>)</li>
        <li><code>make local</code> = the Azure pipeline, locally with an MLflow SQLite registry</li>
        <li><b>Spec-driven:</b> <code>docs/SPEC.md</code> lists every requirement with its acceptance test; the traceability table maps requirement → code → test</li>
      </ul></div>
    </div>
    <div class="card section"><h3>EU AI Act: what applies, and when</h3>
      <div class="grid g2 small">
        <div><table><tbody>
          <tr><td><b>1 Aug 2024</b></td><td>Regulation (EU) 2024/1689 in force</td></tr>
          <tr><td><b>2 Feb 2025</b></td><td>Prohibited practices; AI literacy for staff</td></tr>
          <tr><td><b>2 Aug 2025</b></td><td>General-purpose AI model obligations</td></tr>
          <tr><td><b>2 Aug 2026</b></td><td>Transparency (Art. 50): users must know they interact with AI · applies to the portfolio agent</td></tr>
          <tr class="hl"><td><b>2 Dec 2027</b></td><td><b>High-risk, Annex III</b> (incl. credit scoring of natural persons): moved from 2 Aug 2026 by the 2026 Digital Omnibus</td></tr>
          <tr><td><b>2 Aug 2028</b></td><td>High-risk AI embedded in regulated products (Annex I)</td></tr>
        </tbody></table></div>
        <div><ul class="clean">
          <li><b>Scope for DKV:</b> Annex III 5(b) covers creditworthiness of <i>natural persons</i>. Fleet customers are mostly companies, so most of the portfolio is out of scope; sole traders and private customers are in.</li>
          <li><b>My stance:</b> apply the high-risk requirements to the whole portfolio anyway; the deadline moved, the expectations did not.</li>
          <li><b>Already covered here:</b> risk management (gate, runbook, monitoring), data governance (quality gate, lineage), documentation and logs (model card, versions), transparency (reason codes), human oversight (no automated decline; escalation owners), accuracy and robustness (backtest, drift).</li>
          <li class="muted">Dates from the adopted Omnibus as published in 2026; confirm against the Official Journal text.</li>
        </ul></div>
      </div></div>
  </div>`;
}

const RUNS = [
  ["Training pipeline", "cool_rhythm_hx51hfjj1r", "Completed", "preprocess (incl. one-time image build) 16 min · train 3 min · evaluate 1.5 min · register 1 min"],
  ["Batch monitoring (downturn)", "ivory_rabbit_9x86flykq4", "Completed", "score 1 min · monitor + insights in parallel 1.5 min · [DRIFT ALERT]"],
  ["Training via GitHub Actions (OIDC)", "brave_ocean_w3b8k8s0dq", "Completed", "triggered by train-and-deploy · gate compared v2 with champion v1 · registered v2"],
  ["Training after the engineering clean-up", "eager_cord_w2yj7rnz4n", "Completed", "uv-locked environment · cluster on D2ds_v5 · gate compared v3 with v2 · registered v3 (git 600cc5b)"],
];

function Live() {
  return html`<div>
    <${Head} title="Live on Azure"
      lead="What I have running in my Azure subscription, the evidence for the deliverable, and what it costs." />
    <div class="row" style=${{ marginBottom: "14px" }}>
      <${LinkBtn} primary href=${STUDIO.model(3)}>Registered model (latest, v3)</${LinkBtn}>
      <${LinkBtn} href=${STUDIO.model(2)}>Model v2 (on the endpoint)</${LinkBtn}>
      <${LinkBtn} href=${STUDIO.endpoint}>Online endpoint</${LinkBtn}>
      <${LinkBtn} href=${studio("eager_cord_w2yj7rnz4n")}>Latest training run</${LinkBtn}>
      <${LinkBtn} href=${STUDIO.data}>Data asset</${LinkBtn}>
      <${LinkBtn} href="https://github.com/akhileshwarsharma95/credit-default-azureml/actions">GitHub Actions</${LinkBtn}>
      <${LinkBtn} href="http://localhost:8501">Cockpit (needs make app)</${LinkBtn}>
    </div>
    <div class="grid g4">
      <div class="card"><${Svc} icon="az/machine-learning" name="mlw-credit-risk">Azure ML workspace · Germany West Central</${Svc}></div>
      <div class="card"><${Svc} icon="az/vm" name="cpu-cluster">Standard_D2ds_v5 · 0–2 nodes · idle = 0 nodes</${Svc}></div>
      <div class="card"><${Svc} icon="az/machine-learning" name="credit-default-model v1–v3">test AUC 0.7785 · Gini 0.557 · ECE 0.0105 · v2, v3 from CI</${Svc}></div>
      <div class="card"><${Svc} icon="az/storage" name="Data assets">credit-default-raw:1 · credit-default-batch-portfolio-downturn:1</${Svc}></div>
    </div>
    <div class="card section"><${Svc} icon="az/machine-learning" name="Online endpoint credit-risk-fb1504">
      deployment <b>blue</b> = credit-default-model:2 · 100% traffic · Standard_D2as_v4 × 1 · smoke test passed · ~86 ms per request ·
      <code>az ml online-endpoint invoke -n credit-risk-fb1504 -r deploy/sample_request.json -g rg-credit-risk -w mlw-credit-risk</code></${Svc}></div>
    <div class="card section"><h3>Pipeline runs</h3>
      <table><thead><tr><th>Pipeline</th><th>Run</th><th>Status</th><th>Steps</th></tr></thead><tbody>
        ${RUNS.map(([p, r, s, d]) => html`<tr key=${r}><td>${p}</td><td><a href=${studio(r)} target="_blank" rel="noopener"><code>${r}</code></a></td>
          <td><${Pill} kind="good">${s}</${Pill}></td><td class="small">${d}</td></tr>`)}
      </tbody></table></div>
    <div class="grid g2 section">
      <div class="card"><h3>What I hit moving to Azure, and how I fixed it</h3><ul class="clean small">
        <li><b>Quota:</b> DSv2 family had 0 vCPUs in the region → cluster on D2ds_v5 (DDSv5), endpoint on D2as_v4 (DASv4)</li>
        <li><b>Shared quota:</b> with both in one 4-vCPU family the endpoint's reserve left the cluster no node, so a training run queued for an hour → separate families</li>
        <li><b>Config bug:</b> .env was loaded after defaults were read, so the VM override was ignored → load first</li>
        <li><b>Output download:</b> pipeline-level outputs have no path of their own → fetch from the register step</li>
        <li><b>OIDC:</b> new GitHub repos send immutable-ID subject claims → register both subject formats</li>
        <li><b>Endpoint:</b> a fresh subscription lacks the Compute, Network, PolicyInsights and Cdn providers → registered in setup</li>
      </ul></div>
      <div class="card"><h3><img src=${I("az/cost")} alt="" style=${{ width: "20px", verticalAlign: "-4px", marginRight: "8px" }} />Cost</h3>
        <table><tbody>
          <tr><td>Compute cluster (scales to 0)</td><td>cents per pipeline run</td></tr>
          <tr><td>Container registry (Basic)</td><td>≈ $5 / month</td></tr>
          <tr><td>Online endpoint</td><td>billed per hour while it exists, delete after use</td></tr>
          <tr><td>Storage, Key Vault, App Insights</td><td>negligible at this volume</td></tr>
        </tbody></table>
        <p class="small muted" style=${{ marginBottom: 0 }}>Clean-up: <code>python deploy/deploy_endpoint.py --delete</code> · <code>az group delete -n rg-credit-risk</code></p>
      </div>
    </div>
  </div>`;
}

const MAPPING = [
  ["LIMIT_BAL · credit limit", "Credit line on the fuel / toll / charging card account", "limit steering is a core lever"],
  ["BILL_AMT · monthly bill", "Invoiced transaction volume (fuel, toll, e-charging, services)", "utilisation = invoiced / limit"],
  ["PAY_AMT · payments", "Payments received, direct-debit returns", "repayment ratio, returned debits"],
  ["PAY_x · repayment status", "Days past due per invoice, dunning level", "delinquency patterns, roll rates"],
  ["Default next month", "B2B payment default / write-off (e.g. 90+ DPD)", "define with credit management first"],
];

function Dkv() {
  return html`<div>
    <div class="prepared"><img src=${I("dkv-mobility")} alt="DKV Mobility" /><span>Transfer to <b>DKV Mobility</b> · how I would apply this engine</span></div>
    <${Head} title="Why this matters for DKV Mobility"
      lead="Fleet customers pay for fuel, tolls and charging on credit and settle by invoice: the same problem as the case study (a credit line, monthly spend, repayments, delinquency), on a B2B portfolio. The platform carries over unchanged; the data, target and risk specifics change." />
    <div class="grid g2">
      <div class="card"><h3>From the case study to a fleet-card portfolio</h3>
        <table><thead><tr><th>Case study (UCI)</th><th>Fleet-card equivalent</th><th>Feature idea</th></tr></thead>
        <tbody>${MAPPING.map(([a, b, c]) => html`<tr key=${a}><td><code>${a.split(" · ")[0]}</code><div class="small muted">${a.split(" · ")[1]}</div></td><td>${b}</td><td class="small muted">${c}</td></tr>`)}</tbody></table>
        <h3 class="section">Signals a fleet portfolio adds</h3>
        <ul class="clean small">
          <li><b>Transaction behaviour:</b> spend velocity and spikes, share of toll vs fuel vs charging, countries, unusual stations</li>
          <li><b>Company data:</b> industry, fleet size, years in business, group structure, external credit bureau scores</li>
          <li><b>Payment behaviour:</b> payment terms, returned direct debits, dunning history, disputes</li>
          <li><b>Macro:</b> freight demand and diesel price shifts, which the drift monitor watches per segment</li>
        </ul>
      </div>
      <div class="card"><h3>Use cases, same engine</h3>
        <div class="grid g2 small">
          <div><${Pill} kind="accent">Onboarding</${Pill}><p>Application score + cost-based cut-off sets the initial credit line; real-time endpoint answers in milliseconds.</p></div>
          <div><${Pill} kind="accent">Limit steering</${Pill}><p>Behavioural PD drives limit increases for good payers and reductions for risk, measured as EL saved.</p></div>
          <div><${Pill} kind="accent">Early warning</${Pill}><p>Watchlist with reason codes routes customers to collections before the first missed payment.</p></div>
          <div><${Pill} kind="accent">Provisioning</${Pill}><p>Calibrated PDs feed expected credit loss on receivables; backtest per grade gives finance the evidence.</p></div>
        </div>
        <h3 class="section">What changes for B2B</h3>
        <ul class="clean small">
          <li><b>Fewer, larger exposures:</b> concentration and group-level limits; low-default techniques where defaults are rare</li>
          <li><b>Target definition</b> agreed with credit management and finance before any modelling</li>
          <li><b>Explainability</b> matters more: credit officers talk to each customer, reason codes become the conversation</li>
          <li><b>Fairness</b> becomes a check on industry/region bias rather than personal attributes</li>
        </ul>
      </div>
    </div>
    <div class="card section"><h3>First 90 days</h3>
      <div class="flow">
        <div class="stage"><h3><span class="n">30</span>Understand</h3><ul class="clean small">
          <li>Meet credit management, collections, finance</li><li>Agree the default definition and the decisions to support</li><li>Map data sources: transactions, invoices, payments, bureau</li></ul></div>
        <div class="stage"><h3><span class="n">60</span>Baseline</h3><ul class="clean small">
          <li>Run this pipeline on real history, out-of-time validation</li><li>Compare with the current rules / scorecard</li><li>Quantify EL and limit-steering impact</li></ul></div>
        <div class="stage"><h3><span class="n">90</span>Operate</h3><ul class="clean small">
          <li>Shadow mode in batch, weekly drift monitoring</li><li>Model card + validation pack for sign-off</li><li>Plan the real-time onboarding endpoint and one React front end</li></ul></div>
      </div>
    </div>
  </div>`;
}

/* ------------------------------------------------------------------ project structure */
// [folder, purpose, [[file, what, why, connects], ...]]
const STRUCTURE = [
  ["src/creditrisk/", "The library: all logic, tested once, used everywhere", [
    ["config.yaml", "Business and technical parameters", "One place for costs, LGD/CCF, PD floor, master scale, gate, thresholds, Azure settings", "read by every step, the app and the submitters; ships inside the package"],
    ["quality.py", "pandera hard and soft checks, cleaning, quality report", "Bad data must stop the run; undocumented codes are cleaned and reported", "preprocess, score_batch, make_batches"],
    ["features.py", "13 engineered features, monotonic directions", "Behaviour summarised into risk signals inside the model pipeline (no skew)", "modeling (part of the saved model)"],
    ["modeling.py", "LightGBM and scorecard builders, Optuna, OOF, isotonic calibration, PD floor", "Model choice by evidence, calibrated probabilities", "train, evaluate"],
    ["metrics.py", "AUC/Gini/KS + CIs, calibration, Jeffreys backtest, fairness, cost curve", "Validation as for a credit model", "train, evaluate, monitor_drift"],
    ["explain.py", "TreeSHAP contributions and reason codes", "Every PD comes with its reasons", "evaluate, scoring"],
    ["policy.py", "Master scale, cut-off, EAD, IFRS 9 stage, EWS, limits, actions, TTC shift, pricing", "Turns a PD into decisions", "scoring, insights, analytics"],
    ["scoring.py", "score_frame: PD → grade → action, reasons, EAD, EL, stage, EWS, limit advice", "One scoring path for batch, endpoint and app", "score_batch, serving/score.py, app, evaluate"],
    ["drift.py", "Reference profile, PSI, downturn simulation", "Monitoring against the baseline stored with each model", "train, monitor_drift, make_batches, app"],
    ["migration.py", "Markov lifetime PD (cohort method)", "IFRS 9 lifetime PD vectors per delinquency bucket", "insights"],
    ["portfolio_risk.py", "Vasicek loss distribution, credit VaR, concentration", "Total risk: expected vs unexpected loss", "insights, pricing"],
    ["insights.py + sql/", "Segments, SQL tables, roll rates, default forecast, KPIs", "Power BI-ready portfolio view", "portfolio_insights step, app, analytics"],
    ["agent.py", "LLM agent with a SELECT-only SQL tool", "Ask the portfolio in plain language, safely", "app"],
    ["feature_report.py · model_card.py · plots.py · tracking.py · observability.py · schema.py · config.py", "Feature analysis, model card, charts, MLflow helpers, logging/telemetry, column definitions, config loader", "Shared building blocks", "steps, scripts, app"],
  ]],
  ["src/", "Pipeline steps: one script = one Azure ML component", [
    ["preprocess.py", "Step 1: validate, clean, split", "Clean data before anything is fitted", "components/preprocess.yml → train"],
    ["train.py", "Step 2: tune, select, calibrate, package", "One self-contained MLflow model", "components/train.yml → evaluate, register"],
    ["evaluate.py", "Step 3: metrics, backtest, SHAP, gate decision", "Evidence + champion comparison on the same test set", "components/evaluate.yml → register"],
    ["register.py", "Step 4: register if the gate passed, with model card", "Only better models reach the registry", "Azure ML model registry"],
    ["score_batch.py · monitor_drift.py · portfolio_insights.py", "Batch pipeline: score → PSI → insights", "Weekly portfolio view and drift alert", "components/*.yml, pipeline/batch_pipeline.yml"],
    ["serving/score.py", "Online endpoint: init() + run()", "Real-time PD with reason codes; 400 on bad input", "deploy/deploy_endpoint.py"],
  ]],
  ["components/", "7 YAML step definitions: inputs, outputs, environment, command", [
    ["*.yml", "preprocess, train, evaluate, register, score_batch, monitor_drift, portfolio_insights", "Single source of truth for SDK and CLI pipelines", "pipeline/run_*.py and pipeline/*.yml; defaults locked to config by tests"],
  ]],
  ["pipeline/", "Orchestration on Azure ML", [
    ["run_pipeline.py", "Builds and submits the training pipeline (SDK v2); --dry-run, --schedule", "The brief's pipeline, plus cluster and data-asset creation", "components/, azure_utils.py"],
    ["run_batch_pipeline.py", "Registers a batch and submits score → monitor → insights", "Batch scoring and drift demo", "components/, azure_utils.py"],
    ["azure_utils.py", "Client, cluster, data assets (sha256), model lookup, log streaming", "Shared Azure plumbing", "submitters, deploy, scripts"],
    ["training_pipeline.yml · batch_pipeline.yml", "The same pipelines for az ml job create", "The brief's YAML + CLI option", "components/"],
  ]],
  ["deploy/ · monitoring/ · infra/", "Serving, model monitor, Azure setup", [
    ["deploy/deploy_endpoint.py", "Blue/green release, smoke test, canary, rollback, delete", "Safe real-time releases", "serving/score.py, model registry"],
    ["monitoring/setup_model_monitor.py", "Azure ML model monitor on the endpoint", "Drift on what the endpoint receives", "online endpoint data collector"],
    ["infra/setup_azure.sh · alerts.sh", "Providers, resource group, workspace, OIDC, GitHub secrets · alerts", "Reproducible setup, no stored secrets", "GitHub Actions, Azure Monitor"],
  ]],
  ["scripts/ · app/", "Local tools and the cockpit", [
    ["scripts/run_local.py", "Every step locally, SQLite MLflow registry", "Same pipeline without Azure (make local)", "src/*.py"],
    ["scripts/risk_analytics.py", "Feature report + portfolio analytics → docs/analytics", "The numbers on this deck (make analytics)", "feature_report, insights, policy"],
    ["scripts/download_data.py · make_batches.py · fetch_model.py · lock_env.py", "UCI data, demo batches, registered model download, Azure environment locks", "One-command chores", "data/, app/model, environments/"],
    ["app/streamlit_app.py", "Cockpit: portfolio, client profile + what-if, limits, monitoring, agent, model card", "The credit officer's tool", "scoring, insights, agent, app/model"],
  ]],
  ["tests/ · .github/ · environments/ · root", "Quality, CI/CD, reproducibility", [
    ["tests/", "53 tests: units, end to end on synthetic data, spec dry-runs, config consistency", "Every requirement has a test", "make check, ci workflow"],
    [".github/workflows/", "ci, train-and-deploy, batch-monitoring, deploy-app", "Checks on every push; training and deployment from GitHub via OIDC", "Azure ML, Entra ID"],
    ["environments/", "Pinned conda environments for jobs and the endpoint", "Azure runs exactly the tested versions", "components/, deploy_endpoint.py, make lock"],
    ["pyproject.toml · uv.lock · Makefile · Dockerfile", "Dependencies, lock, commands, app container", "Reproducible installs and one command per task", "uv, ruff, pytest, make"],
    ["docs/SPEC.md", "The specification: requirements → code → test", "Spec-driven development", "everything above"],
  ]],
];

function ProjectStructure() {
  const [open, setOpen] = useState(0);
  const [file, setFile] = useState([0, 0]);
  const [fi, fj] = file;
  const [, , files] = STRUCTURE[fi];
  const [name, what, why, connects] = files[fj];
  return html`<div>
    <${Head} title="Project structure"
      lead="One library, many entry points: the same creditrisk code runs in every pipeline step, the endpoint, the cockpit and the tests. Click a folder, then a file." />
    <div class="arch-layout">
      <div class="card tree">${STRUCTURE.map(([folder, purpose, items], i) => html`<div key=${folder} class="tree-folder">
        <button class=${"tree-head" + (open === i ? " on" : "")} onClick=${() => { setOpen(open === i ? -1 : i); setFile([i, 0]); }}>
          <span class="caret">${open === i ? "▾" : "▸"}</span><code>${folder}</code><span class="muted small">${purpose}</span></button>
        ${open === i && html`<div class="tree-files">${items.map(([n, w], j) => html`<button key=${n}
          class=${"tree-file" + (fi === i && fj === j ? " on" : "")} onClick=${() => setFile([i, j])}><code>${n}</code><span class="muted small">${w}</span></button>`)}</div>`}
      </div>`)}</div>
      <aside class="card side-panel">
        <div class="muted small">${STRUCTURE[fi][0]}</div>
        <h3 style=${{ marginTop: "4px" }}><code>${name}</code></h3>
        <div class="side-label">What it does</div><p class="small" style=${{ margin: 0 }}>${what}</p>
        <div class="side-label">Why it exists</div><p class="small" style=${{ margin: 0 }}>${why}</p>
        <div class="side-label">Connects to</div><p class="small" style=${{ margin: 0 }}>${connects}</p>
      </aside>
    </div>
    <div class="card section"><h3>One prediction, through the code</h3>
      <div class="flow">
        <div class="stage"><h3><span class="n">1</span>Data</h3><div class="small muted">schema.load_raw → quality.validate / clean → stratified split</div></div>
        <div class="stage"><h3><span class="n">2</span>Model</h3><div class="small muted">FeatureEngineer + LightGBM → isotonic → PD floor → MLflow package</div></div>
        <div class="stage"><h3><span class="n">3</span>Evidence</h3><div class="small muted">metrics, backtest, SHAP → gate → registry + model card</div></div>
        <div class="stage"><h3><span class="n">4</span>Decision</h3><div class="small muted">scoring.score_frame → grade, EL, stage, EWS, action, limit, reason codes</div></div>
      </div></div>
  </div>`;
}

/* ------------------------------------------------------------------ analytics tabs (data from scripts/risk_analytics.py) */
const A = window.ANALYTICS;
const fmtPct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const fmtM = (v) => `NT$ ${(v / 1e6).toFixed(1)} M`;

function Bars({ rows, max, fmt, color }) {
  const top = max || Math.max(...rows.map((r) => r.value));
  return html`<div class="bars">${rows.map((r) => html`<div class="bar-row" key=${r.label}>
    <div class="bar-label" title=${r.label}>${r.label}</div>
    <div class="bar-track"><div class="bar-fill" style=${{ width: `${(100 * r.value) / top}%`, background: color || "var(--accent)" }}></div></div>
    <div class="bar-value">${fmt(r.value)}${r.note && html` <span class="muted small">${r.note}</span>`}</div>
  </div>`)}</div>`;
}

function DataTab() {
  const feats = A.features;
  const binKeys = Object.keys(A.feature_bins);
  const [cur, setCur] = useState(binKeys[0]);
  const byName = Object.fromEntries(feats.map((f) => [f.feature, f]));
  const bins = A.feature_bins[cur];
  return html`<div>
    <${Head} title="The data, feature by feature"
      lead="30,000 clients, 23 raw fields and 13 engineered features. Before any model: what drives default, one feature at a time (information value and default rate per decile)." />
    <div class="grid g2">
      <div class="card"><h3>Information value ranking (top 14 of 36)</h3>
        <${Bars} rows=${feats.slice(0, 14).map((f) => ({ label: f.description, value: f.iv, note: f.strength }))} fmt=${(v) => v.toFixed(2)} />
        <p class="small muted" style=${{ marginBottom: 0 }}>IV bands: ${"< 0.02"} not predictive · 0.1 weak · 0.3 medium · 0.5 strong. Bill amounts sit at the bottom (IV ${"<"} 0.02).</p>
      </div>
      <div class="card"><h3>Default rate by bin</h3>
        <div class="thumbs"><${Seg} value=${cur} onChange=${setCur} options=${binKeys.map((k) => [k, k])} /></div>
        <div class="small muted" style=${{ marginBottom: "8px" }}>${byName[cur].description} · IV ${byName[cur].iv.toFixed(2)} · ${byName[cur].trend}</div>
        <${Bars} rows=${bins.map((b) => ({ label: b.x_min === b.x_max ? `${b.x_min}` : `${Number(b.x_min).toFixed(2)} – ${Number(b.x_max).toFixed(2)}`, value: b.default_rate, note: `${b.clients.toLocaleString("en-US")} clients` }))} max=${1} fmt=${(v) => fmtPct(v, 0)} color="var(--warn)" />
      </div>
    </div>
    <div class="card section"><h3>What the data told me</h3><div class="grid g3 small">
      <div><b>Behaviour beats balances.</b> Delinquency patterns reach IV 0.78–0.86; bill amounts are not predictive (IV ${"≤"} 0.02). Risk is in how clients pay, not how much they owe.</div>
      <div><b>Engineered features win.</b> My <code>n_delays</code> (0.86) outranks the raw latest status <code>PAY_1</code> (0.84); recency (<code>months_since_delay</code>) adds what a single month cannot.</div>
      <div><b>Undocumented codes matter.</b> PAY = −2 / −1 / 0 (no consumption, paid in full, revolving) are not in the data dictionary and are not ordered by risk, so raw PAY is non-monotonic. I keep them as they are and let the engineered delinquency counts carry the signal cleanly.</div>
      <div><b>Limits carry history.</b> Default rate falls from 35.9% to 12.5% across limit deciles: the limit embeds the lender's earlier assessment.</div>
      <div><b>Demographics carry almost nothing.</b> Sex (IV 0.009) and marital status (0.005) are excluded anyway and used only for the fairness check.</div>
      <div><b>A coding quirk.</b> A 1-month delay is almost never recorded before the latest month (7 transitions in the 5,993 test histories), so the 30-dpd migration rate cannot be estimated from this data.</div>
    </div></div>
  </div>`;
}

function Explainability() {
  const reasons = JSON.parse(SAMPLE_RESPONSE).predictions[0].reason_codes;
  return html`<div>
    <${Head} title="Explainability: every PD comes with its reasons"
      lead="Exact TreeSHAP contributions per client, turned into reason codes that credit officers, the endpoint, the batch output and the cockpit all share." />
    <div class="grid g2">
      <div class="card"><figure><img src=${IMG("waterfall_high_risk")} alt="SHAP waterfall" />
        <figcaption>One client: each bar is a feature's contribution in log-odds, from the portfolio average to this client's PD.</figcaption></figure></div>
      <div class="card"><figure><img src=${IMG("shap_beeswarm")} alt="SHAP beeswarm" />
        <figcaption>All clients: delinquency recency, latest status and longest delay drive the PD; colour = feature value.</figcaption></figure></div>
    </div>
    <div class="grid g3 section">
      <div class="card"><h3>From SHAP to reason codes</h3><ol class="small" style=${{ paddingLeft: "18px", margin: 0 }}>
        <li>TreeSHAP gives each feature's exact contribution for this client (no sampling).</li>
        <li>Keep the risk-raising contributions, ranked by size.</li>
        <li>Map each to a stable code and a plain sentence with the client's own values.</li>
        <li>Deliver the top 4 everywhere: batch file, endpoint response, watchlist, cockpit.</li></ol></div>
      <div class="card"><h3>Live example (endpoint, client 7926)</h3>
        <table><tbody>${reasons.map((r) => html`<tr key=${r.code}><td><code>${r.code}</code></td><td>${r.text}</td></tr>`)}</tbody></table>
        <p class="small muted" style=${{ marginBottom: 0 }}>PD 83% · grade G · action Intervene</p></div>
      <div class="card"><h3>Why it holds up</h3><ul class="clean small">
        <li><b>Monotonic constraints:</b> paying later can never lower the PD, so no reason code contradicts common sense</li>
        <li><b>Same code path</b> for training, batch, endpoint and app: the reason a client sees is the reason the model used</li>
        <li><b>For DKV:</b> a credit officer can explain a limit decision to a fleet customer in one sentence; EBA loan-origination guidelines expect exactly that</li></ul></div>
    </div>
  </div>`;
}

function PortfolioRisk() {
  const k = A.kpis;
  const lq = A.loss_quantiles;
  const lifetime = A.lifetime_pd.filter((r) => [1, 3, 6, 12].includes(r.month));
  const buckets = [...new Set(lifetime.map((r) => r.from_bucket))];
  const cell = (b, m) => lifetime.find((r) => r.from_bucket === b && r.month === m);
  return html`<div>
    <${Head} title="Portfolio risk: the total, not the single client"
      lead="Defaults cluster in a downturn, so the monthly portfolio loss matters more than any one PD. Test portfolio (5,993 clients) scored by model v2." />
    <div class="grid g4">
      <${Kpi} accent label="Expected loss" value=${fmtM(lq[0].loss)} note="sum of PD × LGD × EAD" />
      <${Kpi} accent label="Loss at 99.9%" value=${fmtM(lq[lq.length - 1].loss)} note=${`single-factor model, ρ = ${A.asset_correlation}`} />
      <${Kpi} accent label="Unexpected loss" value=${fmtM(k.unexpected_loss_999)} note="capital the portfolio needs above EL" />
      <${Kpi} accent label="Concentration" value=${fmtPct(k.top10_ead_share)} note=${`top-10 EAD share · largest segment ${fmtPct(k.largest_group_ead_share, 0)}`} />
    </div>
    <div class="grid g3 section">
      <div class="card"><h3>Loss distribution</h3>
        <${Bars} rows=${lq.map((r) => ({ label: r.confidence, value: r.loss }))} fmt=${fmtM} />
        <p class="small muted" style=${{ marginBottom: 0 }}>Retail cards use Basel's QRRE correlation (0.04); SME and corporate fleet clients would sit at 0.12–0.24, roughly tripling unexpected loss.</p></div>
      <div class="card"><h3>Lifetime PD from migration (cohort method)</h3>
        <table><thead><tr><th>From bucket</th>${[1, 3, 6, 12].map((m) => html`<th class="num" key=${m}>${m} m</th>`)}</tr></thead>
        <tbody>${buckets.map((b) => html`<tr key=${b}><td>${b}</td>${[1, 3, 6, 12].map((m) => html`<td class="num" key=${m}>${fmtPct(cell(b, m).cumulative_pd)}</td>`)}</tr>`)}
          <tr><td>1 30 dpd</td><td class="num muted" colspan="4">not estimable (7 transitions)</td></tr></tbody></table>
        <p class="small muted" style=${{ marginBottom: 0 }}>Cumulative PD = (Mᵗ)[default], marginal = Mᵗ − Mᵗ⁻¹ with 90+ dpd absorbing. Covers ${fmtPct(k.markov_coverage, 0)} of the portfolio.</p></div>
      <div class="card"><h3>Monthly default forecast</h3>
        <${Bars} rows=${A.default_forecast.map((r) => ({ label: `month ${r.month}`, value: r.expected_defaults }))} fmt=${(v) => `${v.toFixed(0)}`} color="var(--bad)" />
        <p class="small muted" style=${{ marginBottom: 0 }}>Expected new 90+ dpd defaults; 12-month loss ${fmtM(k.expected_loss_12m_forecast)}. Feeds the IFRS 9 engine as marginal PD vectors.</p></div>
    </div>
    <div class="grid g3 section">
      <div class="card"><h3>Through-the-cycle calibration</h3>
        <table><thead><tr><th>Grade</th><th class="num">PIT PD</th><th class="num">TTC PD</th><th class="num">Observed</th></tr></thead>
        <tbody>${A.ttc_by_grade.map((r) => html`<tr key=${r.grade}><td><b>${r.grade}</b></td><td class="num">${fmtPct(r.pd_pit)}</td><td class="num">${fmtPct(r.pd_ttc)}</td><td class="num">${fmtPct(r.observed_dr)}</td></tr>`)}</tbody></table>
        <p class="small muted" style=${{ marginBottom: 0 }}>The model sets the ranking; one log-odds shift sets the level to the long-run target (${fmtPct(A.ttc_target_pd, 0)}, illustrative). Slope unchanged.</p></div>
      <div class="card"><h3>Risk-based pricing per grade</h3>
        <table><thead><tr><th>Grade</th><th class="num">EL</th><th class="num">Capital</th><th class="num">Total price</th></tr></thead>
        <tbody>${A.pricing_by_grade.map((r) => html`<tr key=${r.grade}><td><b>${r.grade}</b></td><td class="num">${fmtPct(r.expected_loss_rate)}</td><td class="num">${fmtPct(r.capital_charge_rate)}</td><td class="num"><b>${fmtPct(r.total_rate)}</b></td></tr>`)}</tbody></table>
        <p class="small muted" style=${{ marginBottom: 0 }}>Funding + expected loss + capital × cost of capital + opex + margin, per unit of exposure. For DKV: card fees and payment terms by grade.</p></div>
      <div class="card"><h3>Escalation matrix</h3>
        <table><thead><tr><th>Level</th><th>Action</th><th>Owner</th></tr></thead>
        <tbody>${A.escalation.map((e) => html`<tr key=${e.level}><td><b>${e.level}</b><div class="small muted">${A.actions[e.level] ? `${A.actions[e.level].toLocaleString("en-US")} clients` : "stage 3"}</div></td><td class="small">${e.action}</td><td class="small">${e.owner}<div class="muted">${e.review}</div></td></tr>`)}</tbody></table></div>
    </div>
  </div>`;
}

function Roadmap() {
  const col = (title, kind, items) => html`<div class="card"><h3><${Pill} kind=${kind}>${title}</${Pill}></h3><ul class="clean small">${items.map((i) => html`<li key=${i}>${i}</li>`)}</ul></div>`;
  return html`<div>
    <${Head} title="Roadmap: in place, next, deliberately not"
      lead="What the engine does today, what I would add with DKV's data, and what I would not build, with the reason." />
    <div class="grid g3">
      ${col("In place", "good", [
        "Pipeline, gated registry, blue/green endpoint, batch monitoring",
        "Logging, App Insights, PSI drift, model monitor, alerts, runbook",
        "Feature analysis, SHAP reason codes, fairness check",
        "Portfolio loss distribution, Markov lifetime PDs, TTC shift, pricing",
        "Client profile, EWS, escalation matrix, limit steering",
        "Central config for business and technical parameters",
      ])}
      ${col("Next, with DKV data", "accent", [
        "Retraining loop: join outcomes to collected inputs once the window matures; out-of-time validation by vintage",
        "Compliance storage: immutable (WORM) blob, lifecycle tiers, 10-year retention, GDPR minimisation",
        "External data: bureau scores and insolvency notices (Creditreform, D&B, Creditsafe)",
        "Group view: parent-company exposure, group limits, contagion when one entity defaults",
        "Industry and macro overlay: default rates by NACE code, freight index, diesel price scenarios",
        "Scenario LGD: cure / collateral recovery / total loss; segment-level TTC scaling (SMB vs corporate)",
        "Lag and velocity features from daily transactions (fuel, toll, charging)",
        "One React front end on the endpoint + a small API",
      ])}
      ${col("Deliberately not", "warn", [
        "A separate model per risk grade: grades come from the PD, so it is circular",
        "Per-segment models on 30k rows: overfit; one model with segment calibration is the industry pattern",
        "Resampling the defaults: it breaks calibration, which pricing and provisioning depend on",
        "Changing the model the night before: new features go through the champion/challenger gate",
      ])}
    </div>
  </div>`;
}

const TABS = [
  ["overview", "Overview", Overview],
  ["business", "Business overview", BusinessOverview],
  ["architecture", "Architecture", Architecture],
  ["structure", "Project structure", ProjectStructure],
  ["data", "Data", DataTab],
  ["training", "Training pipeline", Training],
  ["model", "Model & validation", Model],
  ["explain", "Explainability", Explainability],
  ["serving", "Serving", Serving],
  ["observability", "Observability", Observability],
  ["reporting", "Reporting", Reporting],
  ["risk", "Portfolio risk", PortfolioRisk],
  ["cockpit", "Cockpit app", Cockpit],
  ["governance", "CI/CD & governance", Governance],
  ["live", "Live on Azure", Live],
  ["roadmap", "Roadmap", Roadmap],
  ["dkv", "For DKV Mobility", Dkv],
];

function App() {
  const initial = Math.max(0, TABS.findIndex((t) => "#" + t[0] === location.hash));
  const [idx, setIdx] = useState(initial);
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("pres-theme") || ""; } catch (e) { return ""; }
  });
  useEffect(() => {
    history.replaceState(null, "", "#" + TABS[idx][0]);
    window.scrollTo(0, 0);
  }, [idx]);
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem("pres-theme", theme); } catch (e) { /* storage unavailable */ }
  }, [theme]);
  useEffect(() => {
    const onHash = () => {
      const i = TABS.findIndex((t) => "#" + t[0] === location.hash);
      if (i >= 0) setIdx(i);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowRight" || e.key === "PageDown") setIdx((i) => Math.min(TABS.length - 1, i + 1));
      else if (e.key === "ArrowLeft" || e.key === "PageUp") setIdx((i) => Math.max(0, i - 1));
      else if (e.key === "Home") setIdx(0);
      else if (e.key === "End") setIdx(TABS.length - 1);
      else if (/^[0-9]$/.test(e.key)) setIdx(Math.min(TABS.length - 1, e.key === "0" ? 9 : +e.key - 1));
      else if (e.key === "f") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const Tab = TABS[idx][2];
  const dark = theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches);
  return html`<div class="app">
    <header>
      <div class="topbar">
        <img class="brand-logo" src=${I("dkv-mobility")} alt="DKV Mobility" />
        <div><h1>Credit default PD engine</h1><div class="sub">Case study for DKV Mobility · Senior Data Scientist · Azure ML SDK v2</div></div>
        <div class="spacer"></div>
        <button class="iconbtn" onClick=${() => setTheme(dark ? "light" : "dark")}>${dark ? "Light" : "Dark"}</button>
      </div>
      <nav class="tabs">${TABS.map(([k, label], i) => html`
        <button key=${k} class=${i === idx ? "on" : ""} onClick=${() => setIdx(i)}><span class="num">${i + 1}</span>${label}</button>`)}</nav>
    </header>
    <main><${Tab} dark=${dark} /></main>
    <footer><span class="keys"><kbd>←</kbd> <kbd>→</kbd> switch tabs · <kbd>1</kbd>–<kbd>0</kbd> <kbd>End</kbd> jump · <kbd>f</kbd> fullscreen</span>
      <span class="row" style=${{ gap: "8px" }}><img src=${I("dkv-mobility")} alt="" style=${{ height: "14px" }} />DKV Mobility case study · ${idx + 1} / ${TABS.length}</span></footer>
  </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
