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
const studio = (run) =>
  `https://ml.azure.com/runs/${run}?wsid=/subscriptions/${WS.sub}/resourcegroups/${WS.rg}/workspaces/${WS.name}&tid=${WS.tenant}`;

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
  ci: { x: 112, y: 290, icon: "githubactions", t: "CI workflow", s: "ruff · 46 tests · dry-runs" },
  cd: { x: 112, y: 440, icon: "githubactions", t: "train-and-deploy", s: "OIDC · approval gate" },
  storage: { x: 315, y: 140, icon: "az/storage", t: "Blob storage", s: "versioned data assets" },
  acr: { x: 315, y: 290, icon: "az/container-registry", t: "Container Registry", s: "pinned env image" },
  entra: { x: 315, y: 440, icon: "az/app-registration", t: "Entra ID app", s: "federated OIDC" },
  kv: { x: 315, y: 590, icon: "az/key-vault", t: "Key Vault", s: "workspace secrets" },
  pre: { x: 478, y: 140, step: "1", t: "preprocess", s: "pandera DQ gate" },
  trn: { x: 598, y: 140, step: "2", t: "train", s: "Optuna · calibrate" },
  evl: { x: 718, y: 140, step: "3", t: "evaluate", s: "backtest · SHAP" },
  reg: { x: 838, y: 140, step: "4", t: "register", s: "gated" },
  cluster: { x: 598, y: 290, icon: "az/vm", t: "cpu-cluster", s: "D2as_v4 · 0–2 nodes" },
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
  ci: ["CI workflow (every push / PR)", "ruff format + lint, 46 tests (end to end on synthetic data) and a dry-run of every Azure ML spec, so a broken component never reaches Azure. Dependencies install from uv.lock."],
  cd: ["train-and-deploy workflow", "On push to main (model code) or manual dispatch: CI, then submit the Azure ML pipeline, wait, publish the model card to the job summary. Deployment only on manual dispatch, behind the GitHub production environment."],
  entra: ["Microsoft Entra ID app registration", "Workload identity federation: GitHub exchanges its OIDC token for an Azure token. No stored client secret. Contributor scoped to rg-credit-risk only."],
  storage: ["Storage account (workspace datastore)", "Raw UCI file registered as data asset credit-default-raw with a sha256 tag; batches registered as their own assets. Every model version points to its data version."],
  acr: ["Azure Container Registry", "Built once from the locked conda environment (environments/), reused by every job and the endpoint. Same image for training and scoring = no train/serve skew."],
  kv: ["Key Vault", "Created with the workspace; holds workspace and connection secrets. Nothing secret lives in the repo or in GitHub."],
  pre: ["Step 1 · preprocess", "pandera hard checks (types, ranges, nulls) fail the run; soft checks are reported. Documented remap of undocumented codes, de-duplication, stratified split before anything is fitted."],
  trn: ["Step 2 · train", "Optuna search; out-of-fold comparison of monotonic LightGBM vs unconstrained LightGBM vs logistic scorecard; isotonic calibration + 0.1% PD floor; cost-based cut-off. All logged to MLflow."],
  evl: ["Step 3 · evaluate", "Test-set AUC/Gini/KS with bootstrap CIs, calibration, Jeffreys backtest per grade, fairness, SHAP. Loads the registered champion and scores it on the same test set."],
  reg: ["Step 4 · register", "Champion/challenger gate: minimum AUC, calibration limit, not worse than the champion by more than 0.005 AUC. Registers model + model card + evaluation evidence + drift baseline."],
  cluster: ["Compute cluster cpu-cluster", "Standard_D2as_v4 (2 vCPU, 8 GB), 0–2 nodes, scales to zero after 3 minutes idle: cents per run, nothing billed when idle."],
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
    <div class="card rf-wrap">
      <${RF.ReactFlow} nodes=${nodes} edges=${edges} nodeTypes=${NODE_TYPES} edgeTypes=${EDGE_TYPES}
        colorMode=${dark ? "dark" : "light"} fitView fitViewOptions=${{ padding: 0.02 }} minZoom=${0.2} maxZoom=${2.5}
        nodesDraggable=${false} nodesConnectable=${false} elementsSelectable=${false}
        zoomOnScroll=${false} zoomOnDoubleClick=${false} preventScrolling=${false}
        onNodeClick=${(e, n) => n.type !== "zone" && setSel(n.id)}>
        <${RF.Controls} showInteractive=${false} position="bottom-right" />
      </${RF.ReactFlow}>
    </div>
    <div class="card detail section"><h3>${title}</h3><div>${body}</div></div>
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
      <span class="small muted">local: <code>make app</code> (registered model v2 in <code>app/model</code>)</span>
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
          <tr><td><code>ci</code></td><td>every push / PR</td><td>ruff, 46 tests, dry-run of every Azure spec</td></tr>
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
        <li>46 tests: units + end to end on synthetic data (~15 s); same suite in CI (<code>make check</code>)</li>
        <li>uv-locked dependencies; Azure environments generated from the same lock (<code>make lock</code>)</li>
        <li><code>make local</code> = the Azure pipeline, locally with an MLflow SQLite registry</li>
      </ul></div>
    </div>
  </div>`;
}

const RUNS = [
  ["Training pipeline", "cool_rhythm_hx51hfjj1r", "Completed", "preprocess (incl. one-time image build) 16 min · train 3 min · evaluate 1.5 min · register 1 min"],
  ["Batch monitoring (downturn)", "ivory_rabbit_9x86flykq4", "Completed", "score 1 min · monitor + insights in parallel 1.5 min · [DRIFT ALERT]"],
  ["Training via GitHub Actions (OIDC)", "brave_ocean_w3b8k8s0dq", "Completed", "triggered by train-and-deploy · gate compared v2 with champion v1 · registered v2"],
];

function Live() {
  return html`<div>
    <${Head} title="Live on Azure"
      lead="What I have running in my Azure subscription, the evidence for the deliverable, and what it costs." />
    <div class="grid g4">
      <div class="card"><${Svc} icon="az/machine-learning" name="mlw-credit-risk">Azure ML workspace · Germany West Central</${Svc}></div>
      <div class="card"><${Svc} icon="az/vm" name="cpu-cluster">Standard_D2as_v4 · 0–2 nodes · idle = 0 nodes</${Svc}></div>
      <div class="card"><${Svc} icon="az/machine-learning" name="credit-default-model v1, v2">test AUC 0.7785 · Gini 0.557 · ECE 0.0105 · v2 from CI</${Svc}></div>
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
        <li><b>Quota:</b> DSv2 family had 0 vCPUs in the region → cluster on Standard_D2as_v4 (DASv4, 4 vCPUs)</li>
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

const TABS = [
  ["overview", "Overview", Overview],
  ["architecture", "Architecture", Architecture],
  ["training", "Training pipeline", Training],
  ["model", "Model & validation", Model],
  ["serving", "Serving", Serving],
  ["observability", "Observability", Observability],
  ["reporting", "Reporting", Reporting],
  ["cockpit", "Cockpit app", Cockpit],
  ["governance", "CI/CD & governance", Governance],
  ["live", "Live on Azure", Live],
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
