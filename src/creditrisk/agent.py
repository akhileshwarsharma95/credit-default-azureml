"""'Talk to your portfolio': an LLM agent that answers business questions by
writing read-only SQL against the scored portfolio.

Guardrails: SELECT-only statements, a single statement, a deny-list of
keywords, a row cap, and a DuckDB connection with external access locked.
Every LLM call and tool call is traced (OpenTelemetry GenAI attribute names)
and also returned as a step list so the UI can show the reasoning trail.

Providers: Azure OpenAI (preferred), any OpenAI-compatible endpoint, or an
offline mode with SQL templates so a demo never depends on an API key.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from .observability import tracer

MAX_ROWS = 200
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|call|set|reset|"
    r"checkpoint|vacuum|grant|revoke|truncate|merge)\b",
    re.IGNORECASE,
)
_FILE_FUNCS = re.compile(r"\b(read_csv\w*|read_parquet|read_json\w*|glob|parquet_scan|csv_scan|http)\b", re.IGNORECASE)

SCHEMA_DOC = """\
Database: DuckDB. Amounts are in NT$ (the UCI dataset currency). Months: 1 = Sep (latest) ... 6 = Apr.
Table portfolio (one row per client of the scored test portfolio):
  ID, LIMIT_BAL (credit limit), EDUCATION (1 grad school, 2 university, 3 high school, 4 other),
  PAY_1..PAY_6 (repayment status: -2 no usage, -1 paid in full, 0 revolving/minimum paid, k = k months late),
  BILL_AMT1..6, PAY_AMT1..6, util_1 (latest utilisation), util_avg, pay_ratio_avg (paid/billed), n_delays, max_delay,
  pd (calibrated probability of payment default next month), grade (A best .. G worst),
  action ('Standard','Watch','Intervene'), flag (1 if pd >= policy cut-off),
  ead (exposure at default), el (expected loss = pd * LGD * ead), ifrs9_stage (1,2,3), ecl,
  ews_count, ews_triggers (early-warning triggers, text), limit_action ('Keep','Reduce limit','Increase limit'),
  recommended_limit, el_change, reason_1..reason_4 (top risk drivers, plain language), segment_name,
  "default" (observed outcome 0/1; quote the column name).
Pre-aggregated tables: grade_summary, segment_summary, stage_summary, limit_actions, roll_rate_matrix,
  delinquency_trend, delinquency_projection, backtest (PD backtest per grade with traffic_light), drift (latest PSI table).
"""

SYSTEM_PROMPT = f"""You are a credit-risk analytics assistant for a portfolio management team.
Answer questions about the scored portfolio by calling the run_sql tool. Never invent numbers:
every figure in your answer must come from a query result. Prefer the pre-aggregated tables when they fit.
Keep answers short (3-6 sentences or a compact table), lead with the business takeaway, and name
the metric definitions you relied on. If a question cannot be answered from the data, say so.

{SCHEMA_DOC}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only DuckDB SELECT query against the portfolio tables. Returns up to 200 rows as JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SELECT (or WITH ... SELECT) statement."}
                },
                "required": ["query"],
            },
        },
    }
]


class SqlGuardError(ValueError):
    pass


def guard_sql(query: str, max_rows: int = MAX_ROWS) -> str:
    q = query.strip().rstrip(";").strip()
    if not q:
        raise SqlGuardError("Empty query")
    if ";" in q:
        raise SqlGuardError("Only a single statement is allowed")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise SqlGuardError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(q):
        raise SqlGuardError("Query contains a forbidden keyword")
    if _FILE_FUNCS.search(q):
        raise SqlGuardError("File and network access is not allowed")
    return f"SELECT * FROM ({q}) AS _q LIMIT {int(max_rows)}"


@dataclass
class Step:
    kind: str
    name: str
    detail: str = ""
    duration_ms: float = 0.0
    tokens: int = 0


@dataclass
class AgentResult:
    answer: str
    mode: str
    steps: list[Step] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)


def _client_from_env():
    """Return (client, model, provider) or (None, None, 'offline')."""
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError:
        return None, None, "offline"
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        return client, os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"), "azure-openai"
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        return client, os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "openai"
    return None, None, "offline"


OFFLINE_TEMPLATES = [
    (
        r"\b(kpi|overview|summary|headline)\b",
        "Portfolio headline KPIs",
        "SELECT COUNT(*) AS clients, SUM(ead) AS total_ead, SUM(el) AS total_el, SUM(el)/SUM(ead) AS el_rate, AVG(pd) AS mean_pd, "
        'AVG(flag) AS intervene_rate, AVG("default") AS observed_dr FROM portfolio',
    ),
    (r"\b(backtest|calibrat|traffic)", "PD backtest by grade", "SELECT * FROM backtest"),
    (
        r"\bgrade",
        "Exposure and expected loss by rating grade",
        "SELECT grade, clients, client_share, mean_pd, observed_dr, ead, el, el_share FROM grade_summary ORDER BY grade",
    ),
    (r"\bsegment|cluster", "Behavioural segments", "SELECT * FROM segment_summary ORDER BY mean_pd DESC"),
    (r"\b(stage|ifrs|ecl)\b", "IFRS 9 staging (illustrative)", "SELECT * FROM stage_summary ORDER BY stage"),
    (r"\blimit", "Limit steering recommendations", "SELECT * FROM limit_actions"),
    (r"\b(drift|psi|stability)\b", "Latest drift check", "SELECT * FROM drift ORDER BY psi DESC LIMIT 10"),
    (
        r"\b(watch|early|warning|riskiest|top)\b",
        "Top early-warning clients by expected loss",
        "SELECT ID, grade, ROUND(pd, 3) AS pd, action, ews_triggers, ROUND(el) AS el, reason_1, reason_2 FROM portfolio "
        "WHERE action <> 'Standard' ORDER BY el DESC LIMIT 10",
    ),
    (
        r"\beducation\b",
        "Default rate and PD by education",
        'SELECT EDUCATION, COUNT(*) AS clients, AVG("default") AS observed_dr, AVG(pd) AS mean_pd FROM portfolio GROUP BY 1 ORDER BY 1',
    ),
    (
        r"\b(trend|delinquen|roll)",
        "Delinquency trend and 3-month projection",
        "SELECT month, delinquency_rate, utilisation FROM delinquency_trend ORDER BY month_idx",
    ),
]


class PortfolioAgent:
    def __init__(self, con, client=None, model: str | None = None, provider: str = "offline", max_turns: int = 6):
        self.con = con
        self.client = client
        self.model = model
        self.provider = provider
        self.max_turns = max_turns
        self._tracer = tracer()

    @classmethod
    def from_env(cls, con) -> PortfolioAgent:
        client, model, provider = _client_from_env()
        return cls(con, client, model, provider)

    @property
    def mode(self) -> str:
        return self.provider

    def run_sql(self, query: str) -> pd.DataFrame:
        safe = guard_sql(query)
        with self._tracer.start_as_current_span("agent.tool.run_sql") as span:
            span.set_attribute("db.system", "duckdb")
            span.set_attribute("db.statement", query[:2000])
            df = self.con.execute(safe).df()
            span.set_attribute("db.rows", len(df))
        return df

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        with self._tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("agent.mode", self.mode)
            span.set_attribute("agent.question", question[:500])
            if self.client is None:
                return self._offline(question)
            return self._llm(question, history or [])

    def _offline(self, question: str) -> AgentResult:
        cid = re.search(r"\b(?:client|id|customer)\s*#?\s*(\d+)", question, re.IGNORECASE)
        if cid:
            q = (
                "SELECT ID, grade, ROUND(pd, 3) AS pd, action, ifrs9_stage, ROUND(el) AS el, ews_triggers, limit_action, "
                f"reason_1, reason_2, reason_3, reason_4 FROM portfolio WHERE ID = {int(cid.group(1))}"
            )
            return self._offline_answer(f"Client {cid.group(1)}", q)
        for pattern, title, q in OFFLINE_TEMPLATES:
            if re.search(pattern, question, re.IGNORECASE):
                return self._offline_answer(title, q)
        help_text = (
            "Offline mode (no LLM configured) understands a fixed set of questions, e.g. "
            "'portfolio overview', 'exposure by grade', 'segments', 'IFRS 9 stages', 'limit recommendations', "
            "'backtest', 'drift', 'top watchlist', 'default rate by education', 'delinquency trend', 'client 123'. "
            "Configure Azure OpenAI to ask anything."
        )
        return AgentResult(answer=help_text, mode="offline")

    def _offline_answer(self, title: str, query: str) -> AgentResult:
        start = time.perf_counter()
        try:
            df = self.run_sql(query)
        except (SqlGuardError, duckdb.Error) as exc:
            return AgentResult(answer=f"Query failed: {exc}", mode="offline", queries=[query])
        step = Step("tool", "run_sql", query, round((time.perf_counter() - start) * 1000, 1))
        answer = f"**{title}** ({len(df)} row{'s' if len(df) != 1 else ''}). Offline template - configure an LLM for free-form questions."
        return AgentResult(answer=answer, mode="offline", steps=[step], queries=[query], tables=[df])

    def _llm(self, question: str, history: list[dict]) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[-6:], {"role": "user", "content": question}]
        result = AgentResult(answer="", mode=self.mode)
        for _ in range(self.max_turns):
            start = time.perf_counter()
            with self._tracer.start_as_current_span("agent.llm_call") as span:
                span.set_attribute("gen_ai.system", self.provider)
                span.set_attribute("gen_ai.request.model", self.model or "")
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=TOOLS, temperature=0
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            msg = resp.choices[0].message
            result.steps.append(
                Step(
                    "llm",
                    self.model or "llm",
                    (msg.content or "tool call")[:300],
                    round((time.perf_counter() - start) * 1000, 1),
                    getattr(usage, "total_tokens", 0) if usage else 0,
                )
            )
            if not msg.tool_calls:
                result.answer = msg.content or ""
                return result
            messages.append(
                {"role": "assistant", "content": msg.content, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]}
            )
            for tc in msg.tool_calls:
                start = time.perf_counter()
                try:
                    query = json.loads(tc.function.arguments).get("query", "")
                    df = self.run_sql(query)
                    result.queries.append(query)
                    result.tables.append(df)
                    content = df.head(50).to_json(orient="records", date_format="iso", double_precision=4)
                    detail = query
                except Exception as exc:  # noqa: BLE001 - any tool error goes back to the model so it can fix its call
                    content = json.dumps({"error": str(exc)})
                    detail = f"ERROR: {exc}"
                result.steps.append(Step("tool", "run_sql", detail, round((time.perf_counter() - start) * 1000, 1)))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        result.answer = "I could not finish within the allowed number of steps. Try a more specific question."
        return result
