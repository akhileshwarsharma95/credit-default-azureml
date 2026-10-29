"""Credit Risk Cockpit - front end for the PD engine.

Portfolio view, client scoring with reason codes, cut-off and limit policy,
model validation and drift monitoring, a 'talk to your portfolio' agent and
the model card. Uses the same creditrisk package as the Azure ML pipelines.

    MODEL_DIR=outputs/local/model_package streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from creditrisk import drift, explain, insights, metrics, policy
from creditrisk.agent import PortfolioAgent
from creditrisk.observability import configure, timed
from creditrisk.schema import EDUCATION_LABELS, MONTHS, PAY_STATUS_LABELS, describe
from creditrisk.scoring import load_package, score_frame

ROOT = Path(__file__).resolve().parents[1]

configure("credit-risk-app")
st.set_page_config(page_title="Credit Risk Cockpit", page_icon=":material/monitoring:", layout="wide")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
STATUS = {
    "green": "#0ca30c",
    "amber": "#fab219",
    "red": "#d03b3b",
    "stable": "#0ca30c",
    "warn": "#fab219",
    "alert": "#d03b3b",
    "insufficient": "#898781",
}
LIGHT_ICON = {"green": "🟢 green", "amber": "🟠 amber", "red": "🔴 red", "insufficient": "⚪ too few clients"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CCY = "NT$"
SHORT_STATUS = {-2: "No use", -1: "Paid full", 0: "Revolving"}


# --------------------------------------------------------------------------- helpers
def money(v: float, ccy: bool = True) -> str:
    v = float(v)
    prefix = f"{CCY} " if ccy else ""
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{prefix}{v / div:,.1f}{unit}"
    return f"{prefix}{v:,.0f}"


def tile(col, label: str, value: str, note: str | None = None, help: str | None = None) -> None:
    col.metric(label, value, note, delta_color="off", delta_arrow="off", border=True, help=help)


def pct(v, nd: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}%}"


def style(fig: go.Figure, title: str | None = None, height: int = 330, legend: bool = True) -> go.Figure:
    fig._cockpit_title = title  # rendered above the chart by show(), so it never collides with the legend
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": INK2, "size": 12},
        margin={"l": 10, "r": 10, "t": 30 if legend else 10, "b": 10},
        height=height,
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "font": {"color": INK2}},
        hoverlabel={"bgcolor": "white", "font_color": INK},
        bargap=0.45,
    )
    fig.update_xaxes(
        gridcolor=GRID,
        linecolor=AXIS,
        zeroline=False,
        tickfont={"color": INK2},
        automargin=True,
        title_font={"color": INK2},
    )
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=AXIS,
        zeroline=False,
        tickfont={"color": INK2},
        automargin=True,
        title_font={"color": INK2},
    )
    return fig


def show(fig: go.Figure) -> None:
    title = getattr(fig, "_cockpit_title", None)
    if title:
        st.markdown(f"**{title}**")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def model_dir() -> Path:
    for candidate in (os.getenv("MODEL_DIR"), ROOT / "app" / "model", ROOT / "outputs" / "local" / "model_package"):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    st.error(
        "No model package found. Run `python scripts/run_local.py` or `python scripts/fetch_model.py --output app/model`."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading model package ...")
def get_package(path: str):
    return load_package(path)


pkg = get_package(str(model_dir()))
FEATURES = pkg.features
PORTFOLIO_RAW = pkg.read_csv("evaluation/test_scored.csv")
MODEL_LABEL = (
    (pkg.path / "REGISTRY_VERSION").read_text().strip() if (pkg.path / "REGISTRY_VERSION").exists() else pkg.path.name
)


@st.cache_data(show_spinner="Scoring portfolio ...")
def scored_portfolio(threshold: float, lgd: float, ccf: float, cost_fn: float, cost_fp: float):
    a = policy.Assumptions(lgd=lgd, ccf=ccf, cost_fn=cost_fn, cost_fp=cost_fp)
    base = PORTFOLIO_RAW
    s = score_frame(pkg, base, assumptions=a, threshold=threshold, with_explanations=False)
    for c in [c for c in base.columns if c.startswith("reason_")]:
        s[c] = base[c].fillna("").to_numpy()
    kpis, tables = insights.portfolio_insights(s)
    return tables["fact_scores"], kpis, tables


def agent_connection(tables: dict, key: tuple):
    if st.session_state.get("con_key") != key:
        extra = {
            "backtest": pkg.read_csv("evaluation/backtest.csv"),
            "drift": st.session_state.get("drift_table", pd.DataFrame({"feature": [], "psi": [], "status": []})),
        }
        db = {"portfolio": tables["fact_scores"], **{k: v for k, v in tables.items() if k != "fact_scores"}, **extra}
        st.session_state["con"] = insights.connect(db)
        st.session_state["con_key"] = key
    return st.session_state["con"]


# --------------------------------------------------------------------------- sidebar
summary = pkg.summary
m = pkg.read_json("evaluation/metrics.json")
with st.sidebar:
    st.markdown("### Credit Risk Cockpit")
    st.caption(f"Model **{MODEL_LABEL}** · {summary['model_family']} · trained {summary['trained_at'][:10]}")
    st.markdown("**Policy assumptions**")
    cost_fn = st.number_input(
        "Cost of a missed default",
        1.0,
        50.0,
        float(summary["assumptions"]["cost_fn"]),
        0.5,
        help="Relative to the cost of acting on a good client (lost margin, friction)",
    )
    cost_fp = st.number_input("Cost of a false alarm", 0.1, 10.0, float(summary["assumptions"]["cost_fp"]), 0.1)
    optimal = policy.cost_optimal_threshold(cost_fn, cost_fp)
    use_optimal = st.toggle(
        "Use cost-optimal cut-off", True, help="With calibrated PDs: cut-off = cost_fp / (cost_fp + cost_fn)"
    )
    threshold = (
        optimal if use_optimal else st.slider("Intervention cut-off (PD)", 0.02, 0.80, float(round(optimal, 2)), 0.01)
    )
    st.caption(f"Cut-off in use: **PD ≥ {threshold:.3f}**")
    lgd = st.slider("LGD", 0.10, 1.00, float(summary["assumptions"]["lgd"]), 0.05)
    ccf = st.slider("CCF on undrawn limit", 0.0, 1.0, float(summary["assumptions"]["ccf"]), 0.05)
    st.divider()
    st.caption("Portfolio = the held-out test clients scored by this model version. Amounts in NT$ (UCI data).")

fact, kpis, tables = scored_portfolio(threshold, lgd, ccf, cost_fn, cost_fp)
con_key = (round(threshold, 4), lgd, ccf, cost_fn, cost_fp, len(st.session_state.get("drift_table", [])))

st.markdown("## Credit Risk Cockpit")
st.caption(
    "Probability of default, expected loss and early warning for a credit card portfolio - trained, validated and registered by an Azure ML pipeline."
)

tab_portfolio, tab_client, tab_policy, tab_model, tab_agent, tab_card = st.tabs(
    ["Portfolio", "Client scoring", "Cut-off & limits", "Model & monitoring", "Ask the portfolio", "Model card"]
)

# --------------------------------------------------------------------------- portfolio
with tab_portfolio:
    c = st.columns(6)
    tile(c[0], "Clients", f"{kpis['clients']:,.0f}")
    tile(c[1], f"Exposure ({CCY})", money(kpis["total_ead"], False), "EAD")
    tile(c[2], f"Expected loss ({CCY})", money(kpis["total_el"], False), f"{kpis['el_rate']:.2%} of EAD")
    tile(c[3], "Mean PD", pct(kpis["mean_pd"]), f"observed {pct(kpis['observed_dr'])}")
    tile(c[4], "Intervene", f"{kpis['intervene_rate']:.0%}", f"watch {kpis['watch_rate']:.0%}")
    tile(c[5], "Stage 2+3 EAD", pct(kpis["stage2_ead_share"] + kpis["stage3_ead_share"]), "IFRS 9, illustrative")

    g = tables["grade_summary"]
    left, right = st.columns(2)
    with left:
        fig = go.Figure(
            go.Bar(
                x=g["grade"],
                y=g["el"],
                marker_color=BLUE,
                width=0.5,
                customdata=np.stack([g["el_share"], g["client_share"]], axis=1),
                hovertemplate="Grade %{x}<br>EL %{y:,.0f}<br>%{customdata[0]:.0%} of EL, %{customdata[1]:.0%} of clients<extra></extra>",
            )
        )
        worst = g.loc[g["el_share"].idxmax()]
        fig.add_annotation(
            x=worst["grade"],
            y=worst["el"],
            text=f"{worst['el_share']:.0%} of EL",
            showarrow=False,
            yshift=12,
            font={"color": INK},
        )
        show(style(fig, "Expected loss by rating grade", legend=False))
    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=g["grade"],
                y=g["mean_pd"],
                mode="markers",
                name="Mean PD",
                marker={"size": 12, "color": SURFACE, "line": {"color": INK2, "width": 2}},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=g["grade"],
                y=g["observed_dr"],
                mode="markers",
                name="Observed default rate",
                marker={"size": 12, "color": ORANGE, "line": {"color": SURFACE, "width": 2}},
            )
        )
        fig.update_yaxes(tickformat=".0%")
        show(style(fig, "Calibration by grade: predicted vs observed"))

    left, right = st.columns(2)
    with left:
        trend, proj = tables["delinquency_trend"], tables["delinquency_projection"]
        last = trend.iloc[-1]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend["month"],
                y=trend["delinquency_rate"],
                mode="lines+markers",
                name="Actual",
                line={"color": BLUE, "width": 2},
                marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
            )
        )
        months_ahead = ["Oct", "Nov", "Dec"][: len(proj)]
        fig.add_trace(
            go.Scatter(
                x=["Sep", *months_ahead],
                y=[last["delinquency_rate"], *proj["delinquency_rate"]],
                mode="lines+markers",
                name="Roll-rate projection",
                line={"color": BLUE, "width": 2, "dash": "dot"},
                marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
            )
        )
        fig.update_yaxes(tickformat=".0%")
        show(style(fig, "Share of clients 30+ days past due"))
    with right:
        mtx = insights.roll_rate_matrix(tables["roll_rates"])
        labels = [b.split(" ", 1)[1] for b in mtx.columns]
        z = mtx.to_numpy()
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)],
                zmin=0,
                zmax=1,
                showscale=False,
                text=[[f"{v:.0%}" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="%{y} -> %{x}: %{z:.1%}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed", title="This month")
        fig.update_xaxes(title="Next month")
        show(style(fig, "Monthly roll rates", legend=False))

    st.markdown("#### Behavioural segments")
    seg = tables["segment_summary"]
    st.dataframe(
        seg,
        hide_index=True,
        column_config={
            "segment_name": "Segment",
            "clients": st.column_config.NumberColumn("Clients", format="%d"),
            "client_share": st.column_config.NumberColumn("Share", format="percent"),
            "mean_pd": st.column_config.NumberColumn("Mean PD", format="percent"),
            "observed_dr": st.column_config.NumberColumn("Observed DR", format="percent"),
            "ead": st.column_config.NumberColumn("EAD", format="compact"),
            "el": st.column_config.NumberColumn("EL", format="compact"),
            "mean_utilisation": st.column_config.NumberColumn("Utilisation", format="percent"),
            "mean_repayment_ratio": st.column_config.NumberColumn("Repayment ratio", format="percent"),
            "mean_late_months": st.column_config.NumberColumn("Late months (of 6)", format="%.1f"),
            "mean_limit": st.column_config.NumberColumn("Mean limit", format="compact"),
        },
    )

# --------------------------------------------------------------------------- client scoring
with tab_client:
    ids = fact["ID"].astype(int).tolist()
    pick_col, preset_col = st.columns([2, 3])
    with pick_col:
        default_id = int(fact.sort_values("el", ascending=False)["ID"].iloc[0])
        client_id = st.selectbox(
            "Load an existing client", ids, index=ids.index(st.session_state.get("client_id", default_id))
        )
    with preset_col:
        st.write("")
        b = st.columns(3)
        if b[0].button("Highest expected loss", width="stretch"):
            st.session_state["client_id"] = int(fact.sort_values("el", ascending=False)["ID"].iloc[0])
            st.rerun()
        if b[1].button("Typical client", width="stretch"):
            st.session_state["client_id"] = int(
                fact.iloc[(fact["pd"] - fact["pd"].median()).abs().argsort().iloc[0]]["ID"]
            )
            st.rerun()
        if b[2].button("Low risk", width="stretch"):
            st.session_state["client_id"] = int(fact.sort_values("pd")["ID"].iloc[10])
            st.rerun()

    base_row = PORTFOLIO_RAW[PORTFOLIO_RAW["ID"] == client_id].iloc[0]
    with st.form("client_form"):
        st.markdown("**Client features** (edit any value for a what-if)")
        c1, c2 = st.columns([1, 3])
        with c1:
            limit = st.number_input("Credit limit", 1000, 2_000_000, int(base_row["LIMIT_BAL"]), 10_000)
            edu = (
                st.selectbox(
                    "Education",
                    list(EDUCATION_LABELS),
                    index=list(EDUCATION_LABELS).index(int(base_row["EDUCATION"])),
                    format_func=lambda k: EDUCATION_LABELS[k],
                )
                if "EDUCATION" in FEATURES
                else None
            )
        with c2:
            cols = st.columns(6)
            status, bills, pays = {}, {}, {}
            for i, month in enumerate(MONTHS, start=1):
                with cols[i - 1]:
                    st.caption(month)
                    status[i] = st.selectbox(
                        f"Status {month}",
                        list(range(-2, 9)),
                        index=int(base_row[f"PAY_{i}"]) + 2,
                        format_func=lambda k: SHORT_STATUS.get(k, f"{k} mo late"),
                        key=f"s{i}",
                        label_visibility="collapsed",
                        help="Repayment status: "
                        + ", ".join(f"{k}: {v}" for k, v in PAY_STATUS_LABELS.items() if k <= 2),
                    )
                    bills[i] = st.number_input(
                        f"Bill {month}", -500_000, 2_000_000, int(base_row[f"BILL_AMT{i}"]), 1000, key=f"b{i}"
                    )
                    pays[i] = st.number_input(
                        f"Paid {month}", 0, 2_000_000, int(base_row[f"PAY_AMT{i}"]), 500, key=f"p{i}"
                    )
        remote_ok = bool(os.getenv("SCORING_URI"))
        use_remote = st.toggle(
            "Score via the Azure ML online endpoint",
            False,
            disabled=not remote_ok,
            help="Set SCORING_URI and SCORING_KEY to call the managed online endpoint" if not remote_ok else None,
        )
        submitted = st.form_submit_button("Score client", type="primary")

    row = {"ID": client_id, "LIMIT_BAL": limit, **({"EDUCATION": edu} if edu is not None else {})}
    for i in range(1, 7):
        row |= {f"PAY_{i}": status[i], f"BILL_AMT{i}": bills[i], f"PAY_AMT{i}": pays[i]}
    for col in FEATURES:
        row.setdefault(col, base_row[col])
    client_df = pd.DataFrame([row])

    with timed("app.score_client", client_id=client_id, remote=use_remote):
        res = score_frame(
            pkg, client_df, policy.Assumptions(lgd=lgd, ccf=ccf, cost_fn=cost_fn, cost_fp=cost_fp), threshold
        ).iloc[0]
        contrib, base_val = explain.contributions(pkg.model, client_df[FEATURES], pkg.background)
    remote = None
    if use_remote and submitted:
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(
                os.environ["SCORING_URI"],
                data=json.dumps({"clients": [row]}, default=float).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('SCORING_KEY', '')}"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                remote = json.loads(r.read())
            remote["roundtrip_ms"] = round((time.perf_counter() - t0) * 1000)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            st.error(f"Endpoint call failed: {exc}")

    k = st.columns(5)
    tile(k[0], "Probability of default", pct(res["pd"]), f"rating grade {res['grade']}")
    tile(k[1], "Recommended action", res["action"], f"cut-off {threshold:.2f}")
    tile(k[2], f"Expected loss ({CCY})", money(res["el"], False), f"EAD {money(res['ead'], False)}")
    tile(k[3], "IFRS 9 stage", f"Stage {int(res['ifrs9_stage'])}", f"ECL {money(res['ecl'], False)}")
    tile(
        k[4],
        "Limit advice",
        res["limit_action"],
        f"to {money(res['recommended_limit'], False)}" if res["limit_action"] != "Keep" else "no change",
    )
    if remote:
        p0 = remote["predictions"][0]
        st.success(
            f"Azure ML endpoint ({remote['model_version']}): PD {p0['pd']:.1%}, grade {p0['grade']}, {p0['action']} - "
            f"server {remote['latency_ms']} ms, round trip {remote['roundtrip_ms']} ms"
        )

    left, right = st.columns([3, 2])
    with left:
        cs = contrib.iloc[0]
        top = list(cs.abs().sort_values(ascending=False).index[:8])
        names = ["All other features"] + [describe(f) for f in reversed(top)]  # plotly draws bottom-up
        vals = [float(cs.drop(top).sum())] + [float(cs[f]) for f in reversed(top)]
        fig = go.Figure(
            go.Bar(
                y=names,
                x=vals,
                orientation="h",
                width=0.55,
                marker_color=[RED if v > 0 else BLUE for v in vals],
                hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=AXIS, line_width=1)
        fig.update_xaxes(title="Contribution to default risk (log-odds; red raises, blue lowers)")
        show(style(fig, "Why this score (SHAP)", height=360, legend=False))
    with right:
        st.markdown("**Reason codes** (main factors raising the risk)")
        reasons = [res.get(f"reason_{i}") for i in range(1, 5) if res.get(f"reason_{i}")]
        for i, r in enumerate(reasons, start=1):
            st.markdown(
                f"{i}. {r}  \n<span style='color:{MUTED};font-size:0.8em'>{res.get(f'reason_code_{i}', '')}</span>",
                unsafe_allow_html=True,
            )
        if not reasons:
            st.caption("No material risk-raising factors.")
        st.markdown("**Early-warning triggers**")
        st.write(res["ews_triggers"] or "None")

# --------------------------------------------------------------------------- cut-off & limits
with tab_policy:
    y, p = fact["default"].astype(int), fact["pd"]
    tm = metrics.threshold_metrics(y, p, threshold, cost_fn, cost_fp)
    c = st.columns(5)
    tile(c[0], "Defaults caught", pct(tm["recall"]), "recall")
    tile(c[1], "Hit rate", pct(tm["precision"]), "precision of interventions")
    tile(c[2], "Clients flagged", pct(tm["flagged_rate"]), f"{tm['tp'] + tm['fp']:,} clients")
    tile(c[3], "False alarms", f"{tm['fp']:,}", f"missed defaults {tm['fn']:,}")
    tile(c[4], "Cost vs. no model", f"-{tm['cost_saving_vs_no_model']:.0%}", "expected cost per client")

    left, right = st.columns(2)
    with left:
        curve = metrics.cost_curve(y, p, cost_fn, cost_fp)
        best = curve.loc[curve["expected_cost_per_client"].idxmin()]
        fig = go.Figure(
            go.Scatter(
                x=curve["threshold"],
                y=curve["expected_cost_per_client"],
                mode="lines",
                line={"color": BLUE, "width": 2},
                name="Expected cost",
                hovertemplate="cut-off %{x:.2f}: cost %{y:.3f}<extra></extra>",
            )
        )
        fig.add_vline(
            x=threshold,
            line_color=INK2,
            line_width=1,
            annotation_text=f"in use {threshold:.2f}",
            annotation_font_color=INK2,
        )
        fig.add_trace(
            go.Scatter(
                x=[best["threshold"]],
                y=[best["expected_cost_per_client"]],
                mode="markers",
                name="Empirical minimum",
                marker={"size": 11, "color": ORANGE, "line": {"color": SURFACE, "width": 2}},
            )
        )
        fig.update_xaxes(title="PD cut-off")
        fig.update_yaxes(title="Expected cost per client")
        show(style(fig, "Cut-off economics (test portfolio)"))
    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["flagged_rate"],
                y=curve["recall"],
                mode="lines",
                line={"color": BLUE, "width": 2},
                name="Model",
                hovertemplate="flag %{x:.0%} -> catch %{y:.0%}<extra></extra>",
            )
        )
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": AXIS, "width": 1}, name="Random"))
        fig.add_trace(
            go.Scatter(
                x=[tm["flagged_rate"]],
                y=[tm["recall"]],
                mode="markers",
                name="Current cut-off",
                marker={"size": 11, "color": ORANGE, "line": {"color": SURFACE, "width": 2}},
            )
        )
        fig.update_xaxes(title="Share of clients flagged", tickformat=".0%")
        fig.update_yaxes(title="Share of defaults caught", tickformat=".0%")
        show(style(fig, "Workload vs. defaults caught"))

    st.markdown("#### Limit steering")
    la = tables["limit_actions"]
    st.dataframe(
        la,
        hide_index=True,
        column_config={
            "limit_action": "Action",
            "clients": st.column_config.NumberColumn("Clients", format="%d"),
            "mean_pd": st.column_config.NumberColumn("Mean PD", format="percent"),
            "current_limit": st.column_config.NumberColumn("Current limits", format="compact"),
            "recommended_limit": st.column_config.NumberColumn("Recommended", format="compact"),
            "ead_change": st.column_config.NumberColumn("EAD change", format="compact"),
            "el_change": st.column_config.NumberColumn("EL change", format="compact"),
        },
    )
    st.markdown("#### Early-warning watchlist (top 50 by expected loss)")
    wl = tables["watchlist"]
    st.dataframe(
        wl,
        hide_index=True,
        height=320,
        column_config={
            "pd": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1),
            "ead": st.column_config.NumberColumn("EAD", format="compact"),
            "el": st.column_config.NumberColumn("EL", format="compact"),
            "recommended_limit": st.column_config.NumberColumn("Recommended limit", format="compact"),
        },
    )
    st.download_button("Download watchlist (CSV)", wl.to_csv(index=False), "watchlist.csv", "text/csv")

# --------------------------------------------------------------------------- model & monitoring
with tab_model:
    c = st.columns(5)
    tile(c[0], "Test AUC", f"{m['auc']:.3f}", f"95% CI {m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f}")
    tile(c[1], "Gini", f"{m['gini']:.3f}", f"CV {summary['cv_gini']:.3f}")
    tile(c[2], "KS", f"{m['ks']:.3f}", "max separation")
    tile(c[3], "Calibration error", f"{m['ece']:.4f}", f"ECE; uncalibrated {m['uncalibrated_ece']:.4f}")
    tile(c[4], "Brier score", f"{m['brier']:.4f}", f"Hosmer-Lemeshow p {m['hosmer_lemeshow_p']:.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**PD backtest by grade** (Jeffreys test on the held-out set)")
        bt = pkg.read_csv("evaluation/backtest.csv")
        bt["result"] = bt["traffic_light"].map(LIGHT_ICON)
        st.dataframe(
            bt[
                ["grade", "pd_range", "n", "defaults", "observed_dr", "mean_pd", "jeffreys_p_underestimation", "result"]
            ],
            hide_index=True,
            column_config={
                "grade": "Grade",
                "pd_range": "PD band",
                "defaults": st.column_config.NumberColumn("Defaults", format="%d"),
                "result": st.column_config.TextColumn("Result", width="medium"),
                "observed_dr": st.column_config.NumberColumn("Observed DR", format="percent"),
                "mean_pd": st.column_config.NumberColumn("Mean PD", format="percent"),
                "jeffreys_p_underestimation": st.column_config.NumberColumn("Jeffreys p", format="%.3f"),
                "n": st.column_config.NumberColumn("Clients", format="%d"),
            },
        )
    with right:
        imp = pkg.read_csv("evaluation/importance.csv").head(12).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=imp["mean_abs_shap"],
                y=imp["description"],
                orientation="h",
                width=0.55,
                marker_color=BLUE,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Mean |SHAP| (log-odds)")
        show(style(fig, "Global risk drivers", height=380, legend=False))

    st.markdown("#### Drift monitor - stress the portfolio and check population stability")
    sev = st.select_slider(
        "Simulated downturn severity",
        options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        value=0.0,
        format_func=lambda v: "None (current portfolio)" if v == 0 else f"{v:g}x",
    )
    batch = PORTFOLIO_RAW[["ID", *FEATURES]].copy()
    if sev > 0:
        batch = drift.simulate_downturn(batch, severity=sev)
    with timed("app.drift_check", severity=sev):
        sb = score_frame(pkg, batch, with_explanations=False)
        dt, ds = drift.drift_report(pkg.profile, sb[[c for c in pkg.profile["features"] if c in sb.columns]], sb["pd"])
    st.session_state["drift_table"] = dt
    msg = (
        f"Score PSI {ds['score_psi']:.3f} · grade PSI {ds['grade_psi']:.3f} · {ds['n_features_alert']} features in alert · "
        f"mean PD {ds['reference_mean_pd']:.1%} → {ds['current_mean_pd']:.1%}"
    )
    {"stable": st.success, "warn": st.warning, "alert": st.error}[ds["overall_status"]](
        f"**{ds['overall_status'].upper()}** - {msg}"
    )
    left, right = st.columns(2)
    with left:
        top = dt.head(12).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=top["psi"],
                y=top["feature"],
                orientation="h",
                width=0.55,
                marker_color=[STATUS[s] for s in top["status"]],
                customdata=top["status"],
                hovertemplate="%{y}: PSI %{x:.3f} (%{customdata})<extra></extra>",
            )
        )
        for v, lbl in ((drift.PSI_WARN, "warn"), (drift.PSI_ALERT, "alert")):
            fig.add_vline(x=v, line_color=INK2, line_width=1, annotation_text=lbl, annotation_font_color=INK2)
        fig.update_xaxes(title="Population stability index")
        show(style(fig, "Feature drift (top 12)", height=380, legend=False))
    with right:
        edges = np.asarray(pkg.profile["score"]["edges"])
        exp_sh = np.asarray(pkg.profile["score"]["expected"])
        act = np.histogram(np.round(sb["pd"], 10), bins=edges)[0] / len(sb)
        labels = [f"≤{edges[i + 1]:.2f}" if i < len(exp_sh) - 1 else f">{edges[i]:.2f}" for i in range(len(exp_sh))]
        fig = go.Figure(
            [
                go.Bar(x=labels, y=exp_sh, name="Training reference", marker_color=BLUE),
                go.Bar(x=labels, y=act, name="Current batch", marker_color=ORANGE),
            ]
        )
        fig.update_yaxes(tickformat=".0%", title="Share of clients")
        fig.update_xaxes(title="PD bin")
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
        show(style(fig, "Score distribution shift", height=380))

# --------------------------------------------------------------------------- agent
with tab_agent:
    con = agent_connection(tables, con_key)
    agent = PortfolioAgent.from_env(con)
    mode_label = {"azure-openai": "Azure OpenAI", "openai": "OpenAI-compatible", "offline": "Offline templates"}[
        agent.mode
    ]
    st.badge(f"LLM: {mode_label}", icon=":material/smart_toy:", color="blue" if agent.mode != "offline" else "gray")
    st.caption(
        "Answers come from read-only SQL over the scored portfolio (DuckDB, SELECT-only, external access locked). Every step is traced."
    )
    examples = [
        "Expected loss by grade",
        "Top watchlist clients",
        "How do segments differ in risk?",
        "IFRS 9 stage split",
        "Did the backtest flag anything?",
        "Explain client " + str(int(fact.sort_values("el").iloc[-1]["ID"])),
    ]
    clicked = None
    for row_start in (0, 3):
        cols = st.columns(3)
        for q, col in zip(examples[row_start : row_start + 3], cols, strict=False):
            if col.button(q, width="stretch", key=f"ex_{q}"):
                clicked = q
    st.session_state.setdefault("chat", [])
    chat_area = st.container()  # conversation renders above the input box
    question = (
        st.chat_input("Ask about the portfolio, e.g. 'expected loss by segment for clients with utilisation above 80%'")
        or clicked
    )

    def table(df: pd.DataFrame) -> None:
        st.dataframe(df.round(4), hide_index=True)

    with chat_area:
        for turn in st.session_state["chat"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                for t in turn.get("tables", []):
                    table(t)
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            history = [{"role": t["role"], "content": t["content"]} for t in st.session_state["chat"]]
            with st.chat_message("assistant"), st.spinner("Querying ..."):
                result = agent.ask(question, history)
                st.markdown(result.answer)
                for t in result.tables[-2:]:
                    table(t)
                with st.expander(f"Trace - {len(result.steps)} steps, {sum(s.tokens for s in result.steps)} tokens"):
                    for step in result.steps:
                        st.markdown(
                            f"`{step.kind}` **{step.name}** · {step.duration_ms:.0f} ms"
                            + (f" · {step.tokens} tokens" if step.tokens else "")
                        )
                        if step.kind == "tool":
                            st.code(step.detail, language="sql")
            st.session_state["chat"] += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.answer, "tables": result.tables[-2:]},
            ]

# --------------------------------------------------------------------------- model card
with tab_card:
    card = pkg.path / "MODEL_CARD.md"
    left, right = st.columns([3, 2])
    with left:
        st.markdown(card.read_text() if card.exists() else "_No model card in this package._")
    with right:
        plots_dir = pkg.path / "evaluation" / "plots"
        if plots_dir.exists():
            options = sorted(p.name for p in plots_dir.glob("*.png"))
            choice = st.selectbox(
                "Validation chart", options, index=options.index("backtest.png") if "backtest.png" in options else 0
            )
            st.image(str(plots_dir / choice))
