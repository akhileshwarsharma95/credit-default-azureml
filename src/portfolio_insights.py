"""Portfolio insights - SQL (DuckDB) analytics on the scored portfolio:
KPIs, grade/stage/segment views, early-warning watchlist, limit steering,
roll rates and a 3-month delinquency projection. Writes Power BI-ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from creditrisk import insights, plots
from creditrisk.schema import read_table
from creditrisk.tracking import log_metrics, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored_data", required=True)
    p.add_argument("--insights_output", required=True)
    p.add_argument("--n_segments", type=int, default=5)
    p.add_argument("--top_n", type=int, default=50)
    return p.parse_args(argv)


def report(kpis: dict, tables: dict) -> str:
    g = tables["grade_summary"]
    worst = g.sort_values("el_share", ascending=False).iloc[0]
    seg = tables["segment_summary"].iloc[0]
    la = tables["limit_actions"].set_index("limit_action")
    reduce = la.loc["Reduce limit"] if "Reduce limit" in la.index else None
    lines = [
        "# Portfolio insights",
        "",
        f"- {kpis['clients']:,.0f} clients, exposure {kpis['total_ead']:,.0f}, expected loss {kpis['total_el']:,.0f} ({kpis['el_rate']:.2%} of EAD)",
        f"- Grade {worst['grade']} carries {worst['el_share']:.0%} of expected loss with {worst['client_share']:.0%} of clients",
        f"- Riskiest segment: {seg['segment_name']} (mean PD {seg['mean_pd']:.1%}, {seg['client_share']:.0%} of clients)",
        f"- Delinquency {kpis['current_delinquency']:.1%} today; roll-rate projection {kpis['projected_delinquency_3m']:.1%} in 3 months",
    ]
    if reduce is not None:
        lines.append(
            f"- Limit reductions for {reduce['clients']:,.0f} high-risk clients cut EAD by {-reduce['ead_change']:,.0f} and expected loss by {-reduce['el_change']:,.0f}"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.insights_output)
    out.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run():
        mlflow.set_tags({"stage": "insights"})
        scored = read_table(args.scored_data, "scored.csv")
        kpis, tables = insights.portfolio_insights(scored, args.n_segments, args.top_n)
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        write_json(kpis, out / "kpis.json")
        text = report(kpis, tables)
        (out / "portfolio_report.md").write_text(text)
        plots.save(plots.roll_rate_heatmap(insights.roll_rate_matrix(tables["roll_rates"])), out / "roll_rates.png")
        log_metrics(kpis, prefix="portfolio_")
        mlflow.log_artifacts(str(out), artifact_path="insights")
        print(text)


if __name__ == "__main__":
    main()
