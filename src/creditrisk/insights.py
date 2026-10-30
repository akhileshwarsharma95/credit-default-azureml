"""Portfolio insights in SQL (DuckDB): KPIs, grade and stage views, early-warning
watchlist, limit steering, behavioural segments, roll rates and a Markov-chain
projection of delinquency. Outputs are flat tables, ready for Power BI."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import migration, portfolio_risk
from .config import load

SQL_DIR = Path(__file__).parent / "sql"
BUCKETS = ["0 Current", "1 30 dpd", "2 60 dpd", "3 90+ dpd"]
SEGMENT_FEATURES = ["util_avg", "pay_ratio_avg", "n_delays", "n_paid_full", "n_no_consumption", "LIMIT_BAL"]


def sql(name: str) -> str:
    return (SQL_DIR / f"{name}.sql").read_text()


def connect(tables: dict[str, pd.DataFrame], read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with real tables; external access locked down when read_only."""
    con = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        con.register(f"_{name}_src", df)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "_{name}_src"')
        con.unregister(f"_{name}_src")
    if read_only:
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
    return con


def _name_segment(c: pd.Series) -> str:
    if c["n_no_consumption"] >= 3:
        return "Dormant / low usage"
    if c["n_delays"] >= 1.5:
        return "Delinquent revolvers"
    if c["n_paid_full"] >= 2.5 or c["pay_ratio_avg"] >= 0.8:
        return "Transactors (pay in full)"
    if c["util_avg"] >= 0.6:
        return "Highly utilised revolvers"
    return "Steady revolvers"


def segment_portfolio(scored: pd.DataFrame, n_segments: int = 5, seed: int = 42) -> pd.DataFrame:
    X = scored[SEGMENT_FEATURES].astype(float).copy()
    X["pay_ratio_avg"] = X["pay_ratio_avg"].clip(0, 1.5)
    X["util_avg"] = X["util_avg"].clip(-0.2, 1.5)
    X["LIMIT_BAL"] = np.log1p(X["LIMIT_BAL"])
    labels = KMeans(n_clusters=n_segments, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(X))
    centroids = scored[SEGMENT_FEATURES].assign(segment=labels).groupby("segment").mean()
    names = {seg: _name_segment(row) for seg, row in centroids.iterrows()}
    groups: dict[str, list] = {}
    for seg, name in names.items():
        groups.setdefault(name, []).append(seg)
    for name, segs in groups.items():  # disambiguate duplicate names by limit size
        if len(segs) > 1:
            segs = sorted(segs, key=lambda s: centroids.loc[s, "LIMIT_BAL"])
            suffixes = (
                ["lower limit", "higher limit"] if len(segs) == 2 else [f"tier {i + 1}" for i in range(len(segs))]
            )
            for s, suffix in zip(segs, suffixes, strict=False):
                names[s] = f"{name} - {suffix}"
    out = scored.copy()
    out["segment"] = labels
    out["segment_name"] = [names[s] for s in labels]
    return out


def roll_rate_matrix(roll_long: pd.DataFrame) -> pd.DataFrame:
    m = (
        roll_long.pivot(index="from_bucket", columns="to_bucket", values="roll_rate")
        .reindex(index=BUCKETS, columns=BUCKETS)
        .fillna(0.0)
    )
    return m.div(m.sum(axis=1).replace(0, 1), axis=0)


def bucket_mix(scored: pd.DataFrame) -> np.ndarray:
    s = scored["PAY_1"].to_numpy()
    return np.array([(s <= 0).mean(), (s == 1).mean(), (s == 2).mean(), (s >= 3).mean()])


def default_forecast(scored: pd.DataFrame, lifetime: pd.DataFrame, lgd: float) -> pd.DataFrame:
    """Expected new defaults and loss per month for the portfolio, from the marginal PD vectors.
    Clients already 90+ dpd are in default (stage 3) and excluded."""
    mix = dict(zip(BUCKETS, bucket_mix(scored), strict=True))
    avg_ead = float(scored["ead"].mean())
    t = lifetime.assign(weighted=lambda d: d["marginal_pd"] * d["from_bucket"].map(mix))
    out = t.groupby("month", as_index=False)["weighted"].sum().rename(columns={"weighted": "default_rate"})
    out["expected_defaults"] = out["default_rate"] * len(scored)
    out["expected_loss"] = out["expected_defaults"] * avg_ead * lgd
    return out


def project_delinquency(scored: pd.DataFrame, matrix: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    """Markov-chain projection of the bucket mix from the latest month."""
    current = bucket_mix(scored)
    P = matrix.to_numpy()
    rows, dist = [], current
    for step in range(1, months + 1):
        dist = dist @ P
        rows.append(
            {
                "months_ahead": step,
                **{b: float(v) for b, v in zip(BUCKETS, dist, strict=False)},
                "delinquency_rate": float(dist[1:].sum()),
            }
        )
    return pd.DataFrame(rows)


def portfolio_insights(
    scored: pd.DataFrame, n_segments: int = 5, top_n: int = 50, seed: int = 42
) -> tuple[dict, dict[str, pd.DataFrame]]:
    scored = segment_portfolio(scored, n_segments, seed)
    for col in ("reason_1", "reason_2", "reason_3"):
        if col not in scored.columns:
            scored[col] = ""
    con = connect({"scored": scored})
    kpis = con.execute(sql("kpis")).df().iloc[0].to_dict()
    tables = {
        "fact_scores": scored,
        "grade_summary": con.execute(sql("grade_summary")).df(),
        "stage_summary": con.execute(sql("stage_summary")).df(),
        "watchlist": con.execute(sql("watchlist"), [top_n]).df(),
        "limit_actions": con.execute(sql("limit_actions")).df(),
        "segment_summary": con.execute(sql("segment_summary")).df(),
        "roll_rates": con.execute(sql("roll_rates")).df(),
        "delinquency_trend": con.execute(sql("delinquency_trend")).df(),
    }
    matrix = roll_rate_matrix(tables["roll_rates"])
    tables["roll_rate_matrix"] = matrix.reset_index().rename(columns={"from_bucket": "from_bucket"})
    tables["delinquency_projection"] = project_delinquency(scored, matrix)
    kpis = {k: (None if pd.isna(v) else float(v)) for k, v in kpis.items()}
    kpis["projected_delinquency_3m"] = float(tables["delinquency_projection"]["delinquency_rate"].iloc[-1])
    kpis["current_delinquency"] = float(tables["delinquency_trend"]["delinquency_rate"].iloc[-1])

    # Portfolio view: lifetime PD vectors, monthly default forecast, loss distribution, concentration
    business = load()["business"]
    lgd, rho = business["loss"]["lgd"], business["portfolio"]["asset_correlation"]
    tables["lifetime_pd"] = migration.lifetime_pd(matrix, months=12)
    tables["default_forecast"] = default_forecast(scored, tables["lifetime_pd"], lgd)
    tables["loss_quantiles"] = portfolio_risk.loss_quantiles(scored["ead"], lgd, scored["pd"], rho)
    kpis["unexpected_loss_999"] = float(tables["loss_quantiles"]["unexpected_loss"].iloc[-1])
    kpis["expected_loss_12m_forecast"] = float(tables["default_forecast"]["expected_loss"].sum())
    kpis |= portfolio_risk.concentration(scored["ead"], scored["segment_name"])
    con.close()
    return kpis, tables
