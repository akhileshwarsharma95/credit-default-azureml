"""Validation and monitoring charts (matplotlib, headless).

One visual system for every chart: light surface, recessive hairline grid,
2px lines, fixed categorical order, status colours reserved for traffic lights.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {
    "green": "#0ca30c",
    "amber": "#fab219",
    "red": "#d03b3b",
    "stable": "#0ca30c",
    "warn": "#fab219",
    "alert": "#d03b3b",
    "insufficient": "#898781",
}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "axes.labelsize": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "text.color": INK,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


apply_style()


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def roc_pr(y, p, metrics: dict):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1)
    a1.plot(fpr, tpr, color=SERIES[0])
    a1.set_title("ROC curve", pad=18)
    _subtitle(a1, f"AUC {metrics['auc']:.3f} | Gini {metrics['gini']:.3f} | KS {metrics['ks']:.3f}")
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    prec, rec, _ = precision_recall_curve(y, p)
    a2.axhline(np.mean(y), color=AXIS, lw=1)
    a2.plot(rec, prec, color=SERIES[0])
    a2.set_title("Precision-recall curve", pad=18)
    _subtitle(a2, f"Average precision {metrics['pr_auc']:.3f} (base rate {np.mean(y):.1%})")
    a2.set_xlabel("Recall")
    a2.set_ylabel("Precision")
    fig.tight_layout()
    return fig


def calibration(tables: dict[str, pd.DataFrame], subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    hi = max(max(t["mean_pd"].max(), t["observed_dr"].max()) for t in tables.values()) * 1.05
    ax.plot([0, hi], [0, hi], color=AXIS, lw=1, label="Perfect calibration")
    for i, (name, t) in enumerate(tables.items()):
        ax.plot(t["mean_pd"], t["observed_dr"], color=SERIES[i], marker="o", ms=6, mec=SURFACE, mew=1.5, label=name)
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (reliability) by PD decile", pad=18)
    _subtitle(ax, subtitle)
    ax.legend(loc="upper left")
    return fig


def score_distribution(y, p, threshold: float):
    """KS chart: cumulative PD distribution of defaulters vs. non-defaulters."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.linspace(0, 1, 501)
    cdf_good = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / max((y == 0).sum(), 1)
    cdf_bad = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / max((y == 1).sum(), 1)
    gap = cdf_good - cdf_bad
    k = int(np.argmax(gap))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, cdf_good, color=SERIES[0], label="No default")
    ax.plot(grid, cdf_bad, color=SERIES[1], label="Default")
    ax.vlines(grid[k], cdf_bad[k], cdf_good[k], color=INK2, lw=1)
    ax.text(
        grid[k] + 0.01,
        (cdf_bad[k] + cdf_good[k]) / 2,
        f"KS {gap[k]:.3f} at PD {grid[k]:.2f}",
        color=INK2,
        fontsize=9,
        va="center",
    )
    ax.axvline(threshold, color=AXIS, lw=1)
    ax.text(threshold, 0.02, f" cut-off {threshold:.2f}", color=INK2, fontsize=9)
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Cumulative share of clients")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Separation of defaulters (KS chart)")
    ax.legend(loc="lower right")
    return fig


def cost_curve(curve: pd.DataFrame, chosen: float, cost_fn: float, cost_fp: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["expected_cost_per_client"], color=SERIES[0])
    best = curve.loc[curve["expected_cost_per_client"].idxmin()]
    ax.axvline(chosen, color=INK2, lw=1)
    ax.plot([best["threshold"]], [best["expected_cost_per_client"]], "o", color=SERIES[1], ms=8, mec=SURFACE, mew=2)
    ax.text(chosen, ax.get_ylim()[1], f" policy cut-off {chosen:.3f}", color=INK2, fontsize=9, va="top")
    ax.annotate(
        f"empirical minimum {best['threshold']:.2f}",
        (best["threshold"], best["expected_cost_per_client"]),
        textcoords="offset points",
        xytext=(14, 8),
        fontsize=9,
        color=INK2,
    )
    ax.set_xlabel("PD cut-off for intervention")
    ax.set_ylabel("Expected cost per client")
    ax.set_title("Cut-off economics on the test set", pad=18)
    _subtitle(ax, f"Cost of a missed default = {cost_fn:g}, cost of a false alarm = {cost_fp:g}")
    return fig


def deciles(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = table["decile"].to_numpy()
    ax.bar(x, table["observed_dr"], width=0.55, color=SERIES[0], label="Observed default rate")
    ax.plot(x, table["mean_pd"], "o", color=SERIES[1], ms=8, mec=SURFACE, mew=2, label="Mean predicted PD")
    top = table.iloc[0]
    ax.text(x[0], top["observed_dr"], f"{top['observed_dr']:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xlabel("Risk decile (1 = highest PD)")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Rank ordering by risk decile", pad=18)
    _subtitle(ax, f"Top 2 deciles capture {table['cum_default_capture'].iloc[1]:.0%} of all defaults")
    ax.legend()
    return fig


def backtest(bt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(bt))
    ax.plot(x, bt["mean_pd"], "o", mfc=SURFACE, mec=INK2, mew=1.5, ms=9, ls="none", label="Mean PD (model)")
    for i, row in bt.iterrows():
        ax.plot(x[i], row["observed_dr"], "o", color=STATUS[row["traffic_light"]], ms=9, mec=SURFACE, mew=2)
        ax.text(
            x[i],
            max(row["observed_dr"], row["mean_pd"]) + 0.03,
            row["traffic_light"].upper(),
            ha="center",
            fontsize=8,
            color=INK2,
        )
    ax.plot([], [], "o", color=MUTED, ms=9, label="Observed DR (coloured by Jeffreys test)")
    ax.set_xticks(x, [f"{g}\n{r}" for g, r in zip(bt["grade"], bt["pd_range"])])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Default rate")
    ax.set_title("PD backtest by rating grade", pad=18)
    _subtitle(ax, "Traffic light: Jeffreys p < 0.1% red, < 5% amber, otherwise green")
    ax.legend(loc="upper left")
    return fig


def importance(imp: pd.DataFrame, top: int = 15):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    ax.barh(t["description"], t["mean_abs_shap"], height=0.55, color=SERIES[0])
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Global feature importance (SHAP)")
    ax.grid(axis="y", visible=False)
    return fig


def fairness(table: pd.DataFrame, threshold: float):
    attrs = list(dict.fromkeys(table["attribute"]))
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs), 3.6), squeeze=False, sharex=True)
    for ax, attr in zip(axes[0], attrs):
        t = table[table["attribute"] == attr]
        t = t.iloc[sorted(range(len(t)), key=lambda i: str(t["group"].iloc[i]).replace("<", "0"), reverse=True)]
        labels = t["group"].astype(str)
        ax.barh(labels, t["flagged_rate"], height=0.35, color=SERIES[0], label="Flagged for intervention")
        ax.plot(
            t["observed_dr"],
            labels,
            "o",
            color=SERIES[1],
            ms=8,
            mec=SURFACE,
            mew=2,
            ls="none",
            label="Observed default rate",
        )
        ax.set_title(attr)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Fairness check at cut-off {threshold:.2f} - attributes are not model inputs",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def psi_bars(table: pd.DataFrame, top: int = 15, warn: float = 0.1, alert: float = 0.25):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.4))
    ax.barh(t["feature"], t["psi"], height=0.55, color=[STATUS[s] for s in t["status"]])
    for i, (v, lbl) in enumerate([(warn, "warn"), (alert, "alert")]):
        ax.axvline(v, color=INK2, lw=1)
        ax.text(v, len(t) - 0.3 - 0.9 * i, f" {lbl} {v}", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("Population stability index")
    ax.set_title("Feature drift vs. training reference")
    ax.grid(axis="y", visible=False)
    return fig


def score_shift(profile_score: dict, current_scores):
    edges = np.asarray(profile_score["edges"])
    expected = np.asarray(profile_score["expected"])
    actual = np.histogram(np.round(np.asarray(current_scores, dtype=float), 10), bins=edges)[0]
    actual = actual / max(actual.sum(), 1)
    x = np.arange(len(expected))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.15, expected, width=0.28, color=SERIES[0], label="Training reference")
    ax.bar(x + 0.15, actual, width=0.28, color=SERIES[1], label="Current batch")
    labels = [f"<{edges[i + 1]:.2f}" if i < len(x) - 1 else f">{edges[i]:.2f}" for i in x]
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("PD bin (training deciles)")
    ax.set_ylabel("Share of clients")
    ax.set_title("Score distribution shift")
    ax.legend()
    return fig


def roll_rate_heatmap(matrix: pd.DataFrame):
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    m = matrix.to_numpy()
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=9, color="#ffffff" if m[i, j] > 0.45 else INK
            )
    labels = [b.split(" ", 1)[1] for b in matrix.columns]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Next month")
    ax.set_ylabel("This month")
    ax.grid(False)
    ax.set_title("Monthly roll rates")
    return fig


def confusion(tm: dict):
    m = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(m, cmap=cmap)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", color="#ffffff" if m[i, j] > m.max() * 0.5 else INK)
    ax.set_xticks([0, 1], ["No action", "Intervene"])
    ax.set_yticks([0, 1], ["No default", "Default"])
    ax.grid(False)
    ax.set_title(f"Decisions at cut-off {tm['threshold']:.3f}")
    return fig


def shap_beeswarm(contrib: pd.DataFrame, features: pd.DataFrame, max_display: int = 15):
    try:
        import shap

        expl = shap.Explanation(
            values=contrib.to_numpy(), data=features[contrib.columns].to_numpy(), feature_names=list(contrib.columns)
        )
        shap.plots.beeswarm(expl, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_facecolor(SURFACE)
        plt.title("SHAP summary (log-odds contribution per client)", loc="left", fontsize=12)
        return fig
    except Exception:  # noqa: BLE001 - shap is only needed for this optional visual
        return None


def waterfall(contrib_row: pd.Series, feature_row: pd.Series, base_value: float, title: str, top: int = 10):
    order = contrib_row.abs().sort_values(ascending=False).index[:top]
    rest = contrib_row.drop(order).sum()
    items = [(f"{f} = {feature_row[f]:,.2f}".rstrip("0").rstrip("."), contrib_row[f]) for f in order]
    items.append((f"{len(contrib_row) - top} other features", rest))
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(items) + 1.4))
    vals = np.array([v for _, v in items])
    left = base_value + np.cumsum(vals) - vals  # bottom (smallest) to top (largest driver)
    for i, ((_label, v), l0) in enumerate(zip(items, left)):
        ax.barh(i, v, left=l0, height=0.55, color=SERIES[7] if v > 0 else SERIES[0])
        ax.text(
            l0 + v + (0.02 if v > 0 else -0.02),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize=8,
            color=INK2,
        )
    ax.set_yticks(range(len(items)), [lbl for lbl, _ in items], fontsize=8)
    ax.axvline(base_value, color=AXIS, lw=1)
    ax.set_xlabel("Log-odds of default (base value -> this client)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return fig
