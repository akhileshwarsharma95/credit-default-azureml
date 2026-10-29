"""Build docs/presentation.html: one self-contained file (React, icons and charts inlined) that works offline.

    python docs/presentation/build.py

Sources: app.js (React via htm, no build tooling), styles.css, assets/ (official Azure architecture icons,
brand marks), vendor/ (React 18.3.1, htm 3.1.1, React Flow @xyflow/react 12.3.6 UMD builds) and the charts
in docs/images/.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
OUT = DOCS / "presentation.html"

# Official brand colours (simple-icons); the marks ship as single-path SVGs without a fill
BRAND_COLORS = {
    "github": "#181717",
    "githubactions": "#2088FF",
    "python": "#3776AB",
    "mlflow": "#0194E2",
    "streamlit": "#FF4B4B",
    "duckdb": "#000000",
    "scikitlearn": "#F7931E",
    "docker": "#2496ED",
    "opentelemetry": "#425CC7",
    "pandas": "#150458",
    "numpy": "#013243",
    "openai": "#412991",
}
IMAGES = [
    "backtest",
    "calibration",
    "cost_curve",
    "deciles",
    "fairness",
    "ks_chart",
    "psi_downturn",
    "roll_rates",
    "shap_beeswarm",
    "waterfall_high_risk",
    "app_portfolio",
    "app_client",
    "app_policy",
    "app_monitoring",
    "app_agent",
]


def data_uri(path: Path, mime: str, text: str | None = None) -> str:
    raw = text.encode() if text is not None else path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def svg_uri(path: Path, fill: str | None = None) -> str:
    svg = path.read_text()
    if re.search(r"<script|\son[a-z]+\s*=|javascript:", svg, re.I):
        raise ValueError(f"refusing to inline active content from {path}")
    if fill:
        svg = svg.replace("<svg ", f'<svg fill="{fill}" ', 1)
    return data_uri(path, "image/svg+xml", svg)


def main() -> None:
    icons = {f"az/{p.stem}": svg_uri(p) for p in sorted((HERE / "assets" / "azure").glob("*.svg"))}
    for p in sorted((HERE / "assets" / "brand").iterdir()):
        if p.suffix == ".svg":
            icons[p.stem] = svg_uri(p, BRAND_COLORS.get(p.stem))
        elif p.suffix == ".png":
            icons[p.stem] = data_uri(p, "image/png")
    images = {name: data_uri(DOCS / "images" / f"{name}.png", "image/png") for name in IMAGES}

    # React Flow's UMD build expects a react/jsx-runtime global; map it onto React.createElement
    jsx_shim = (
        "window.jsxRuntime = (function () {"
        " var jsx = function (t, p, k) { return React.createElement(t, k === undefined ? p : Object.assign({}, p, { key: k })); };"
        " return { jsx: jsx, jsxs: jsx, Fragment: React.Fragment }; })();"
    )
    vendor = "\n".join(
        [
            (HERE / "vendor" / f).read_text()
            for f in ("react.production.min.js", "react-dom.production.min.js", "htm.umd.js")
        ]
        + [jsx_shim, (HERE / "vendor" / "xyflow-react.umd.js").read_text()]
    )
    html = (HERE / "template.html").read_text()
    html = html.replace(
        "/*STYLES*/", (HERE / "vendor" / "xyflow-react.css").read_text() + "\n" + (HERE / "styles.css").read_text()
    )
    html = html.replace("/*VENDOR*/", vendor.replace("</script", "<\\/script"))
    html = html.replace("/*ASSETS*/", f"window.ICONS = {json.dumps(icons)};\nwindow.IMAGES = {json.dumps(images)};")
    html = html.replace("/*APP*/", (HERE / "app.js").read_text())
    html = html.replace("/*FAVICON*/", icons["dkv-mobility"])
    OUT.write_text(html)
    print(
        f"Wrote {OUT.relative_to(DOCS.parent)} ({OUT.stat().st_size / 1e6:.1f} MB, {len(icons)} icons, {len(images)} images)"
    )


if __name__ == "__main__":
    main()
