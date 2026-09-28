"""Server-rendered, linked SVG views.

Every interactive element carries a stable id derived from the object it
represents (injection ordinal, material version, model name) so display
values can be traced back to the archive. Linkage (cross-highlight) is done
with a tiny vanilla-JS attribute toggle, never by recomputing rules.
"""

from __future__ import annotations

import html
from typing import Any

_COLORS = {"one_site": "#2563eb", "two_site": "#d97706", "dilution": "#059669"}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def power_svg(project: dict[str, Any], state: dict[str, Any], *, width: int = 900, height: int = 300) -> str:
    power = None
    for key, mid in state["heads"].items():
        if key == "power:main":
            power = project["materials"][mid]
    if power is None:
        return f'<svg id="chart-power" width="{width}" height="{height}"></svg>'
    pts = [p for p in power["payload"]["points"] if p.get("convertible")]
    xs = [p["time_s"] for p in pts]
    ys = [p["value"] * 1e6 for p in pts]  # display µW
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 46

    def px(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin + 1e-12) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin + 1e-12) * (height - 2 * pad)

    path = "M" + "L".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    parts = [
        f'<svg id="chart-power" class="chart" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="power curve">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        f'<path d="{path}" fill="none" stroke="#111827" stroke-width="1.2" id="power-line"/>',
    ]
    for i, gap in enumerate(power["payload"].get("gaps_s", [])):
        x0, x1 = px(gap["start"]), px(gap["end"])
        parts.append(
            f'<rect class="gap" id="gap-{i}" data-gap="{i}" x="{x0:.1f}" y="0" '
            f'width="{max(x1 - x0, 2):.1f}" height="{height}" fill="#fecaca" opacity="0.55"/>'
        )

    int_record = None
    if state["current_integration_id"]:
        int_record = project["results"][state["current_integration_id"]]["result"]
    injections = project["materials"][state["heads"]["injections:main"]] if "injections:main" in state["heads"] else None
    if injections:
        for entry in injections["payload"]["entries"]:
            if not entry.get("convertible"):
                continue
            o = entry["ordinal"]
            x = px(entry["time_s"])
            parts.append(
                f'<line class="inj-mark" data-inj="{o}" id="inj-mark-{o}" x1="{x:.1f}" '
                f'y1="{pad - 10}" x2="{x:.1f}" y2="{height - pad}" stroke="#7c3aed" stroke-dasharray="4 3"/>'
            )
    if int_record:
        rule = int_record["inputs"]["window_rule"]
        for p in int_record["per_injection"]:
            if p.get("status") != "ok":
                continue
            o = p["ordinal"]
            w0, w1 = px(p["window_s"][0]), px(p["window_s"][1])
            parts.append(
                f'<rect class="window" data-inj="{o}" id="window-{o}" x="{w0:.1f}" y="{pad - 14}" '
                f'width="{(w1 - w0):.1f}" height="{height - 2 * pad + 20}" fill="#93c5fd" opacity="0.18"/>'
            )
            net = p["net_heat_J"]
            parts.append(
                f'<circle class="int-point" data-inj="{o}" id="int-point-{o}" cx="{w1:.1f}" '
                f'cy="{pad - 4:.1f}" r="4" fill="#1d4ed8"><title>inj {o}: {net:.3e} J</title></circle>'
            )
    parts.append(
        f'<text x="{pad}" y="20" class="axis-label">power ({_esc(power["payload"]["value_unit_raw"])} vs {_esc(power["payload"]["time_unit_raw"])})</text>'
    )
    parts.append(f'<text x="{width - pad - 180}" y="20" class="basis">{_esc(power["payload"]["coordinate_basis"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def fit_svg(project: dict[str, Any], state: dict[str, Any], *, width: int = 900, height: int = 260) -> str:
    fit_id = state.get("current_fit_id")
    if not fit_id:
        return f'<svg id="chart-fit" width="{width}" height="{height}"></svg>'
    fit = project["results"][fit_id]["result"]
    residuals = fit["residuals"]
    ordinals = [r["ordinal"] for r in residuals]
    obs = [r["observed_J"] * 1e3 for r in residuals]
    ymin = min(obs + [0.0])
    ymax = max(obs + [0.0])
    pad = 46

    def px(i: int) -> float:
        return pad + i / max(len(ordinals) - 1, 1) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin + 1e-12) * (height - 2 * pad)

    parts = [
        f'<svg id="chart-fit" class="chart" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<line x1="{pad}" y1="{py(0):.1f}" x2="{width - pad}" y2="{py(0):.1f}" stroke="#9ca3af"/>',
    ]
    for i, o in enumerate(ordinals):
        parts.append(
            f'<circle class="obs-point" data-inj="{o}" id="obs-{o}" cx="{px(i):.1f}" cy="{py(obs[i]):.1f}" r="4.5" fill="#111827"/>'
        )
    for cand in fit["candidates"]:
        color = _COLORS.get(cand["model"], "#333")
        pts = [(px(i), py(v * 1e3)) for i, v in enumerate(cand["predicted_J"])]
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(
            f'<path class="fit-curve" data-model="{cand["model"]}" id="fit-curve-{cand["model"]}" '
            f'd="{d}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
    parts.append(
        f'<text x="{pad}" y="20" class="axis-label">heat per injection (mJ); lines: one-site / two-site / dilution</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def residual_svg(project: dict[str, Any], state: dict[str, Any], *, width: int = 900, height: int = 200) -> str:
    fit_id = state.get("current_fit_id")
    if not fit_id:
        return f'<svg id="chart-residual" width="{width}" height="{height}"></svg>'
    fit = project["results"][fit_id]["result"]
    residuals = fit["residuals"]
    models = [c["model"] for c in fit["candidates"]]
    all_vals = [r["per_model"][m] * 1e3 for r in residuals for m in models]
    limit = max(abs(min(all_vals)), abs(max(all_vals)), 1e-12)
    pad = 46

    def px(i: int) -> float:
        return pad + i / max(len(residuals) - 1, 1) * (width - 2 * pad)

    def py(v: float) -> float:
        return height / 2 - v / limit * (height / 2 - pad)

    parts = [
        f'<svg id="chart-residual" class="chart" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<line x1="{pad}" y1="{height / 2:.1f}" x2="{width - pad}" y2="{height / 2:.1f}" stroke="#9ca3af"/>',
    ]
    offsets = {m: (i - (len(models) - 1) / 2) * 5 for i, m in enumerate(models)}
    for i, r in enumerate(residuals):
        for m in models:
            v = r["per_model"][m] * 1e3
            color = _COLORS.get(m, "#333")
            parts.append(
                f'<line class="residual" data-inj="{r["ordinal"]}" data-model="{m}" '
                f'id="resid-{r["ordinal"]}-{m}" x1="{px(i) + offsets[m]:.1f}" y1="{height / 2:.1f}" '
                f'x2="{px(i) + offsets[m]:.1f}" y2="{py(v):.1f}" stroke="{color}" stroke-width="2"/>'
            )
    parts.append('<text x="46" y="20" class="axis-label">fit residuals (mJ), coloured by model</text>')
    parts.append("</svg>")
    return "".join(parts)
