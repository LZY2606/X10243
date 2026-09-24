"""Server-rendered SVG panels with stable element identities.

Each drawable carries the same stable id exposed in the JSON view, so the
linked page (curve / injection windows / integral points / residuals) can
cross-highlight by id without recomputing anything in the browser.
"""

from __future__ import annotations

import math

import numpy as np

W, H = 900, 320
ML, MR, MT, MB = 64, 16, 18, 38

PALETTE = {
    "power": "#1f4e79", "baseline": "#8a8f99", "blank": "#7e57c2",
    "window": "#f1c40f", "window_gap": "#e67e22", "point": "#c0392b",
    "residual": "#16a085", "fit": "#2e8b57", "grid": "#e3e6ea",
}


def _scale(xs, ys, w=W, h=H):
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        y1 = y0 + abs(y0) * 0.1 + 1e-12

    def sx(x):
        return ML + (x - x0) / (x1 - x0) * (w - ML - MR)

    def sy(y):
        return h - MB - (y - y0) / (y1 - y0) * (h - MT - MB)

    return sx, sy, (x0, x1, y0, y1)


def _axes(sx, sy, limits, *, xlabel: str, ylabel: str, w=W, h=H,
          xfmt="{:.0f}", yfmt="{:.3g}") -> list[str]:
    x0, x1, y0, y1 = limits
    out = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>']
    for i in range(5):
        gx = x0 + (x1 - x0) * i / 4
        gy = y0 + (y1 - y0) * i / 4
        out.append(f'<line x1="{sx(gx):.1f}" y1="{MT}" x2="{sx(gx):.1f}" y2="{h-MB}"'
                   f' stroke="{PALETTE["grid"]}"/>')
        out.append(f'<line x1="{ML}" y1="{sy(gy):.1f}" x2="{w-MR}" y2="{sy(gy):.1f}"'
                   f' stroke="{PALETTE["grid"]}"/>')
        out.append(f'<text x="{sx(gx):.1f}" y="{h-MB+16}" font-size="11" '
                   f'text-anchor="middle" fill="#555">{xfmt.format(gx)}</text>')
        out.append(f'<text x="{ML-8}" y="{sy(gy)+4:.1f}" font-size="11" '
                   f'text-anchor="end" fill="#555">{yfmt.format(gy)}</text>')
    out.append(f'<text x="{(ML+w-MR)/2}" y="{h-4}" font-size="12" '
               f'text-anchor="middle" fill="#222">{xlabel}</text>')
    out.append(f'<text x="14" y="{(MT+h-MB)/2}" font-size="12" fill="#222" '
               f'transform="rotate(-90 14 {(MT+h-MB)/2})" text-anchor="middle">{ylabel}</text>')
    return out


def _path(t, v, sx, sy, max_points=1800) -> str:
    if t.size > max_points:
        idx = np.linspace(0, t.size - 1, max_points).astype(int)
        t, v = t[idx], v[idx]
    return "M " + " L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(t, v))


def power_svg(*, power_t, power_v, baseline_t=None, baseline_v=None,
              shots: list[dict], svg_id: str = "panel-power") -> str:
    parts = [f'<svg id="{svg_id}" viewBox="0 0 {W} {H}" class="panel" '
             f'role="img" aria-label="功率曲线与注射区间">']
    all_t = [power_t] + ([baseline_t] if baseline_t is not None else [])
    all_v = [power_v] + ([baseline_v] if baseline_v is not None else [])
    tcat = np.concatenate(all_t)
    vcat = np.concatenate(all_v)
    sx, sy, lim = _scale(tcat, vcat)
    parts += _axes(sx, sy, lim, xlabel="实验时间 (s, experiment_seconds)",
                   ylabel="功率 (W)", yfmt="{:.2e}")
    # injection windows
    for s in shots:
        win = s["window"]
        color = PALETTE["window_gap"] if s.get("gap_terminated") else PALETTE["window"]
        parts.append(
            f'<rect class="hit window" data-id="shot-{s.get("integration_event_id","")}-{s["index"]}"'
            f' data-shot="{s["index"]}" x="{sx(win["start_s"]):.1f}" '
            f'y="{MT}" width="{max(sx(win["end_s"])-sx(win["start_s"]),1):.1f}" '
            f'height="{H-MT-MB}" fill="{color}" fill-opacity="0.16"/>')
        parts.append(
            f'<line x1="{sx(s["injection_time_s"]):.1f}" y1="{MT}" '
            f'x2="{sx(s["injection_time_s"]):.1f}" y2="{H-MB}" '
            f'stroke="#b7950b" stroke-dasharray="3 3"/>')
    parts.append(f'<path d="{_path(power_t, power_v, sx, sy)}" fill="none" '
                 f'stroke="{PALETTE["power"]}" stroke-width="1.2"/>')
    if baseline_t is not None:
        parts.append(f'<path d="{_path(baseline_t, baseline_v, sx, sy)}" '
                     f'fill="none" stroke="{PALETTE["baseline"]}" stroke-width="1" '
                     f'stroke-dasharray="5 3"/>')
    parts.append("</svg>")
    return "".join(parts)


def integral_svg(shots: list[dict], *, integration_event_id: str,
                 svg_id: str = "panel-integrals") -> str:
    usable = [s for s in shots]
    x = np.array([s["index"] for s in usable], dtype=float)
    y = np.array([s["net_j"] for s in usable], dtype=float)
    parts = [f'<svg id="{svg_id}" viewBox="0 0 {W} {H}" class="panel" '
             f'role="img" aria-label="每次注射积分点">']
    sx, sy, lim = _scale(np.r_[x - 0.5, x + 0.5], y)
    parts += _axes(sx, sy, lim, xlabel="注射序号 (shot index)",
                   ylabel="积分热 (J)", yfmt="{:.2e}")
    for s in shots:
        xi, yi = sx(s["index"]), sy(s["net_j"])
        gap = s.get("gap_terminated")
        missing = s.get("blank_status") == "missing"
        color = PALETTE["window_gap"] if (gap or missing) else PALETTE["point"]
        parts.append(
            f'<circle class="hit integral-point" data-id="shot-{integration_event_id}-{s["index"]}"'
            f' data-shot="{s["index"]}" cx="{xi:.1f}" cy="{yi:.1f}" r="6" '
            f'fill="{color}" fill-opacity="0.9"/>')
        parts.append(
            f'<text x="{xi:.1f}" y="{yi-10:.1f}" font-size="10" text-anchor="middle" '
            f'fill="#444">{s["index"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def residual_svg(candidate: dict, shot_indices: list[int], *,
                 candidate_id: str, svg_id: str = "panel-residuals") -> str:
    residual = np.array(candidate.get("residual", []), dtype=float)
    x = np.array(shot_indices[: residual.size], dtype=float)
    parts = [f'<svg id="{svg_id}" viewBox="0 0 {W} {H}" class="panel" '
             f'role="img" aria-label="拟合残差">']
    if residual.size == 0:
        parts.append(f'<text x="{W/2}" y="{H/2}" text-anchor="middle">无残差数据</text></svg>')
        return "".join(parts)
    ymin, ymax = float(residual.min()), float(residual.max())
    pad = max(abs(ymin), abs(ymax), 1e-12)
    sx, sy, lim = _scale(np.r_[x - 0.3, x + 0.3],
                         np.array([-pad, pad]))
    parts += _axes(sx, sy, lim, xlabel="注射序号 (shot index)",
                   ylabel="残差 (J)", yfmt="{:.2e}")
    parts.append(f'<line x1="{ML}" y1="{sy(0):.1f}" x2="{W-MR}" y2="{sy(0):.1f}" '
                 f'stroke="#999"/>')
    for xi, ri in zip(x, residual):
        parts.append(
            f'<line class="hit residual-bar" data-candidate="{candidate_id}" '
            f'data-shot="{int(xi)}" x1="{sx(xi):.1f}" x2="{sx(xi):.1f}" '
            f'y1="{sy(0):.1f}" y2="{sy(ri):.1f}" stroke="{PALETTE["residual"]}" '
            f'stroke-width="3"/>')
    parts.append("</svg>")
    return "".join(parts)
