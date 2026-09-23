"""服务端 SVG 渲染: 功率曲线(注射区间/积分点/数据缺口) 与拟合残差。

每个图形元素带 data-oid / data-inj 稳定身份, 供页面联动。
"""
from __future__ import annotations

from .integrate import normalize_power

W, H = 760, 260
PAD_L, PAD_R, PAD_T, PAD_B = 56, 12, 18, 30


def _fmt(v):
    return f"{v:.3g}"


def power_curve_svg(power_payload, integrals_payload=None, highlight=None):
    samples, gaps, _ = normalize_power(power_payload)
    if not samples:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    ts = [s[0] for s in samples]
    ps = [s[1] for s in samples]
    t0, t1 = min(ts), max(ts)
    p0, p1 = min(ps), max(ps)
    span_t = max(t1 - t0, 1e-9)
    span_p = max(p1 - p0, 1e-9)

    def x(t):
        return PAD_L + (t - t0) / span_t * (W - PAD_L - PAD_R)

    def y(p):
        return PAD_T + (1 - (p - p0) / span_p) * (H - PAD_T - PAD_B)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' "
        f"class='power-curve' role='img'>",
        f"<text x='{PAD_L}' y='12' class='axis-label'>功率 (µJ/s) × 时间 (s, "
        f"基准 {power_payload.get('basis',{}).get('time','elapsed_s_from_run_start')})</text>",
        f"<line x1='{PAD_L}' y1='{H-PAD_B}' x2='{W-PAD_R}' y2='{H-PAD_B}' class='axis'/>",
        f"<line x1='{PAD_L}' y1='{PAD_T}' x2='{PAD_L}' y2='{H-PAD_B}' class='axis'/>",
        f"<text x='4' y='{y(p1)}' class='tick'>{_fmt(p1)}</text>",
        f"<text x='4' y='{y(p0)}' class='tick'>{_fmt(p0)}</text>",
        f"<text x='{PAD_L}' y='{H-8}' class='tick'>{_fmt(t0)}</text>",
        f"<text x='{W-PAD_R-30}' y='{H-8}' class='tick'>{_fmt(t1)}</text>",
    ]
    # 数据缺口(阴影)
    for gi, (gs, ge, reason) in enumerate(gaps):
        parts.append(
            f"<rect x='{x(gs):.1f}' y='{PAD_T}' width='{max(x(ge)-x(gs),1.5):.1f}' "
            f"height='{H-PAD_T-PAD_B}' class='gap' data-oid='gap-{gi}'>"
            f"<title>数据缺口 [{gs:.0f}, {ge:.0f}] s {reason}</title></rect>")
    # 注射积分窗
    if integrals_payload:
        for inj in integrals_payload.get("injections", []):
            if not inj.get("window_s"):
                continue
            w0, w1 = inj["window_s"]
            cls = "inj-window" + (" highlight" if inj["injection_id"] == highlight else "")
            if inj["status"] != "ok":
                cls += " inj-error"
            flags = ",".join(inj.get("flags", []))
            parts.append(
                f"<rect x='{x(w0):.1f}' y='{PAD_T}' "
                f"width='{max(x(w1)-x(w0),1.5):.1f}' height='{H-PAD_T-PAD_B}' "
                f"class='{cls}' data-inj='{inj['injection_id']}'>"
                f"<title>{inj['injection_id']} 窗 [{w0:.0f},{w1:.0f}]s "
                f"热={_fmt(inj['heat_uJ']) if inj['heat_uJ'] is not None else '不可用'} µJ "
                f"{flags}</title></rect>")
    # 功率轨迹
    pts = " ".join(f"{x(t):.1f},{y(p):.1f}" for t, p in zip(ts[::2], ps[::2]))
    parts.append(f"<polyline points='{pts}' class='trace'/>")
    # 注射时刻标记
    if integrals_payload:
        for inj in integrals_payload.get("injections", []):
            parts.append(
                f"<line x1='{x(inj['t_s']):.1f}' y1='{PAD_T}' "
                f"x2='{x(inj['t_s']):.1f}' y2='{H-PAD_B}' class='inj-line' "
                f"data-inj='{inj['injection_id']}'/>")
    parts.append("</svg>")
    return "".join(parts)


def residuals_svg(fit_payload, candidate_idx=0, highlight=None):
    cands = fit_payload.get("candidates", [])
    if not cands or candidate_idx >= len(cands):
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    cand = cands[candidate_idx]
    res = cand["residuals_uJ"]
    ids = cand["injection_ids"]
    rmax = max([abs(r) for r in res] + [1e-9])
    n = len(res)
    bw = (W - PAD_L - PAD_R) / max(n, 1)
    y0 = PAD_T + (H - PAD_T - PAD_B) / 2

    def bh(r):
        return r / rmax * (H - PAD_T - PAD_B) / 2

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' "
        f"class='residuals' role='img'>",
        f"<text x='{PAD_L}' y='12' class='axis-label'>拟合残差 (µJ) — "
        f"{fit_payload.get('model')} 候选 {cand['candidate_id']}</text>",
        f"<line x1='{PAD_L}' y1='{y0}' x2='{W-PAD_R}' y2='{y0}' class='axis'/>",
        f"<text x='4' y='{PAD_T+8}' class='tick'>{_fmt(rmax)}</text>",
        f"<text x='4' y='{H-PAD_B}' class='tick'>-{_fmt(rmax)}</text>",
    ]
    for i, (rid, r) in enumerate(zip(ids, res)):
        h = bh(r)
        cls = "res-bar" + (" highlight" if rid == highlight else "")
        parts.append(
            f"<rect x='{PAD_L + i*bw + 1:.1f}' y='{min(y0, y0-h):.1f}' "
            f"width='{max(bw-2,1):.1f}' height='{abs(h):.1f}' class='{cls}' "
            f"data-inj='{rid}'><title>{rid} 残差 {r:.4g} µJ</title></rect>")
    parts.append("</svg>")
    return "".join(parts)
