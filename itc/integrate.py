"""Per-injection integration of the ITC power curve.

Rules encoded here (machine-checkable):
  * post-injection peaks MAY overlap: a window may extend past the next
    injection's start;
  * an integration window must NOT cross a data gap: it is clipped at the gap
    edge and the injection is flagged partial (or invalid if the gap swallows
    the injection start);
  * integration endpoints are reviewer-adjustable through endpoint overrides.
"""
from __future__ import annotations

from .domain import BudgetExhausted

DEFAULT_TAU_S = 25.0
WINDOW_SPAN_TAU = 5.0   # window = [t0 - PRE_S, t0 + SPAN*tau]
PRE_S = 2.0


def default_window(t0, tau=DEFAULT_TAU_S):
    return (t0 - PRE_S, t0 + WINDOW_SPAN_TAU * tau)


def integrate_injections(t, power, baseline, injections, gaps,
                         endpoint_overrides=None, tau=DEFAULT_TAU_S,
                         budget=2_000_000):
    """Integrate (power - baseline) over each injection window.

    injections: list of {"index", "time_s", "volume_uL", "injection_id"}
    gaps: list of [start_s, end_s)
    Returns (results, diagnostics). Raises BudgetExhausted.
    """
    overrides = endpoint_overrides or {}
    used = 0
    results, diags = [], []
    for inj in injections:
        idx = inj["index"]
        iid = inj["injection_id"]
        start, end = overrides.get(iid, default_window(inj["time_s"], tau))
        flags = []
        status = "ok"
        # clip against data gaps: window must not cross a gap
        for g0, g1 in sorted(gaps):
            if g0 <= start < g1 or g0 <= inj["time_s"] < g1:
                status = "invalid"
                flags.append("injection_in_gap")
                diags.append({
                    "severity": "error", "code": "GAP_CROSSES_WINDOW",
                    "message": f"injection {idx} starts inside data gap "
                               f"[{g0},{g1}); window cannot cross a data gap",
                    "location": {"injection_id": iid, "gap": [g0, g1]},
                    "entity_ids": [iid]})
                break
            if start < g0 < end:
                end = g0
                status = "partial"
                flags.append("clipped_at_gap")
                diags.append({
                    "severity": "warning", "code": "GAP_CROSSES_WINDOW",
                    "message": f"injection {idx} window clipped at gap edge "
                               f"{g0} s (windows must not cross data gaps)",
                    "location": {"injection_id": iid, "gap": [g0, g1]},
                    "entity_ids": [iid]})
        # overlap with the next injection is allowed; record it for review
        nxt = injections[idx + 1] if idx + 1 < len(injections) else None
        if nxt is not None and end > nxt["time_s"]:
            flags.append("overlaps_next_peak")
        heat = None
        if status != "invalid":
            acc = 0.0
            n = len(t)
            for k in range(n - 1):
                used += 1
                if used > budget:
                    raise BudgetExhausted(
                        "integration sample budget exhausted", used, budget)
                t0, t1 = t[k], t[k + 1]
                if t1 < start or t0 > end:
                    continue
                a = max(t0, start)
                b = min(t1, end)
                if b <= a:
                    continue
                pa = _interp(t[k], power[k], t[k + 1], power[k + 1], a) \
                    - _interp(t[k], baseline[k], t[k + 1], baseline[k + 1], a)
                pb = _interp(t[k], power[k], t[k + 1], power[k + 1], b) \
                    - _interp(t[k], baseline[k], t[k + 1], baseline[k + 1], b)
                acc += 0.5 * (pa + pb) * (b - a)
            heat = acc
        results.append({
            "injection_id": iid, "index": idx,
            "window": [round(start, 3), round(end, 3)],
            "heat_uJ": None if heat is None else round(heat, 6),
            "status": status, "flags": flags,
        })
    return results, diags


def _interp(t0, y0, t1, y1, x):
    if t1 == t0:
        return y0
    return y0 + (y1 - y0) * (x - t0) / (t1 - t0)
