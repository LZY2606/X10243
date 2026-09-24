"""Injection integration engine.

Domain rules implemented here (the service layer also emits them as
machine-readable ``rule_judgment`` events):

- peaks after an injection MAY overlap: windows simply abut at the next
  injection time, no separation is forced;
- an integration window NEVER crosses a data gap: it is truncated at the gap
  start and the injection is flagged ``gap_truncated``;
- window endpoints that fall between samples are linearly interpolated
  (``endpoint="interpolate"``) or snapped to the nearest sample
  (``endpoint="nearest"``); the rule used is recorded per injection;
- resource budget exhaustion is a distinct result (``budget_exhausted``),
  not a partial success.
"""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class IntegParams:
    max_window_s: float = 90.0
    gap_factor: float = 3.0
    endpoint: str = "interpolate"   # or "nearest"
    version: str = "int-1.0"

    def as_dict(self):
        return asdict(self)


def detect_gaps(t, p, gap_factor):
    """Gaps from sampling jumps; invalid (None) samples act as point gaps."""
    gaps = []
    valid = [(tt, pp) for tt, pp in zip(t, p) if tt is not None and pp is not None]
    for i, (tt, pp) in enumerate(zip(t, p)):
        if tt is None or pp is None:
            pos = tt if tt is not None else (valid[0][0] if valid else 0.0)
            gaps.append((pos, pos))
    dts = [valid[i + 1][0] - valid[i][0] for i in range(len(valid) - 1)]
    if not dts:
        return gaps
    med = sorted(dts)[len(dts) // 2]
    for i, dt in enumerate(dts):
        if dt > gap_factor * med:
            gaps.append((valid[i][0], valid[i + 1][0]))
    return sorted(gaps)


def _value_at(t, p, x, mode="interpolate"):
    if x <= t[0]:
        return p[0]
    if x >= t[-1]:
        return p[-1]
    lo, hi = 0, len(t) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= x:
            lo = mid
        else:
            hi = mid
    if mode == "nearest":
        return p[lo] if (x - t[lo]) <= (t[hi] - x) else p[hi]
    t0, t1 = t[lo], t[hi]
    if t1 == t0:
        return p[lo]
    f = (x - t0) / (t1 - t0)
    return p[lo] * (1 - f) + p[hi] * f


def baseline_at(base, x):
    if base["mode"] == "constant":
        return base["c"]
    return _value_at(base["t"], base["p"], x)


def _incomplete(path, code, message, params):
    return {"status": "incomplete_input", "param_version": params.version,
            "diagnostics": [{"path": path, "code": code, "message": message}],
            "injections": []}


def integrate(power, schedule, baseline, params, budget=None):
    t_all, p_all = power["t"], power["p"]
    valid = sorted((tt, pp) for tt, pp in zip(t_all, p_all)
                   if tt is not None and pp is not None)
    if len(valid) < 2:
        return _incomplete("power_trace", "insufficient_samples",
                           "fewer than 2 valid power samples", params)
    t = [a for a, _ in valid]
    p = [b for _, b in valid]
    gaps = detect_gaps(t_all, p_all, params.gap_factor)
    gaps = sorted(set(gaps) | {(g[0], g[1]) for g in power.get("gaps", [])})

    inj = list(zip(range(len(schedule["t"])), schedule["t"], schedule["v_uL"]))
    bad = [i for i, ti, _ in inj if ti is None]
    if bad:
        return _incomplete(f"injection_schedule.time[{bad[0]}]", "value_not_numeric",
                           "injection time could not be converted", params)
    inj.sort(key=lambda x: (x[1], x[0]))  # stable order for equivalent inputs

    ops = 0
    out = []
    for k, (idx, t0, vol) in enumerate(inj):
        t1 = t0 + params.max_window_s
        if k + 1 < len(inj):
            t1 = min(t1, inj[k + 1][1])
        status = "ok"
        di = []
        gap_limited = False
        for (g0, g1) in gaps:
            if g0 <= t0 < g1:
                status = "in_gap"
                di.append({"path": f"injections[{idx}]", "code": "injection_in_gap",
                           "message": f"injection at {t0} s falls inside data gap [{g0}, {g1}]"})
                break
            if t0 < g0 < t1:
                t1 = g0
                gap_limited = True
                status = "gap_truncated"
                di.append({"path": f"injections[{idx}].window", "code": "gap_truncated",
                           "message": f"window truncated at data gap start {g0} s"})
        if status == "in_gap":
            out.append({"index": idx, "order": k, "t_inject": t0, "volume_uL": vol,
                        "window": [t0, t0], "n_points": 0, "heat_uJ": None,
                        "status": status, "endpoint": params.endpoint,
                        "gap_limited": False, "diagnostics": di})
            continue
        xs = [t0] + [tt for tt in t if t0 < tt < t1] + [t1]
        ops += len(xs) - 1
        if budget is not None and ops > budget:
            return {"status": "budget_exhausted", "param_version": params.version,
                    "diagnostics": [{"path": "budget", "code": "budget_exhausted",
                                     "message": f"operation budget {budget} exhausted after {ops} ops"}],
                    "injections": []}
        ys = [_value_at(t, p, x, params.endpoint) for x in xs]
        heat = 0.0
        for a in range(len(xs) - 1):
            b0 = ys[a] - baseline_at(baseline, xs[a])
            b1 = ys[a + 1] - baseline_at(baseline, xs[a + 1])
            heat += 0.5 * (b0 + b1) * (xs[a + 1] - xs[a])
        out.append({"index": idx, "order": k, "t_inject": t0, "volume_uL": vol,
                    "window": [t0, t1], "n_points": len(xs), "heat_uJ": heat,
                    "status": status, "endpoint": params.endpoint,
                    "gap_limited": gap_limited, "diagnostics": di})
    return {"status": "ok", "param_version": params.version, "injections": out,
            "diagnostics": [], "gaps": [list(g) for g in gaps]}
