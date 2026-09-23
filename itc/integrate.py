"""Injection integration over a power trace.

Hard rules implemented here (as structured state + machine-readable events,
not just logs):
- post-injection peaks may overlap: the shared boundary is the local minimum
  between the two peaks, and a judgment.overlap_boundary event is emitted;
- an integration window never crosses a data gap: it is clamped at the last
  valid sample before the gap, and a judgment.gap_clamp event is emitted;
- every window records its endpoints and the basis for each endpoint;
- the scan runs under an operation budget; exhausting it yields the distinct
  result status budget_exhausted with the position of exhaustion.
"""
from __future__ import annotations

import math

from .results import Result, OK, DIAGNOSTICS, BUDGET_EXHAUSTED, SEMANTIC_UNSOLVABLE
from .units import Diagnostic

PARAM_VERSION = "integrate-v1"
DEFAULT_PARAMS = {
    "threshold_fraction": 0.015,
    "threshold_floor_uJ_s": 0.01,
    "settle_samples": 8,
    "overlap_boundary": "local_minimum",
    "gap_policy": "clamp",
    "budget": 200000,
}


def _trapz(ts, ds):
    return float(sum((ds[i] + ds[i + 1]) / 2.0 * (ts[i + 1] - ts[i]) for i in range(len(ts) - 1)))


def integrate(power, injections, baseline, params=None):
    """Returns (Result, events). Result.value['heats'] is per-injection."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    diags = []
    events = []
    for what, norm in (("power_trace", power), ("injections", injections), ("baseline", baseline)):
        if norm.get("units_unknown"):
            return Result(SEMANTIC_UNSOLVABLE, None, [Diagnostic(
                "units_unknown",
                f"{what} has unknown/missing units; refusing to integrate unnormalizable values",
                what, {})]), events

    samples = power["samples"]
    ts = [s["t"] for s in samples]
    ps = [s["p"] for s in samples]
    valid = [s["valid"] for s in samples]
    n = len(samples)
    a = baseline["a_uJ_per_s"]
    b = baseline["b_uJ_per_s2"]
    gaps = power["gaps"]

    def gap_at(t):
        for g in gaps:
            if g["start_t"] < t < g["end_t"]:
                return g
        return None

    inj_valid = sorted((i for i in injections["injections"] if i["valid"]),
                       key=lambda r: (r["time"], r["line"]))
    heats = []
    budget = p["budget"]

    for k, inj in enumerate(inj_valid):
        tk = inj["time"]
        gap = gap_at(tk)
        if gap is not None:
            diags.append(Diagnostic(
                "injection_in_gap",
                f"injection {inj['id']} at {tk}s falls inside a data gap",
                inj["id"], {"gap": gap}))
            heats.append({
                "injection_id": inj["id"], "time": tk, "volume_uL": inj["volume_uL"],
                "status": "in_gap", "heat_uJ": None, "window": None, "flags": ["in_gap"],
            })
            continue
        si = None
        for idx in range(n):
            if valid[idx] and ts[idx] >= tk:
                si = idx
                break
        if si is None:
            diags.append(Diagnostic("injection_after_data",
                                    f"injection {inj['id']} at {tk}s is past the end of the trace",
                                    inj["id"], {}))
            heats.append({
                "injection_id": inj["id"], "time": tk, "volume_uL": inj["volume_uL"],
                "status": "no_data", "heat_uJ": None, "window": None, "flags": ["no_data"],
            })
            continue

        peak = 0.0
        idx = si
        while idx < n and valid[idx] and ts[idx] <= tk + 90.0:
            peak = max(peak, abs(ps[idx] - (a + b * ts[idx])))
            idx += 1
        thr = p["threshold_fraction"] * max(peak, 1e-9) + p["threshold_floor_uJ_s"]

        t_next = inj_valid[k + 1]["time"] if k + 1 < len(inj_valid) else math.inf
        settle = p["settle_samples"]
        run = 0
        end_idx = None
        end_basis = None
        flags = []
        i = si
        while i < n:
            budget -= 1
            if budget < 0:
                diags.append(Diagnostic("budget_exhausted",
                                        "integration budget exhausted mid-scan",
                                        inj["id"], {"time": ts[i], "param_version": PARAM_VERSION}))
                return Result(BUDGET_EXHAUSTED, {"heats": heats, "param_version": PARAM_VERSION}, diags), events
            if not valid[i]:
                end_idx = i - 1
                end_basis = "gap_clamp"
                flags.append("gap_clamp")
                gap = next((g for g in gaps if g["start_index"] <= i <= g["end_index"]), None)
                events.append({"kind": "judgment.gap_clamp", "payload": {
                    "injection_id": inj["id"],
                    "gap": gap,
                    "clamped_end": ts[end_idx],
                    "rule": "integration window must not cross a data gap",
                }})
                break
            if ts[i] >= t_next:
                seg = [(ps[j] - (a + b * ts[j]), j) for j in range(si, i + 1) if valid[j]]
                mi = min(seg)[1] if seg else i
                end_idx = mi
                end_basis = "local_minimum"
                flags.append("overlap")
                events.append({"kind": "judgment.overlap_boundary", "payload": {
                    "injection_id": inj["id"],
                    "neighbor_id": inj_valid[k + 1]["id"],
                    "boundary_time": ts[mi],
                    "rule": "local_minimum",
                }})
                break
            d = ps[i] - (a + b * ts[i])
            if abs(d) <= thr:
                run += 1
            else:
                run = 0
            if run >= settle:
                end_idx = i - settle + 1
                end_basis = "baseline_return"
                break
            i += 1
        if end_idx is None:
            end_idx = n - 1
            end_basis = "data_end"
        if end_idx <= si:
            end_idx = min(si + 1, n - 1)

        wts = [ts[j] for j in range(si, end_idx + 1)]
        wds = [ps[j] - (a + b * ts[j]) for j in range(si, end_idx + 1)]
        heat = _trapz(wts, wds)
        heats.append({
            "injection_id": inj["id"], "time": tk, "volume_uL": inj["volume_uL"],
            "status": "ok", "heat_uJ": heat,
            "window": {
                "start_time": ts[si], "end_time": ts[end_idx],
                "start_basis": "injection_time", "end_basis": end_basis,
            },
            "flags": flags, "n_samples": end_idx - si + 1,
        })

    value = {
        "heats": heats,
        "param_version": PARAM_VERSION,
        "params": p,
        "units": {"heat": "uJ", "time": "s", "volume": "uL"},
        "coordinate_basis": power["coordinate_basis"],
    }
    return Result(DIAGNOSTICS if diags else OK, value, diags), events
