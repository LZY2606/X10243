"""Per-injection integration engine.

Rules enforced here (machine-checkable):
  * post-injection peaks MAY overlap; overlap is flagged, never an error
  * an integration window NEVER crosses a data gap; it is truncated at the
    gap edge and the injection is marked partial with a positioned diagnostic
  * dilution-heat subtraction REQUIRES an explicit blank-correspondence rule;
    missing correspondences are diagnostics, not zero-filled defaults
"""
from dataclasses import dataclass, field, asdict

import numpy as np

PARAM_VERSION = "integrate/1.0"


@dataclass
class IntegrateParams:
    max_window_s: float = 90.0
    gap_factor: float = 3.0
    overlap_tol_uJ_s: float = 0.05
    param_version: str = PARAM_VERSION


def find_gaps(times, factor=3.0):
    """Data gaps: sampling intervals larger than factor * median dt.

    Returns list of {"start_s", "end_s"} (coordinates in seconds).
    """
    t = np.asarray(times, dtype=float)
    if len(t) < 3:
        return []
    dt = np.diff(t)
    med = np.median(dt)
    gaps = []
    for i, d in enumerate(dt):
        if d > factor * med:
            gaps.append({"start_s": float(t[i]), "end_s": float(t[i + 1])})
    return gaps


def _baseline_at(baseline, t):
    """baseline: {"mode": "constant", "value_uJ_s": x} or
    {"mode": "curve", "time_s": [...], "power_uJ_s": [...]}"""
    if baseline["mode"] == "constant":
        return np.full_like(np.asarray(t, dtype=float), float(baseline["value_uJ_s"]))
    return np.interp(t, baseline["time_s"], baseline["power_uJ_s"])


def integrate(times, powers, injections, baseline, params=None, budget=None):
    """Integrate each injection peak against the baseline.

    Returns (payload, diagnostics, status). status is 'ok' even when some
    injections are partial/unintegrable; those carry per-injection
    diagnostics. 'budget_exhausted' if the evaluation budget ran out.
    """
    params = params or IntegrateParams()
    t = np.asarray(times, dtype=float)
    p = np.asarray(powers, dtype=float)
    gaps = find_gaps(t, params.gap_factor)
    base = _baseline_at(baseline, t)
    results = []
    diags = []
    evals = 0

    for inj in injections:
        evals += 1
        if budget is not None and evals > budget:
            return ({"injections": results, "gaps": gaps},
                    diags + [{"code": "budget_exhausted",
                              "path": f"injections[{inj['index']}]",
                              "message": "integration budget exhausted",
                              "severity": "error"}],
                    "budget_exhausted")
        idx = inj["index"]
        t0 = float(inj["time_s"])
        nxt = [j for j in injections if float(j["time_s"]) > t0]
        t_next = min((float(j["time_s"]) for j in nxt), default=t0 + params.max_window_s)
        t1 = min(t_next, t0 + params.max_window_s)
        flags = list(inj.get("flags", []))
        if inj.get("first"):
            flags.append("first_injection")
        inj_diags = []

        # gap rules: window must not cross a gap
        inside_gap = next((g for g in gaps if g["start_s"] < t0 < g["end_s"]), None)
        if inside_gap:
            results.append({
                "index": idx, "window_s": [t0, t0], "heat_uJ": None,
                "status": "unintegrable", "flags": flags + ["inside_data_gap"],
            })
            diags.append({
                "code": "injection_inside_gap",
                "path": f"injections[{idx}]",
                "message": f"injection at {t0}s falls inside data gap "
                           f"[{inside_gap['start_s']}, {inside_gap['end_s']}]",
                "severity": "error"})
            continue
        crossing = next((g for g in gaps if t0 < g["start_s"] < t1), None)
        status = "ok"
        if crossing:
            t1 = crossing["start_s"]  # truncate at gap edge, never cross
            flags.append("gap_truncated")
            status = "partial"
            inj_diags.append({
                "code": "window_crosses_gap",
                "path": f"injections[{idx}].window_s",
                "message": f"integration window truncated at gap start "
                           f"{crossing['start_s']}s (gap to {crossing['end_s']}s)",
                "severity": "warning"})

        mask = (t >= t0) & (t <= t1)
        if mask.sum() < 3:
            results.append({
                "index": idx, "window_s": [t0, t1], "heat_uJ": None,
                "status": "unintegrable", "flags": flags + ["too_few_samples"],
            })
            diags.append({
                "code": "too_few_samples",
                "path": f"injections[{idx}]",
                "message": f"fewer than 3 samples in window [{t0}, {t1}]",
                "severity": "error"})
            continue

        heat = float(np.trapezoid((p - base)[mask], t[mask]))

        # overlap: power has not returned to baseline before window end
        after = mask & (np.abs(p - base) <= params.overlap_tol_uJ_s)
        returned = bool(after.any()) and float(t[after][-1]) < t1 - 1e-9
        if not returned:
            flags.append("overlap")
        results.append({
            "index": idx, "window_s": [t0, t1], "heat_uJ": heat,
            "status": status, "flags": flags,
        })
        diags.extend(inj_diags)

    return {"injections": results, "gaps": gaps}, diags, "ok"


def apply_blank(sample_results, blank_results, rule):
    """Subtract dilution heat using an explicit correspondence rule.

    rule = {"type": "by_index"} | {"type": "by_volume", "tolerance_uL": x}
    Missing correspondences become diagnostics; those injections keep their
    raw heat and get corrected=None. A missing rule is 'unsatisfiable'.
    """
    if not rule:
        return None, [{
            "code": "blank_rule_required",
            "path": "blank_rule",
            "message": "dilution-heat subtraction requires an explicit "
                       "blank-correspondence rule",
            "severity": "error"}], "unsatisfiable"

    blank_by_idx = {b["index"]: b for b in blank_results}
    corrected = []
    diags = []
    for s in sample_results:
        idx = s["index"]
        match = None
        if rule["type"] == "by_index":
            match = blank_by_idx.get(idx)
        elif rule["type"] == "by_volume":
            tol = float(rule.get("tolerance_uL", 0.5))
            cand = [b for b in blank_results
                    if abs(b.get("volume_uL", -1e9) - s.get("volume_uL", -2e9)) <= tol]
            match = cand[0] if cand else None
        else:
            return None, [{
                "code": "blank_rule_unknown",
                "path": "blank_rule.type",
                "message": f"unknown blank rule type {rule['type']!r}",
                "severity": "error"}], "unsatisfiable"

        entry = dict(s)
        if match is None or match.get("heat_uJ") is None:
            entry["blank_index"] = None
            entry["corrected_uJ"] = None
            entry["flags"] = list(s["flags"]) + ["blank_missing"]
            diags.append({
                "code": "blank_missing",
                "path": f"injections[{idx}].blank",
                "message": f"no blank injection corresponds to sample injection "
                           f"{idx} under rule {rule['type']}",
                "severity": "warning"})
        else:
            entry["blank_index"] = match["index"]
            entry["corrected_uJ"] = (None if s["heat_uJ"] is None
                                     else s["heat_uJ"] - match["heat_uJ"])
        corrected.append(entry)
    return {"injections": corrected, "rule": rule}, diags, "ok"
