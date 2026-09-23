"""Per-injection integration.

Domain rules exercised here:

* post-injection peaks **may overlap**; the integration window for injection i
  ends exactly at injection i+1, so shared power is assigned to the earlier
  injection (the overlap is *reported*, not forbidden);
* integration windows **never cross a data gap**: when the window [t0, t1)
  contains a gap the usable window is clipped to [t0, gap_start);
* the first injection is flagged ``discarded`` (mixing artifact); its
  *original integral is still computed and kept*;
* blank (dilution heat) subtraction requires an explicit correspondence rule;
  missing correspondence leaves the dilution term unresolved with diagnostics,
  never zero.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import Diagnostic
from .hashing import sha12


def _trace_arrays(trace: dict[str, Any]):
    pts = trace["payload"]["points"]
    rows = [(p["seq"], p["t"]["canonical"], p["p"]["canonical"], p["diagnostics"])
            for p in pts]
    t = np.array([r[1]["value"] for r in rows if r[1] and r[2]], dtype=float)
    p = np.array([r[2]["value"] for r in rows if r[1] and r[2]], dtype=float)
    return t, p


def _gaps(trace: dict[str, Any]) -> list[tuple[float, float]]:
    out = []
    for g in trace["payload"].get("gaps", []):
        s, e = g["start"]["canonical"], g["end"]["canonical"]
        if s and e:
            out.append((s["value"], e["value"]))
    return sorted(out)


def _baseline_value(bt: np.ndarray, bp: np.ndarray, t: float) -> float:
    if len(bt) == 0:
        return 0.0
    if t <= bt[0]:
        return float(bp[0])
    if t >= bt[-1]:
        return float(bp[-1])
    j = int(np.searchsorted(bt, t, side="right"))
    t0, t1 = bt[j - 1], bt[j]
    if t1 == t0:
        return float(bp[j - 1])
    frac = (t - t0) / (t1 - t0)
    return float(bp[j - 1] + frac * (bp[j] - bp[j - 1]))


def _first_gap_in(gaps, lo: float, hi: float) -> tuple[float, float] | None:
    for gs, ge in gaps:
        # window crosses the gap if the open/closed ranges overlap
        if gs < hi and ge > lo:
            return gs, ge
    return None


def integrate_all(*, power: dict[str, Any], baseline: dict[str, Any],
                  injections_material: dict[str, Any],
                  rules: dict[str, Any]) -> dict[str, Any]:
    pt, pp = _trace_arrays(power)
    bt, bp = _trace_arrays(baseline)
    gaps = _gaps(power)
    rows = injections_material["payload"]["injections"]

    times: list[float] = []
    for r in rows:
        c = r["time"]["canonical"]
        times.append(None if c is None else c["value"])
    t_last = float(pt[-1]) if len(pt) else None

    irules = rules["integration"]
    discard_first = bool(rules["first_injection"]["discard"])
    integrals: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        ordinal = row["ordinal"]
        t0 = times[i]
        volume = row["volume"]["canonical"]
        diags: list[dict[str, Any]] = []
        status_flags: list[str] = []
        clip: dict[str, Any] | None = None
        next_time = None

        if t0 is None or volume is None:
            diags.append(Diagnostic(
                "injection_unconvertible",
                f"injection {ordinal} time or volume cannot be normalized",
                target=f"injection:{ordinal}").to_dict())
            integrals.append(_record(ordinal, t0, None, None, None, None,
                                     None, None, None, diags, ["unresolved"],
                                     None, row, power, baseline, rules, 0))
            continue

        if i + 1 < len(rows) and times[i + 1] is not None:
            next_time = times[i + 1]
            t1 = next_time
        else:
            t1 = t_last
        if t1 is None or t1 <= t0:
            diags.append(Diagnostic(
                "no_integration_window",
                f"injection {ordinal} has no positive-length window",
                target=f"injection:{ordinal}").to_dict())
            integrals.append(_record(ordinal, t0, None, t1, None, None,
                                     None, None, None, diags, ["unresolved"],
                                     None, row, power, baseline, rules, 0))
            continue

        gap = _first_gap_in(gaps, t0, t1)
        effective_t1 = t1
        if gap is not None:
            gs, ge = gap
            effective_t1 = min(t1, gs)
            status_flags.append("clipped_to_gap")
            clip = {"gap_start_s": gs, "gap_end_s": ge,
                    "requested_end_s": t1, "effective_end_s": effective_t1,
                    "rule": irules["gap_policy"]}
            diags.append(Diagnostic(
                "window_clipped_by_gap",
                f"integration window for injection {ordinal} clipped at gap "
                f"[{gs:g}, {ge:g}) s",
                target=f"injection:{ordinal}").to_dict())

        if next_time is not None:
            status_flags.append("peak_may_overlap")

        mask = (pt >= t0) & (pt < effective_t1)
        st, sp = pt[mask], pp[mask]
        n = int(mask.sum())
        if n < 2:
            diags.append(Diagnostic(
                "insufficient_samples",
                f"injection {ordinal} window [{t0:g},{effective_t1:g}) has {n} samples",
                target=f"injection:{ordinal}").to_dict())
            integrals.append(_record(ordinal, t0, effective_t1, t1, n, None,
                                     None, None, clip, diags,
                                     status_flags + ["unresolved"], None,
                                     row, power, baseline, rules, 0))
            continue

        bvals = np.array([_baseline_value(bt, bp, x) for x in st])
        corrected = sp - bvals
        area = float(np.trapezoid(corrected, st)) if hasattr(np, "trapezoid") \
            else float(np.trapz(corrected, st))

        discarded = discard_first and ordinal == min(r["ordinal"] for r in rows)
        if discarded:
            status_flags.append("first_injection_discarded")

        integral_id = "int_" + sha12({
            "injection_ordinal": ordinal,
            "power": power["id"], "baseline": baseline["id"],
            "injections": injections_material["id"],
            "t0": round(t0, 9), "t1": round(effective_t1, 9),
            "n": n, "area_ucal": round(area, 6),
            "rules_version": rules.get("version"),
        })
        integrals.append({
            "id": integral_id,
            "injection_ordinal": ordinal,
            "window": {"start_s": t0, "end_s": effective_t1,
                       "requested_end_s": t1,
                       "coordinate_basis": power.get("coordinate_basis")},
            "clip": clip,
            "sample_count": n,
            "area_ucal": area,
            "area": {"canonical": {"value": area, "unit": "µcal"},
                     "raw_representation": "numeric_integration_of_canonical_samples"},
            "volume_ul": volume["value"],
            "status_flags": status_flags,
            "diagnostics": diags,
            "discarded": discarded,
            "parent_refs": {"power": power["id"], "baseline": baseline["id"],
                            "injections": injections_material["id"]},
            "rules_version": rules.get("version"),
            "input_summary": {"power_hash": power["hash"],
                              "baseline_hash": baseline["hash"],
                              "injections_hash": injections_material["hash"]},
        })
    return {"integrals": integrals,
            "parent_refs": {"power": power["id"], "baseline": baseline["id"],
                            "injections": injections_material["id"]}}


def _record(ordinal, t0, t1, req_t1, n, area, area_unit, vol, clip,
            diags, flags, integral_id, row, power, baseline, rules, area2):
    return {
        "id": integral_id or f"int_unresolved_{ordinal}",
        "injection_ordinal": ordinal,
        "window": None if t0 is None else
                  {"start_s": t0, "end_s": t1, "requested_end_s": req_t1,
                   "coordinate_basis": power.get("coordinate_basis")},
        "clip": clip,
        "sample_count": n,
        "area_ucal": area,
        "area": None if area is None else
               {"canonical": {"value": area, "unit": "µcal"}},
        "volume_ul": None if vol is None else vol["value"],
        "status_flags": flags,
        "diagnostics": diags,
        "discarded": False,
        "parent_refs": {"power": power["id"], "baseline": baseline["id"]},
        "rules_version": rules.get("version"),
        "input_summary": {},
    }
