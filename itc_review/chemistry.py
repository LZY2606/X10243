"""Series geometry, data-gap handling and per-injection integration.

Hard domain rules implemented here (and unit tested directly):

* Post-injection peaks may overlap, but an overlap must be declared via an
  explicit allocation policy; otherwise the overlapping window is blocked.
* An integration window must never integrate *across* a missing-sample gap.
  Windows that end exactly at a gap edge (no observed data beyond) are legal.
* Dilution-heat subtraction must state its correspondence rule; a blank with no
  corresponding injection blocks correction (never a guessed match).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import Diagnostics, SemanticImpossible, Severity
from .units import heat_in_ucal, ucal_to_cal


@dataclass(frozen=True)
class Gap:
    start: float
    end: float
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "path": self.path}


def observed(power_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = power_payload["series"]["rows"]
    t = np.array([r["time"] for r in rows], dtype=float)
    p = np.array([np.nan if r["power"] is None else r["power"]
                  for r in rows], dtype=float)
    return t, p


def find_gaps(power_payload: dict[str, Any]) -> list[Gap]:
    """Structural gaps: explicit gap declarations and runs of missing samples."""
    gaps: list[Gap] = []
    rows = power_payload["series"]["rows"]
    times = [r["time"] for r in rows]
    path = power_payload.get("path", "")
    i = 0
    while i < len(rows):
        if rows[i]["power"] is None:
            j = i
            while j < len(rows) and rows[j]["power"] is None:
                j += 1
            lo = times[i - 1] if i > 0 else times[i]
            hi = times[j] if j < len(rows) else times[j - 1]
            gaps.append(Gap(lo, hi, f"{path}#series:{rows[i]['line']}"))
            i = j
        else:
            i += 1
    for g in power_payload["series"].get("explicit_gaps", []):
        gaps.append(Gap(float(g["start"]), float(g["end"]),
                        f"{path}#gap"))
    gaps.sort(key=lambda g: (g.start, g.end))
    return gaps


def baseline_values(power_payload: dict[str, Any], t: np.ndarray,
                    baseline_payload: dict[str, Any] | None) -> np.ndarray:
    """Subtract a separately versioned baseline. Missing baseline -> zeros."""
    if baseline_payload is None:
        return np.zeros_like(t)
    bt, bp = observed(baseline_payload)
    mask = ~np.isnan(bp)
    if not np.any(mask):
        return np.zeros_like(t)
    if np.sum(mask) >= 2:
        slope, intercept = np.polyfit(bt[mask], bp[mask], 1)
        return slope * t + intercept
    return np.full_like(t, float(bp[mask][0]))


def _observed_intervals(t: np.ndarray, p: np.ndarray) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    i = 0
    n = len(t)
    while i < n:
        if np.isnan(p[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and not np.isnan(p[j + 1]):
            j += 1
        intervals.append((float(t[i]), float(t[j])))
        i = j + 1
    return intervals


def valid_window_mask(t: np.ndarray, p: np.ndarray, start: float, end: float,
                      gaps: list[Gap]) -> tuple[np.ndarray, bool, str]:
    """Mask observed samples in [start, end]; detect gap-bridging.

    Returns (mask, bridges_gap, reason). A mask whose observed samples sit on
    both sides of any gap bridges that gap -> illegal.
    """
    mask = (t >= start) & (t <= end) & ~np.isnan(p)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return mask, False, "no observed samples in window"
    sel_t = t[idx]
    for g in gaps:
        if g.start < start or g.end > end:
            # gap partially/fully outside window; only counts if observed data
            # exists on both sides *within the window*
            left = np.sum((sel_t >= start) & (sel_t <= g.start))
            right = np.sum((sel_t >= g.end) & (sel_t <= end))
            if left > 0 and right > 0:
                return mask, True, (
                    f"window [{start:g},{end:g}] bridges data gap "
                    f"({g.start:g},{g.end:g})")
            continue
        left = np.sum(sel_t <= g.start)
        right = np.sum(sel_t >= g.end)
        if left > 0 and right > 0:
            return mask, True, (
                f"window [{start:g},{end:g}] bridges data gap "
                f"({g.start:g},{g.end:g})")
    return mask, False, ""


@dataclass
class IntegrationParams:
    version: str
    overlap_policy: str = "partition"  # partition | shared | reject
    baseline_version: str | None = None
    tail_factor: float = 3.0  # default tail length multiplier on decay tau

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "overlap_policy": self.overlap_policy,
                "baseline_version": self.baseline_version,
                "tail_factor": self.tail_factor}


def default_window(inj: dict[str, Any], next_time: float | None,
                   power_t: np.ndarray, tau_seconds: float,
                   tail_factor: float) -> tuple[float, float, str]:
    start = float(inj["time"]["canonical_value"])
    candidate = start + tail_factor * tau_seconds
    if next_time is not None and candidate >= next_time:
        return start, float(next_time), "partition:next-injection-start"
    tmax = float(np.nanmax(power_t))
    if candidate >= tmax:
        return start, tmax, "end-of-recording"
    return start, float(candidate), "tail-factor"


def integrate_injection(
    *,
    injection: dict[str, Any],
    power: dict[str, Any],
    baseline: dict[str, Any] | None,
    start: float,
    end: float,
    overlap: str,  # none | overlap-next | overlap-prev
    params: IntegrationParams,
    path: str,
) -> dict[str, Any]:
    """Integrate one injection. Returns a structured attempt (success or block)."""
    diags = Diagnostics()
    t, p = observed(power)
    gaps = find_gaps(power)

    attempt: dict[str, Any] = {
        "injection": injection["injection"],
        "window": {"start_s": start, "end_s": end,
                   "time_base": "injection-timestamp; seconds"},
        "overlap": overlap,
        "params_version": params.version,
        "baseline_version": params.baseline_version,
        "status": "ok",
        "diagnostics": [],
        "observed_samples": 0,
        "observed_start_s": None,
        "observed_end_s": None,
    }

    if overlap != "none" and params.overlap_policy == "reject":
        diags.add("OVERLAP_REJECTED",
                  f"peak overlaps (post-injection peak reaches next injection); "
                  f"policy 'reject' forbids allocation",
                  path, Severity.BLOCKED,
                  detail={"overlap": overlap})
        attempt["status"] = "blocked"
    if overlap != "none" and params.overlap_policy == "partition" and \
            overlap == "overlap-prev":
        # Previous peak owns the shared region by partition: this injection's
        # window legally starts at its timestamp (the shared tail is excluded
        # because its end == next start for the previous one).
        diags.add("OVERLAP_PARTITIONED",
                  "shared tail allocated to previous injection (partition rule)",
                  path, Severity.INFO, detail={"overlap": overlap})
    elif overlap != "none" and params.overlap_policy == "shared":
        diags.add("OVERLAP_SHARED",
                  "shared region counted for both peaks (declared shared rule)",
                  path, Severity.WARNING, detail={"overlap": overlap})

    mask, bridges, reason = valid_window_mask(t, p, start, end, gaps)
    if bridges:
        diags.add("WINDOW_BRIDGES_GAP", reason, path, Severity.BLOCKED,
                  start=start, end=end)
        attempt["status"] = "blocked"
    idx = np.where(mask)[0]
    if idx.size == 0:
        diags.add("EMPTY_WINDOW", "no observed samples inside window", path,
                  Severity.BLOCKED, start=start, end=end)
        attempt["status"] = "blocked"

    attempt["diagnostics"] = diags.to_list()
    if attempt["status"] == "blocked":
        return attempt

    tt = t[idx]
    base = baseline_values(power, tt, baseline)
    sig = p[idx] - base
    area_w = float(np.trapezoid(sig, tt)) if hasattr(np, "trapezoid") \
        else float(np.trapz(sig, tt))  # type: ignore[attr-defined]
    heat_cal = area_w  # W*s = J ... but instrument columns are ucal/s: handled below
    # power_unit determines interpretation; raw instrument unit preserved.
    unit = (power.get("power_unit", {}) or {}).get("unit", "").strip().lower()
    if unit in ("ucal/s", "µcal/s", "μcal/s", "cal/s"):
        factor = {"ucal/s": 1e-6, "µcal/s": 1e-6, "μcal/s": 1e-6,
                  "cal/s": 1.0}[unit]
        area_ucal = area_w * factor
        area_cal = area_ucal
    else:  # W -> J -> cal
        area_cal = area_w / 4.184
        area_ucal = heat_in_ucal(area_cal)
    attempt.update({
        "observed_samples": int(idx.size),
        "observed_start_s": float(tt[0]),
        "observed_end_s": float(tt[-1]),
        "integrated_heat_ucal": float(area_ucal),
        "integrated_heat": {"value": float(area_ucal), "unit": "µcal",
                            "canonical_cal": float(ucal_to_cal(area_ucal))},
        "endpoint_rule": attempt.get("endpoint_rule", ""),
    })
    return attempt


def corrected_heat(attempt: dict[str, Any], blank_attempt: dict[str, Any] | None,
                   blank_rule: dict[str, Any], *, path: str) -> dict[str, Any]:
    """Apply dilution-heat subtraction with an explicit correspondence rule."""
    diags = Diagnostics()
    for d in attempt.get("diagnostics", []):
        diags.add(d["code"], d["message"], d["path"],
                  Severity(d.get("severity", "error")),
                  d.get("value"), **(d.get("detail") or {}))
    if attempt["status"] != "ok":
        out = dict(attempt)
        out["correction_rule_version"] = blank_rule["version"]
        out["correction_status"] = "not-corrected:blocked-integration"
        return out
    if blank_attempt is None:
        mode = blank_rule.get("mode", "by_index")
        diags.add(
            "BLANK_CORRESPONDENCE_MISSING",
            f"no blank injection corresponds under rule "
            f"{blank_rule['version']}/{mode}; dilution heat cannot be "
            f"subtracted (no guessed match)",
            path, Severity.BLOCKED,
            rule=blank_rule["version"], mode=mode)
        out = dict(attempt)
        out["status"] = "blocked"
        out["diagnostics"] = diags.to_list()
        out["correction_rule_version"] = blank_rule["version"]
        out["correction_status"] = "blocked:blank-correspondence-missing"
        out["dilution_heat_ucal"] = None
        out["corrected_heat_ucal"] = None
        return out
    if blank_attempt.get("status") != "ok":
        out = dict(attempt)
        out["status"] = "blocked"
        out["diagnostics"] = attempt.get("diagnostics", []) + [{
            "code": "BLANK_BLOCKED",
            "message": "corresponding blank integration is blocked",
            "path": path, "severity": Severity.BLOCKED.value,
            "value": None, "detail": {}}]
        out["correction_status"] = "blocked:blank-blocked"
        return out
    dilution = float(blank_attempt["integrated_heat_ucal"])
    corrected = float(attempt["integrated_heat_ucal"]) - dilution
    out = dict(attempt)
    out["dilution_heat_ucal"] = dilution
    out["corrected_heat_ucal"] = corrected
    out["correction_status"] = "corrected"
    out["correction_rule_version"] = blank_rule["version"]
    out["correction_correspondence"] = {
        "mode": blank_rule.get("mode"),
        "sample_injection": attempt["injection"],
        "blank_injection": blank_attempt["injection"],
        "blank_version": blank_rule.get("blank_version"),
    }
    return out
