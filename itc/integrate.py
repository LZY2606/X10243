"""Per-injection peak integration with explicit overlap / gap semantics.

Hard rules (encoded here, enforced by tests - never only in the browser):

1. Post-injection peaks *may* overlap.  An injection's integration window ends
   no later than the next injection time, so overlapping peaks are never
   double counted; the overlap is reported, not hidden.
2. An integration window *never* crosses a data gap.  When a gap falls inside
   the natural window, the window terminates at the gap start and the record
   is marked ``gap_truncated`` with a diagnostic and an explicit endpoint
   decision (``before_gap``).
3. The first injection may be flagged for discard ("first injection discard");
   it is still integrated for evidence but excluded from model fitting.
4. Dilution heat subtraction requires a stated correspondence rule.  A missing
   blank counterpart yields ``blank_heat = None`` plus a diagnostic instead of
   a silent zero.
5. Trapezoidal integration of baseline-subtracted power gives energy.  All
   energies are returned in cal (canonical) with their unit attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import Diagnostic, SemanticInfeasible
from .materials import MaterialRevision
from .units import normalise, require_number

EPS = 1e-12


@dataclass
class RunArrays:
    run_id: str
    time_s: np.ndarray
    power_cal_s: np.ndarray
    flags: list[str]
    gaps_s: list[tuple[float, float]]
    time_reference: str
    power_diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass
class InjectionPoint:
    ordinal: int
    time_s: float
    volume_l: float
    window_start_s: float
    window_end_s: float
    window_end_reason: str
    n_samples: int
    heat_cal: float | None
    baseline_cal_s: float
    overlapping_next: bool
    peak_tail_after_next_s: float | None
    discard: bool
    gap_truncated: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    override: dict | None = None

    def to_dict(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "time_s": self.time_s,
            "time_reference": "seconds from acquisition start",
            "volume_l": self.volume_l,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "window_end_reason": self.window_end_reason,
            "n_samples": self.n_samples,
            "heat_cal": self.heat_cal,
            "heat_ucal": None if self.heat_cal is None else self.heat_cal * 1e6,
            "energy_unit": "cal",
            "baseline_cal_s": self.baseline_cal_s,
            "overlapping_next": self.overlapping_next,
            "peak_tail_after_next_s": self.peak_tail_after_next_s,
            "discard": self.discard,
            "gap_truncated": self.gap_truncated,
            "override": self.override,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def extract_run(power_rev: MaterialRevision, injections_rev: MaterialRevision,
                baseline_rev: MaterialRevision) -> tuple[RunArrays, list[dict[str, Any]]]:
    """Normalise one run's immutable materials into SI arrays + injections."""
    diagnostics: list[Diagnostic] = []
    times: list[float] = []
    powers: list[float] = []
    flags: list[str] = []
    for i, row in enumerate(power_rev.data["samples"]):
        tv = row["t"]
        pv = row["p"]
        t_val, _, t_diag = normalise(tv, "time")
        p_val, _, p_diag = normalise(pv, "power")
        if t_diag:
            diagnostics.append(_relocate(t_diag, f"power:{tv.location}"))
        if p_diag:
            diagnostics.append(_relocate(p_diag, f"power:{pv.location}"))
        if t_val is not None and p_val is not None:
            times.append(t_val)
            powers.append(p_val)
            flags.append(str(row.get("flag", "ok")))
    gaps_s: list[tuple[float, float]] = []
    for g0, g1 in power_rev.data["gaps"]:
        a, _, da = normalise(g0, "time")
        b, _, db = normalise(g1, "time")
        if da:
            diagnostics.append(_relocate(da, f"power:{g0.location}"))
        if db:
            diagnostics.append(_relocate(db, f"power:{g1.location}"))
        if a is not None and b is not None:
            gaps_s.append((a, b))
    order = np.argsort(np.asarray(times, dtype=float)) if times else np.array([], int)
    time_s = np.asarray(times, dtype=float)[order]
    power_cal_s = np.asarray(powers, dtype=float)[order]
    flags = [flags[i] for i in order.tolist()]
    if len(time_s) >= 2 and np.any(np.diff(time_s) <= 0):
        raise SemanticInfeasible(
            "power time stamps must be strictly increasing",
            [Diagnostic("power:non_monotonic_time", "non-monotonic time",
                        "power.samples.t", None)])

    base = baseline_rev.data
    level = require_number(base["level"], "power")
    slope = 0.0
    if base.get("slope") is not None:
        slope, _, slope_diag = normalise(base["slope"], "power")
        if slope_diag:
            diagnostics.append(slope_diag)
            slope = 0.0

    injections = []
    seen: set[int] = set()
    for row in injections_rev.data["injections"]:
        ordinal = row["ordinal"]
        if ordinal in seen:
            raise SemanticInfeasible(
                "duplicate injection ordinal",
                [Diagnostic("injections:duplicate_ordinal", "duplicate ordinal",
                            "injections.ordinal", ordinal)])
        seen.add(ordinal)
        t_s = require_number(row["time"], "time")
        vol_l = require_number(row["volume"], "volume")
        injections.append({"ordinal": ordinal, "time_s": t_s, "volume_l": vol_l,
                           "discard": bool(row.get("discard_first", False))})
    injections.sort(key=lambda r: r["time_s"])
    if injections and time_s.size:
        if injections[0]["time_s"] < time_s[0] - EPS:
            raise SemanticInfeasible(
                "an injection precedes the sampled interval",
                [Diagnostic("injections:before_trace",
                            f"ordinal {injections[0]['ordinal']} at "
                            f"{injections[0]['time_s']}s < first sample {time_s[0]}s",
                            "injections.time", injections[0]["ordinal"])])
    arrays = RunArrays(power_rev.run_id, time_s, power_cal_s, flags,
                       sorted(gaps_s), power_rev.data["time_reference"], diagnostics)
    arrays.power_diagnostics = diagnostics
    return arrays, injections


def baseline_value(level: float, slope: float, t_s: float) -> float:
    return level + slope * t_s


def integrate_run(arrays: RunArrays, injections: list[dict], baseline_rev: MaterialRevision,
                  *, overrides: dict[int, dict] | None = None) -> list[InjectionPoint]:
    """Integrate every injection. See module docstring for the hard rules."""
    overrides = overrides or {}
    base = baseline_rev.data
    level = require_number(base["level"], "power")
    slope = 0.0
    slope_rv = base.get("slope")
    if slope_rv is not None and slope_rv.value is not None:
        slope = require_number(slope_rv, "power")

    t_all = arrays.time_s
    trace_end = float(t_all[-1]) if t_all.size else 0.0
    points: list[InjectionPoint] = []
    for idx, inj in enumerate(injections):
        ordinal = inj["ordinal"]
        start = inj["time_s"]
        next_start = injections[idx + 1]["time_s"] if idx + 1 < len(injections) else trace_end
        natural_end = next_start if idx + 1 < len(injections) else trace_end
        end_reason = "next_injection" if idx + 1 < len(injections) else "trace_end"

        ov = overrides.get(ordinal)
        if ov:
            if "window_end_s" in ov and ov["window_end_s"] is not None:
                natural_end = float(ov["window_end_s"])
                end_reason = "manual_override"

        # gaps never get crossed: shrink the window to the first gap start.
        gap_truncated = False
        for ga, gb in arrays.gaps_s:
            if start < ga < natural_end:
                natural_end = ga
                end_reason = "before_gap"
                gap_truncated = True
                break

        # manual overrides may not cross a gap either
        if ov and ov.get("window_end_s") is not None and gap_truncated:
            pass  # already shortened

        seg_t = t_all[(t_all >= start - EPS) & (t_all <= natural_end + EPS)]
        seg_p = arrays.power_cal_s[(t_all >= start - EPS) & (t_all <= natural_end + EPS)]
        diagnostics: list[Diagnostic] = []
        heat: float | None = None
        n_samples = seg_t.size
        if seg_t.size >= 2:
            corrected = seg_p - (level + slope * seg_t)
            heat = float(np.trapezoid(corrected, seg_t))
        else:
            diagnostics.append(Diagnostic(
                "integrate:empty_window",
                f"ordinal {ordinal}: fewer than two samples in window",
                f"injections[{idx}]", ordinal))

        overlapping = False
        tail: float | None = None
        if idx + 1 < len(injections):
            # detect residual peak at the next injection boundary: the corrected
            # signal is still materially above baseline there.
            boundary_p = _interp_power(arrays, next_start) - (level + slope * next_start)
            if boundary_p is not None and boundary_p > 1e-8:
                overlapping = True
                tail = float(boundary_p)
                diagnostics.append(Diagnostic(
                    "integrate:overlapping_peak",
                    f"ordinal {ordinal}: peak is still present when ordinal "
                    f"{injections[idx+1]['ordinal']} starts; window ends at next injection",
                    f"injections[{idx}]", ordinal, severity="warning"))

        if gap_truncated:
            diagnostics.append(Diagnostic(
                "integrate:gap_truncated",
                f"ordinal {ordinal}: window truncated before a data gap; "
                "post-gap tail is not integrated",
                f"injections[{idx}]", ordinal, severity="warning"))

        if inj["discard"]:
            diagnostics.append(Diagnostic(
                "integrate:first_injection_discarded",
                f"ordinal {ordinal}: first injection discarded from model fitting",
                f"injections[{idx}]", ordinal, severity="info"))

        points.append(InjectionPoint(
            ordinal=ordinal, time_s=float(start), volume_l=float(inj["volume_l"]),
            window_start_s=float(start), window_end_s=float(natural_end),
            window_end_reason=end_reason, n_samples=n_samples, heat_cal=heat,
            baseline_cal_s=level, overlapping_next=overlapping,
            peak_tail_after_next_s=tail, discard=bool(inj["discard"]),
            gap_truncated=gap_truncated, diagnostics=diagnostics,
            override=ov or None))
    return points


def _interp_power(arrays: RunArrays, t_s: float) -> float | None:
    t = arrays.time_s
    if t.size == 0 or t_s < t[0] or t_s > t[-1]:
        return None
    return float(np.interp(t_s, t, arrays.power_cal_s))


def _relocate(diag: Diagnostic, location: str) -> Diagnostic:
    return Diagnostic(diag.code, diag.message, location, diag.value, diag.severity)


# --------------------------------------------------------------------------- #
# Dilution / blank correspondence
# --------------------------------------------------------------------------- #

def match_blanks(sample_points: list[InjectionPoint],
                 blank_points: list[InjectionPoint],
                 blank_link: MaterialRevision) -> list[dict]:
    """Attach dilution heats using the *stated* correspondence rule.

    ``rule = ordinal`` pairs identical ordinals; when
    ``discard_first_in_blank`` is true the blank's first shot is dropped and
    blank ordinal ``i`` maps to sample ordinal ``i+1``.  A missing counterpart
    produces ``blank_heat_cal=None`` and an explicit diagnostic (never zero).
    """
    rule = blank_link.data["rule"]
    shift = 1 if blank_link.data.get("discard_first_in_blank") else 0
    by_ord = {p.ordinal: p for p in blank_points}
    out = []
    for p in sample_points:
        wanted = p.ordinal + shift
        partner = by_ord.get(wanted)
        rec = {
            "ordinal": p.ordinal,
            "rule": rule,
            "blank_ordinal": wanted,
            "blank_heat_cal": None,
            "blank_heat_ucal": None,
            "status": "matched",
            "diagnostics": [],
        }
        if partner is None or partner.heat_cal is None:
            rec["status"] = "missing_counterpart"
            rec["diagnostics"].append(Diagnostic(
                "blank:missing_counterpart",
                f"sample ordinal {p.ordinal}: no blank counterpart ordinal "
                f"{wanted} under rule '{rule}' (blank run "
                f"{blank_link.data['blank_run']}); dilution heat left unassigned",
                f"blank_link/{p.ordinal}", wanted, severity="warning").to_dict())
        else:
            rec["blank_heat_cal"] = partner.heat_cal
            rec["blank_heat_ucal"] = partner.heat_cal * 1e6
        out.append(rec)
    return out


def corrected_heats(points: list[InjectionPoint], blank_matches: list[dict]
                    ) -> list[dict]:
    """Merge integration + blank results.  Missing blank => unassigned heat."""
    match_by = {m["ordinal"]: m for m in blank_matches}
    rows = []
    for p in points:
        m = match_by.get(p.ordinal, {})
        net = None
        if p.heat_cal is not None and m.get("blank_heat_cal") is not None:
            net = p.heat_cal - m["blank_heat_cal"]
        rows.append({
            "ordinal": p.ordinal,
            "time_s": p.time_s,
            "volume_l": p.volume_l,
            "raw_heat_cal": p.heat_cal,
            "raw_heat_ucal": None if p.heat_cal is None else p.heat_cal * 1e6,
            "blank_heat_cal": m.get("blank_heat_cal"),
            "blank_heat_ucal": m.get("blank_heat_ucal"),
            "blank_status": m.get("status", "no_rule"),
            "net_heat_cal": net,
            "net_heat_ucal": None if net is None else net * 1e6,
            "discard": p.discard,
            "overlapping_next": p.overlapping_next,
            "gap_truncated": p.gap_truncated,
            "window_start_s": p.window_start_s,
            "window_end_s": p.window_end_s,
            "window_end_reason": p.window_end_reason,
            "integrations": [d.to_dict() for d in p.diagnostics],
            "blank_diagnostics": m.get("diagnostics", []),
        })
    return rows
