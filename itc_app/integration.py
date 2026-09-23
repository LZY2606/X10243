"""Injection integration engine.

Structural rules (rule pack ITC-RULES-2026.09.1):

* Injection windows are [t_i, t_{i+1}); the final window ends at the
  nominal endpoint or the last sample, whichever comes first.
* Peaks of consecutive injections may overlap; that never merges windows.
* A window never spans a declared data gap. The window is clipped at the gap
  boundary and the integral is BLOCKED (no heat reported) when the gap cuts
  into the post-peak return region.
* The first injection is integrated normally but flagged DISCARDED for
  fitting (standard ITC first-shot practice).
* Dilution-heat subtraction requires a stated correspondence rule and a
  matched blank per injection; unmatched injections are reported, not
  silently zeroed.
* Every output records input summaries, rule version and parent material ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .materials import MaterialSet, Curve, Sample, Gap
from .units import RULES_VERSION, Diagnostic

STATUS_OK = "ok"
STATUS_OK_TRUNCATED = "ok_truncated_endpoint"
STATUS_DISCARDED = "discarded_first_injection"
STATUS_BLOCKED_GAP = "blocked_gap"
STATUS_BLOCKED_RANGE = "blocked_out_of_range"
STATUS_BLANK_MISSING = "blank_missing"
STATUS_BLANK_UNMATCHED = "blank_unmatched"
STATUS_BASELINE_MISSING = "baseline_missing"

FLAG_OVERLAP = "overlap_peak"
FLAG_GAP_CLIPPED = "gap_clipped_window"
FLAG_ENDPOINT_CLIPPED = "endpoint_clipped_at_last_sample"
FLAG_BLANK_SUBTRACTED = "blank_subtracted"


@dataclass
class Segment:
    start: float
    end: float
    touched_gap: bool = False

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "touched_gap": self.touched_gap}


@dataclass
class IntegralResult:
    injection_index: int
    window_start: float
    window_end: float
    status: str
    discarded: bool = False
    flags: list[str] = field(default_factory=list)
    gross_heat_uJ: float | None = None
    blank_heat_uJ: float | None = None
    net_heat_uJ: float | None = None
    blank_ref: int | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    endpoints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "injection_index": self.injection_index,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "status": self.status,
            "discarded": self.discarded,
            "flags": self.flags,
            "gross_heat_uJ": self.gross_heat_uJ,
            "blank_heat_uJ": self.blank_heat_uJ,
            "net_heat_uJ": self.net_heat_uJ,
            "blank_ref": self.blank_ref,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "segments": [s.to_dict() for s in self.segments],
            "endpoints": self.endpoints,
        }


@dataclass
class IntegrationRun:
    rule_version: str
    input_summary: dict
    parents: dict
    results: list[IntegralResult]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule_version": self.rule_version,
            "input_summary": self.input_summary,
            "parents": self.parents,
            "results": [r.to_dict() for r in self.results],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _series(curve: Curve):
    ts = [s.t.canonical for s in curve.samples]
    ps = [s.p.canonical if s.p.convertible else None for s in curve.samples]
    return ts, ps


def _gap_intervals(curve: Curve):
    out = []
    for g in curve.gaps:
        out.append((g.start.canonical, g.end.canonical))
    return sorted(out)


def _segments_for_window(start: float, end: float, gaps):
    """Split [start, end) at declared gaps. Never crosses a gap."""
    segs = []
    cur = start
    touched = False
    for gs, ge in gaps:
        if ge <= cur:
            continue
        if gs >= end:
            break
        touched = True
        if gs > cur:
            segs.append(Segment(cur, min(gs, end)))
        cur = max(cur, ge)
        if cur >= end:
            break
    if cur < end:
        segs.append(Segment(cur, end))
    for s in segs:
        s.touched_gap = touched
    return segs, touched


def _interp_integrate(ts, ps, a: float, b: float):
    """Integrate trapezoidal over [a,b], linearly interpolating endpoints.

    Unconvertible sample values force the segment to be skipped upstream;
    here ps values are assumed numeric or None only outside [a,b].
    """
    if b <= a:
        return None
    lo = _bisect_left(ts, a) - 1
    hi = _bisect_left(ts, b)
    lo = max(lo, 0)
    hi = min(hi, len(ts) - 1)
    pts = []
    for k in range(lo, hi + 1):
        if a <= ts[k] <= b:
            if ps[k] is None:
                return None
            pts.append((ts[k], ps[k]))
    if not pts:
        return None
    # interpolate endpoint values
    pa = _interp(ts, ps, a)
    pb = _interp(ts, ps, b)
    if pa is None or pb is None:
        return None
    if pts[0][0] > a:
        pts.insert(0, (a, pa))
    if pts[-1][0] < b:
        pts.append((b, pb))
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


def _interp(ts, ps, x: float):
    import bisect as _b

    if x < ts[0] or x > ts[-1]:
        return None
    k = _b.bisect_left(ts, x)
    if k < len(ts) and ts[k] == x:
        return ps[k]
    x0, x1 = ts[k - 1], ts[k]
    y0, y1 = ps[k - 1], ps[k]
    if y0 is None or y1 is None:
        return None
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _bisect_left(ts, x: float) -> int:
    import bisect as _b

    return _b.bisect_left(ts, x)

STATUS_BLOCKED_ENDPOINT = "blocked_endpoint"
RECOVERY_TOL = 0.05      # 5% of peak amplitude counts as "returned"
OVERLAP_TOL = 0.02       # 2% residual at next shot counts as overlapping peaks


def _baseline_at(mat: MaterialSet, t: float):
    if mat.baseline is None or not mat.baseline.samples:
        return None
    bts, bps = _series(mat.baseline)
    return _interp(bts, bps, t)


def _window_power(mat: MaterialSet, a: float, b: float):
    """Baseline-corrected power samples restricted to [a,b], with diagnostics."""
    ts, ps = _series(mat.power)
    lo = max(0, _bisect_left(ts, a) - 1)
    hi = min(len(ts) - 1, _bisect_left(ts, b))
    xs, ys, missing_value = [], [], False
    baseline_missing = mat.baseline is None or not mat.baseline.samples
    for k in range(lo, hi + 1):
        if ps[k] is None:
            missing_value = True
            continue
        bl = _baseline_at(mat, ts[k])
        if bl is None:
            bl = 0.0
        xs.append(ts[k])
        ys.append(ps[k] - bl)
    return xs, ys, missing_value, baseline_missing


def _peak_recovery(xs, ys, start: float):
    """Time of peak and first time it returns within RECOVERY_TOL of peak amp."""
    if not xs:
        return None, None, None
    base = ys[0]
    pk_idx = max(range(len(ys)), key=lambda k: abs(ys[k] - base))
    peak = ys[pk_idx]
    amp = abs(peak - base)
    recovery = None
    if amp > 0:
        for k in range(pk_idx, len(ys)):
            if abs(ys[k] - base) <= RECOVERY_TOL * amp:
                recovery = xs[k]
                break
    return xs[pk_idx], amp, recovery


def _residual_fraction(xs, ys, t: float):
    """Fraction of peak amplitude still present at time t."""
    if not xs:
        return 0.0
    base = ys[0]
    amp = max(abs(y - base) for y in ys)
    if amp <= 0:
        return 0.0
    val = _interp(xs, ys, t)
    if val is None:
        return 1.0
    return abs(val - base) / amp


def integrate(mat: MaterialSet, parents: dict | None = None, rule_version: str = RULES_VERSION):
    ts, ps = _series(mat.power)
    gaps = _gap_intervals(mat.power)
    n = len(mat.injections)
    times = [inj.time.canonical for inj in mat.injections]
    t_last = ts[-1]
    # nominal final endpoint: last sample time (documented coordinate basis)
    nominal_end = t_last
    run_diags: list[Diagnostic] = []
    baseline_missing_global = mat.baseline is None or not mat.baseline.samples
    if baseline_missing_global:
        run_diags.append(
            Diagnostic(
                "W_BASELINE_MISSING",
                "warning",
                "no baseline curve supplied; integrals use zero baseline (stated assumption)",
                "@baseline",
            )
        )
    blank_plan = _plan_blank(mat, run_diags)
    results: list[IntegralResult] = []
    for i, inj in enumerate(mat.injections):
        start = times[i]
        end = times[i + 1] if i + 1 < n else nominal_end
        res = IntegralResult(inj.index, start, end, STATUS_OK)
        if start < ts[0] or start > t_last:
            res.status = STATUS_BLOCKED_RANGE
            res.diagnostics.append(
                Diagnostic(
                    "E_OUT_OF_RANGE",
                    "error",
                    f"injection {inj.index} at {start}s lies outside sampled power range",
                    f"line {inj.line}",
                )
            )
            results.append(res)
            continue
        clipped_end = False
        if end > t_last:
            end = t_last
            clipped_end = True
        res.window_end = end
        segs, touched = _segments_for_window(start, end, gaps)
        xs, ys, missing_value, baseline_missing = _window_power(mat, start, end)
        peak_t, amp, recovery = _peak_recovery(xs, ys, start)
        if baseline_missing:
            res.flags.append(STATUS_BASELINE_MISSING)
        # overlap with the following injection?
        if i + 1 < n:
            frac = _residual_fraction(xs, ys, times[i + 1])
            if frac > OVERLAP_TOL:
                res.flags.append(FLAG_OVERLAP)
                res.endpoints["residual_at_next_injection"] = round(frac, 6)
        # data gap handling: block when the gap removes part of the peak body
        if touched:
            # A gap that starts only after the peak has demonstrably returned
            # to baseline leaves the integral intact; any earlier gap blocks.
            window_gaps = [(gs, ge) for gs, ge in gaps if ge > start and gs < end]
            first_gap = min(gs for gs, ge in window_gaps)
            gap_blocks = recovery is None or first_gap < recovery
            if gap_blocks:
                res.status = STATUS_BLOCKED_GAP
                res.flags.append(FLAG_GAP_CLIPPED)
                res.segments = segs
                res.diagnostics.append(
                    Diagnostic(
                        "E_GAP_CROSSES_WINDOW",
                        "error",
                        f"integration window of injection {inj.index} intersects a data gap; "
                        "window is clipped and heat is not reported",
                        "@gap",
                    )
                )
                results.append(res)
                continue
            res.flags.append(FLAG_GAP_CLIPPED)
        # endpoint test on the final window
        if clipped_end:
            tail = abs(ys[-1] - ys[0]) if ys else 0.0
            if amp and tail > RECOVERY_TOL * amp:
                res.status = STATUS_BLOCKED_ENDPOINT
                res.flags.append(FLAG_ENDPOINT_CLIPPED)
                res.diagnostics.append(
                    Diagnostic(
                        "E_ENDPOINT_UNRECOVERED",
                        "error",
                        f"final injection {inj.index} truncates before the peak recovers "
                        f"(residual {tail:.3g} > {RECOVERY_TOL:.0%} of amplitude {amp:.3g})",
                        "@power/end",
                    )
                )
                results.append(res)
                continue
            res.status = STATUS_OK_TRUNCATED
            res.flags.append(FLAG_ENDPOINT_CLIPPED)
        # integrate segment by segment (never across gaps)
        gross = 0.0
        ok = True
        for seg in segs:
            part = _interp_integrate(ts, ps, seg.start, seg.end)
            if part is None:
                ok = False
                break
            # subtract baseline analytically over the same interval
            if not baseline_missing_global:
                bts, bps = _series(mat.baseline)
                bl_part = _interp_integrate(bts, bps, seg.start, seg.end)
                if bl_part is not None:
                    part -= bl_part
            gross += part
        if not ok:
            res.status = STATUS_BLOCKED_GAP
            res.diagnostics.append(
                Diagnostic(
                    "E_GAP_CROSSES_WINDOW",
                    "error",
                    f"injection {inj.index}: missing power value inside window",
                    "@gap",
                )
            )
            results.append(res)
            continue
        res.gross_heat_uJ = gross
        # blank dilution heat under the stated correspondence rule
        blank_heat, blank_status = blank_plan.get(i)
        if blank_status is not None:
            res.status = blank_status
        if blank_heat is not None:
            res.blank_heat_uJ = blank_heat
            res.net_heat_uJ = gross - blank_heat
            res.flags.append(FLAG_BLANK_SUBTRACTED)
        else:
            res.net_heat_uJ = None
        if i == 0:
            res.discarded = True
        res.segments = segs
        res.endpoints.setdefault("peak_time_s", peak_t)
        res.endpoints.setdefault("peak_amplitude_uW", amp)
        results.append(res)
    summary = {
        "power_samples": len(mat.power.samples),
        "power_gaps": len(mat.power.gaps),
        "injections": n,
        "blank_rule": mat.blank_rule,
        "has_baseline": not baseline_missing_global,
        "has_blank": mat.blank is not None and bool(mat.blank.samples),
        "time_basis": f"{mat.power.time_unit} -> s (canonical)",
    }
    return IntegrationRun(
        rule_version=rule_version,
        input_summary=summary,
        parents=parents or {},
        results=results,
        diagnostics=run_diags,
    )


def _blank_integrals(mat: MaterialSet):
    """Integrate the blank dilution curve per blank-injection ordinal."""
    if mat.blank is None or not mat.blank.samples:
        return None
    ts, ps = _series(mat.blank)
    btimes = [q.canonical for q in mat.blank.injection_times]
    gaps = _gap_intervals(mat.blank)
    heats: list[float | None] = []
    n = len(btimes)
    for k, st in enumerate(btimes):
        en = btimes[k + 1] if k + 1 < n else ts[-1]
        if st < ts[0] or st > ts[-1] or en > ts[-1]:
            heats.append(None)
            continue
        segs, touched = _segments_for_window(st, en, gaps)
        total = 0.0
        failed = False
        for seg in segs:
            part = _interp_integrate(ts, ps, seg.start, seg.end)
            if part is None:
                failed = True
                break
            total += part
        heats.append(None if failed else total)
    return {"times": btimes, "heats": heats}


def _plan_blank(mat: MaterialSet, run_diags: list[Diagnostic]):
    """Return mapping ordinal -> (heat|None, status|None) under the stated rule."""
    n = len(mat.injections)
    plan: dict[int, tuple[float | None, str | None]] = {i: (None, None) for i in range(n)}
    blank = _blank_integrals(mat)
    if blank is None:
        run_diags.append(
            Diagnostic(
                "W_BLANK_MISSING",
                "warning",
                "no blank control supplied; dilution heat is not subtracted (correspondence rule: "
                + mat.blank_rule
                + ")",
                "@blank",
            )
        )
        for i in range(n):
            plan[i] = (None, STATUS_BLANK_MISSING)
        return plan
    times = [inj.time.canonical for inj in mat.injections]
    for i, inj in enumerate(mat.injections):
        ref: int | None
        if mat.blank_rule == "index":
            ref = i + 1
        elif mat.blank_rule == "manual":
            ref = inj.blank_ref
        else:  # nearest
            j = min(range(len(blank["times"])), key=lambda k: abs(blank["times"][k] - times[i]))
            ref = j + 1
        if ref is None:
            plan[i] = (None, STATUS_BLANK_MISSING)
            run_diags.append(
                Diagnostic(
                    "E_BLANK_CORRESPONDENCE_MISSING",
                    "warning",
                    f"injection {inj.index}: manual rule but no blank_ref declared",
                    f"line {inj.line}",
                )
            )
            continue
        if not 1 <= ref <= len(blank["heats"]):
            plan[i] = (None, STATUS_BLANK_UNMATCHED)
            run_diags.append(
                Diagnostic(
                    "E_BLANK_NO_MATCH",
                    "warning",
                    f"injection {inj.index}: no blank injection #{ref} (blank has {len(blank['heats'])})",
                    f"line {inj.line}",
                )
            )
            continue
        heat = blank["heats"][ref - 1]
        if heat is None:
            plan[i] = (None, STATUS_BLANK_MISSING)
            run_diags.append(
                Diagnostic(
                    "E_BLANK_INTEGRAL_UNUSABLE",
                    "warning",
                    f"injection {inj.index}: matched blank #{ref} integral is blocked by gap/range",
                    "@blank",
                )
            )
            continue
        plan[i] = (heat, None)
    return plan
