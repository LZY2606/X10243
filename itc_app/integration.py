"""Per-injection integration engine.

Hard rules exercised by the counter-example archive:

* post-injection peaks may overlap; the integration *window* of injection i
  is [t_i, t_{i+1}) by default and never extends past the next injection.
* an integration window must not *cross* a data gap.  A window that starts or
  ends exactly at a gap boundary is valid (endpoints are allowed to touch);
  a missing sample strictly inside rejects that injection with a positioned
  diagnostic -- no interpolation fills it.
* dilution-heat subtraction is applied only under an explicit, versioned
  correspondence rule (which blank injection belongs to which primary
  injection).  A missing correspondence leaves the raw heat in place and
  records ``blank_correspondence_missing`` instead of silently subtracting.
* first-injection discard is a *decision*, never implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .errors import Diagnostic, SemanticUnresolvable, diag
from .models import IntegrationPoint, Trace

INTEGRATION_RULE_VERSION = "integrate-v1.0"


@dataclass
class Window:
    injection_index: int
    start: float
    end: float
    source: str = "default"               # default | manual | clipped


@dataclass
class IntegrationBatch:
    rule_version: str
    origin: str
    points: list[IntegrationPoint]
    windows: list[Window]
    inputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(p.status == "rejected" for p in self.points):
            return "rejected"
        if any(p.status == "warning" for p in self.points):
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        return d


def _arrays(trace: Trace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, power, valid_mask, gap_times) in SI units."""
    times, values, mask, gap_times = [], [], [], []
    for s in trace.samples:
        if s.time_si is None:
            continue
        times.append(s.time_si)
        if s.is_gap or s.value_si is None:
            mask.append(False)
            values.append(np.nan)
            gap_times.append(s.time_si)
        else:
            mask.append(True)
            values.append(s.value_si)
    return (np.asarray(times, dtype=float), np.asarray(values, dtype=float),
            np.asarray(mask, dtype=bool), np.asarray(gap_times, dtype=float))


def default_windows(injection_times: list[float], trace_end: float,
                    trace_start: float = 0.0) -> list[Window]:
    windows: list[Window] = []
    for i, t in enumerate(injection_times):
        end = injection_times[i + 1] if i + 1 < len(injection_times) else trace_end
        start = max(t, trace_start)
        end = min(end, trace_end)
        windows.append(Window(i + 1, start, end, "default"))
    return windows


def _interval_integral(time: np.ndarray, power: np.ndarray, start: float,
                       end: float, label: str) -> tuple[float | None, int,
                                                        list[Diagnostic]]:
    """Trapezoid integral over [start, end], snapping values at endpoints."""
    diagnostics: list[Diagnostic] = []
    if end <= start:
        return None, 0, [diag("empty_window", "integration end <= start", label)]
    # Select interior samples.
    inside = (time > start) & (time < end)
    sel_t = np.concatenate([[start], time[inside], [end]])
    sel_v = np.empty_like(sel_t)
    sel_v[0] = np.interp(start, time, power)
    sel_v[-1] = np.interp(end, time, power)
    sel_v[1:-1] = power[inside]
    if np.isnan(sel_v).any():
        return None, int(inside.sum()) + 2, [
            diag("gap_crossed",
                 "integration window crosses a data gap; interpolation across "
                 "missing samples is forbidden", label,
                 suggestion="shorten the window to a gap-free segment"),
        ]
    area = float(np.trapezoid(sel_v, sel_t)) if hasattr(np, "trapezoid") \
        else float(np.trapz(sel_v, sel_t))
    return area, int(sel_t.size), diagnostics


def integrate_batch(
    power: Trace,
    injections: list,                  # models.Injection (SI-populated)
    *,
    baseline: Trace | None = None,
    blank_trace: Trace | None = None,
    blank_injections: list | None = None,
    baseline_policy: dict[str, Any] | None = None,
    blank_policy: dict[str, Any] | None = None,
    windows: list[Window] | None = None,
    discarded: set[int] | None = None,
    strict: bool = False,
) -> IntegrationBatch:
    """Integrate every injection. See module docstring for the hard rules."""
    discarded = discarded or set()
    batch_diags: list[Diagnostic] = []
    time, pwr, mask, gap_times = _arrays(power)
    if time.size < 2:
        raise SemanticUnresolvable(
            "power trace has fewer than two usable time samples",
            [diag("insufficient_samples", "need >= 2 time samples", "power trace")])
    t0, t1 = float(time[0]), float(time[-1])

    good_inj = [(i, inj) for i, inj in enumerate(injections)
                if inj.time_si is not None and inj.volume_si is not None]
    if len(good_inj) != len(injections):
        batch_diags.append(diag(
            "unconvertible_injection",
            "some injection times/volumes could not be converted; those "
            "injections are not integrated", "@injections"))

    auto_windows = default_windows([inj.time_si for _, inj in good_inj], t1, t0)
    if windows is None:
        use_windows = auto_windows
    else:
        use_windows = list(windows)

    # Baseline handling -------------------------------------------------
    b_time = b_value = None
    b_policy = baseline_policy or {}
    b_mode = b_policy.get("mode")
    if baseline is not None:
        b_time, b_value, _, _ = _arrays(baseline)
        b_mode = b_mode or "baseline_material"
    if b_mode is None:
        b_mode = "required"

    # Blank handling ----------------------------------------------------
    bl_policy = blank_policy or {}
    correspondence = bl_policy.get("correspondence", {})  # {"i": blank_index}
    bl_time = bl_pwr = None
    bl_areas: dict[int, float] = {}
    if blank_trace is not None:
        bl_time, bl_pwr, _, bl_gaps = _arrays(blank_trace)
        bl_inj = {inj.index: inj for inj in (blank_injections or [])
                  if inj.time_si is not None and inj.volume_si is not None}
        bl_times_sorted = sorted(inj.time_si for inj in bl_inj.values())
        bl_end = float(bl_time[-1])
        for blank_index, inj in bl_inj.items():
            order = bl_times_sorted.index(inj.time_si)
            w_end = (bl_times_sorted[order + 1]
                     if order + 1 < len(bl_times_sorted) else bl_end)
            area, _, ds = _interval_integral(
                bl_time, bl_pwr, inj.time_si, min(w_end, bl_end),
                f"blank injection #{blank_index}")
            if area is not None:
                bl_areas[blank_index] = area
            else:
                batch_diags.extend(ds)

    by_index = {inj.index: inj for _, inj in good_inj}
    points: list[IntegrationPoint] = []
    for pos, (inj_pos, inj) in enumerate(good_inj):
        label = f"injection #{inj.index}"
        win = use_windows[pos] if pos < len(use_windows) else auto_windows[pos]
        point_diags: list[dict[str, Any]] = []
        status = "ok"

        if inj.index in discarded:
            point_diags.append(diag(
                "first_injection_discarded",
                "injection excluded from modelling by a reviewer decision; "
                "raw integral still computed for the record", label,
                severity="warning").to_dict())
            status = "warning"

        # Baseline validity / subtraction.
        local_pwr = pwr
        if b_mode == "baseline_material" and b_time is not None:
            lo_lo = b_time[0] > win.start + 1e-9
            hi_hi = b_time[-1] < win.end - 1e-9
            if lo_lo or hi_hi:
                point_diags.append(diag(
                    "baseline_out_of_range",
                    "baseline curve does not span the integration window",
                    label, severity="warning",
                    suggestion="attach a baseline version covering this window"
                ).to_dict())
                status = "warning"
            else:
                base = np.interp(time, b_time, b_value)
                local_pwr = pwr - base
        elif b_mode == "none_declared":
            point_diags.append(diag(
                "baseline_none_declared",
                "no baseline subtraction (explicit reviewer declaration)",
                label, severity="warning",
                raw=b_policy.get("declared_by", "")).to_dict())
            if status == "ok":
                status = "warning"
        else:
            point_diags.append(diag(
                "baseline_required",
                "no baseline material and no explicit baseline policy; "
                "refusing to fabricate a default baseline", label,
                suggestion="attach the versioned baseline or declare "
                           "baseline mode none_declared").to_dict())
            points.append(IntegrationPoint(
                inj.index, win.start, win.end, None, float("nan"),
                status="rejected", dilution_rule=None,
                n_samples=0, diagnostics=point_diags))
            continue

        # Gap crossing: gap strictly inside the open interval is forbidden.
        crossed = [float(g) for g in gap_times
                   if win.start + 1e-9 < g < win.end - 1e-9]
        if crossed:
            point_diags.append(diag(
                "gap_crossed",
                "integration window crosses a data gap; endpoints may touch "
                "but the window must not span a missing sample",
                f"{label} t in ({win.start:g},{win.end:g})",
                suggestion="move the endpoint so the window stays on one side "
                           "of t=" + ", ".join(f"{g:g}s" for g in crossed)
            ).to_dict())
            points.append(IntegrationPoint(
                inj.index, win.start, win.end, None, float("nan"),
                status="rejected", n_samples=0, diagnostics=point_diags))
            continue

        area, n_samples, ds = _interval_integral(time, local_pwr,
                                                 win.start, win.end, label)
        if ds:
            point_diags.extend(d.to_dict() for d in ds)
            points.append(IntegrationPoint(
                inj.index, win.start, win.end, None, float("nan"),
                status="rejected", n_samples=n_samples,
                diagnostics=point_diags))
            continue

        # Dilution / blank subtraction under an explicit correspondence rule.
        blank_heat = None
        rule_name = None
        mapped = correspondence.get(str(inj.index),
                                    correspondence.get(inj.index))
        if bl_policy:
            rule_name = (f"{bl_policy.get('rule', 'blank-correspondence')}@"
                         f"{bl_policy.get('rule_version', '?')}/"
                         f"{bl_policy.get('method', 'unstated')}")
        if bl_policy and blank_trace is not None:
            if mapped is None:
                point_diags.append(diag(
                    "blank_correspondence_missing",
                    "no blank injection is declared to correspond to this "
                    "injection; dilution heat NOT subtracted and raw heat "
                    "is retained", label,
                    suggestion=f"add correspondence {inj.index}-><blank index> "
                               "with a rule version").to_dict())
                status = "warning" if status == "ok" else status
            else:
                bl_area = bl_areas.get(int(mapped))
                if bl_area is None:
                    point_diags.append(diag(
                        "blank_correspondence_target_invalid",
                        f"correspondence points at blank injection {mapped} "
                        "which has no integral (missing/gap)", label).to_dict())
                    status = "rejected"
                else:
                    ratio = inj.volume_si / bl_inj2_vol(blank_injections,
                                                        int(mapped))
                    blank_heat = bl_area * ratio
                    if abs(ratio - 1.0) > 1e-6:
                        point_diags.append(diag(
                            "blank_volume_scaled",
                            f"blank heat scaled by injection volume ratio "
                            f"{ratio:.6g}", label, severity="warning").to_dict())
                        if status == "ok":
                            status = "warning"
        elif bl_policy and mapped is not None:
            point_diags.append(diag(
                "blank_trace_missing",
                "correspondence declared but no blank curve attached",
                label).to_dict())
            status = "rejected"

        net = area
        if blank_heat is not None:
            net = area - blank_heat
        points.append(IntegrationPoint(
            injection_index=inj.index, start_si=win.start, end_si=win.end,
            heat_si=float(net) if net is not None else None,
            raw_heat_si=float(area), blank_heat_si=(float(blank_heat)
                                                    if blank_heat is not None
                                                    else None),
            dilution_rule=rule_name, n_samples=n_samples, status=status,
            diagnostics=point_diags))

    batch = IntegrationBatch(
        rule_version=INTEGRATION_RULE_VERSION, origin=power.origin,
        points=points, windows=use_windows,
        inputs={
            "power_origin": power.origin,
            "n_injections": len(injections),
            "baseline_mode": b_mode,
            "blank_rule": (f"{bl_policy.get('rule')}@"
                           f"{bl_policy.get('rule_version')}"
                           if bl_policy else None),
            "discarded": sorted(discarded),
        })
    batch.diagnostics = [d.to_dict() for d in batch_diags]
    if strict and any(p.status == "rejected" for p in points):
        rejected = [p.injection_index for p in points if p.status == "rejected"]
        raise SemanticUnresolvable(
            f"rejected injections: {rejected}",
            [diag("rejected_injections", "strict integration mode",
                  f"injections {rejected}")])
    return batch


def bl_inj2_vol(blank_injections, index: int) -> float:
    for inj in blank_injections or []:
        if inj.index == index:
            return inj.volume_si
    return 1.0
