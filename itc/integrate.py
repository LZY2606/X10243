"""Per-injection integration of ITC power curves.

Hard business rules implemented here (not in the browser):

* Post-injection peaks are allowed to overlap.  Each injection owns the
  integration window ``[t_i, t_{i+1})`` (or a tail window for the final
  injection); when the detected peak of injection ``i`` has not returned to
  baseline before injection ``i+1`` both injections are flagged
  ``overlapping_peaks`` but their windows remain independently integrable.
* An integration window never crosses a data gap.  Windows whose interior
  intersects a gap receive ``window_crosses_gap`` and an *invalid* status with
  no accepted area; windows that merely end exactly on a gap boundary are
  valid but carry a ``gap_at_endpoint`` diagnostic.
* Baselines are never extrapolated: baseline coverage must span the window.
* Dilution-heat subtraction states its correspondence rule explicitly
  (``by_index`` / ``by_volume`` / ``none``, plus a rule version).  A missing
  blank counterpart yields ``blank_match_missing`` and the injection stays
  un-normalized rather than silently subtracting zero.
* The first injection can be marked discarded; its area is retained and
  displayed but excluded from molar heats and model fitting.

Raw energy areas depend only on the power trace, baseline, blank and the
injection schedule + rule set; they do *not* depend on concentrations.
Molar normalization is a separately identified derived step, so a later
concentration revision invalidates models without rewriting the original
integral areas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import Diagnostic, SemanticImpossible, diag
from .hashing import canonical_json, stable_id
from .ingest import Material
from .units import CANONICAL, Quantity, quantity


BLANK_RULES = ("by_index", "by_volume", "none")


@dataclass(frozen=True)
class IntegrationRules:
    """Versioned rule set for an integration run."""

    rule_version: str = "rules/v1"
    tail_window_seconds: float = 180.0
    discard_first: bool = False
    discard_reason: str = "first injection mixing artifact"
    blank_correspondence: str = "by_index"
    peak_return_sigma: float = 3.0
    blank_tail_window_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.blank_correspondence not in BLANK_RULES:
            raise SemanticImpossible(
                f"unknown blank correspondence rule {self.blank_correspondence!r}",
                [diag(
                    "unknown_blank_rule",
                    f"must be one of {BLANK_RULES}",
                    path="rules.blank_correspondence",
                )],
            )
        if self.tail_window_seconds <= 0:
            raise SemanticImpossible(
                "tail_window_seconds must be positive",
                [diag("invalid_rule", "tail_window_seconds <= 0",
                      path="rules.tail_window_seconds")],
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "tail_window_seconds": self.tail_window_seconds,
            "discard_first": self.discard_first,
            "discard_reason": self.discard_reason,
            "blank_correspondence": self.blank_correspondence,
            "peak_return_sigma": self.peak_return_sigma,
            "blank_tail_window_seconds": self.blank_tail_window_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntegrationRules":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InjectionIntegral:
    index: int
    injection_id: str
    window_start: float
    window_end: float
    raw_energy: Quantity | None
    blank_energy: Quantity | None
    net_energy: Quantity | None
    molar_heat: Quantity | None
    injected_moles: Quantity | None
    status: str  # "valid" | "invalid" | "excluded"
    excluded: bool
    overlapping_peaks: bool
    peak_end: float | None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # Stable point identity: depends only on energy-bearing inputs + window.
    point_id: str = ""
    partial_energy: Quantity | None = None
    matched_blank_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        def q(value: Quantity | None) -> dict[str, Any] | None:
            return value.to_dict() if value is not None else None

        return {
            "index": self.index,
            "injection_id": self.injection_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window": {
                "unit": "s",
                "basis": "run_start",
                "start": self.window_start,
                "end": self.window_end,
            },
            "raw_energy": q(self.raw_energy),
            "blank_energy": q(self.blank_energy),
            "net_energy": q(self.net_energy),
            "molar_heat": q(self.molar_heat),
            "injected_moles": q(self.injected_moles),
            "status": self.status,
            "excluded": self.excluded,
            "overlapping_peaks": self.overlapping_peaks,
            "peak_end": self.peak_end,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "point_id": self.point_id,
            "partial_energy": q(self.partial_energy),
            "matched_blank_index": self.matched_blank_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InjectionIntegral":
        def q(value: dict[str, Any] | None) -> Quantity | None:
            return Quantity.from_dict(value) if value else None

        return cls(
            index=data["index"],
            injection_id=data["injection_id"],
            window_start=data["window_start"],
            window_end=data["window_end"],
            raw_energy=q(data.get("raw_energy")),
            blank_energy=q(data.get("blank_energy")),
            net_energy=q(data.get("net_energy")),
            molar_heat=q(data.get("molar_heat")),
            injected_moles=q(data.get("injected_moles")),
            status=data["status"],
            excluded=data.get("excluded", False),
            overlapping_peaks=data.get("overlapping_peaks", False),
            peak_end=data.get("peak_end"),
            diagnostics=[Diagnostic(**d) for d in data.get("diagnostics", [])],
            point_id=data.get("point_id", ""),
            partial_energy=q(data.get("partial_energy")),
            matched_blank_index=data.get("matched_blank_index"),
        )


@dataclass
class IntegrationResult:
    result_id: str
    rules: IntegrationRules
    injections: list[InjectionIntegral]
    inputs: dict[str, Any]
    parent_dependencies: dict[str, str]
    status: str  # "complete" | "partial" | "failed"
    diagnostics: list[Diagnostic] = field(default_factory=list)
    concentration_dependency: dict[str, str] | None = None
    normalization: dict[str, Any] | None = None
    # Energy-only identity: concentration revisions never alter this.
    energy_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "kind": "integration",
            "rules": self.rules.to_dict(),
            "injections": [i.to_dict() for i in self.injections],
            "inputs": self.inputs,
            "parent_dependencies": self.parent_dependencies,
            "status": self.status,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "concentration_dependency": self.concentration_dependency,
            "normalization": self.normalization,
            "energy_fingerprint": self.energy_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntegrationResult":
        return cls(
            result_id=data["result_id"],
            rules=IntegrationRules.from_dict(data["rules"]),
            injections=[InjectionIntegral.from_dict(i)
                        for i in data.get("injections", [])],
            inputs=data.get("inputs", {}),
            parent_dependencies=data.get("parent_dependencies", {}),
            status=data["status"],
            diagnostics=[Diagnostic(**d) for d in data.get("diagnostics", [])],
            concentration_dependency=data.get("concentration_dependency"),
            normalization=data.get("normalization"),
            energy_fingerprint=data.get("energy_fingerprint", ""),
        )

    def usable_points(self) -> list[InjectionIntegral]:
        return [
            i for i in self.injections
            if i.status == "valid" and not i.excluded and i.molar_heat is not None
        ]


# ---------------------------------------------------------------------------
# curve extraction
# ---------------------------------------------------------------------------


def _series(curve: Material) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Return ``(times, values, gaps)`` with gaps as canonical-second intervals.

    Missing/unreadable values become NaN in ``values`` (their raw tokens remain
    on the material); gaps come from both missing runs and detected time jumps.
    """

    times = np.array(
        [s.t.value for s in curve.samples if s.t.value is not None], dtype=float
    )
    values = np.full(len(times), np.nan, dtype=float)
    j = 0
    for sample in curve.samples:
        if sample.t.value is None:
            continue
        if not sample.missing and sample.y is not None and sample.y.valid:
            values[j] = sample.y.value
        j += 1
    gaps = [(g.start, g.end) for g in curve.gaps]
    return times, values, gaps


def _gap_relation(
    start: float, end: float, gaps: list[tuple[float, float]]
) -> tuple[str, list[Diagnostic]]:
    """Classify a window against gaps.

    Returns a relation of ``clear`` / ``endpoint`` / ``crosses``.
    Touching exactly at a window boundary is an *endpoint* condition, not a
    crossing: integral windows never span a missing interior, but data that
    resumes exactly at the window edge is integrable and is reported.
    """

    diagnostics: list[Diagnostic] = []
    relation = "clear"
    for gap_start, gap_end in gaps:
        lo = max(start, gap_start)
        hi = min(end, gap_end)
        if lo < hi - 1e-12:
            return "crosses", [diag(
                "window_crosses_gap",
                f"window [{start:g}, {end:g}] s crosses data gap "
                f"[{gap_start:g}, {gap_end:g}] s; no integral accepted",
                path="integration_window",
            )]
        if abs(lo - hi) <= 1e-12 and (
            abs(end - gap_start) <= 1e-12 or abs(start - gap_end) <= 1e-12
        ):
            relation = "endpoint"
            diagnostics.append(diag(
                "gap_at_endpoint",
                f"window boundary coincides with gap edge "
                f"[{gap_start:g}, {gap_end:g}] s",
                path="integration_window",
            ))
    return relation, diagnostics


def _baseline_values(
    times: np.ndarray,
    window_start: float,
    window_end: float,
    baseline: Material | None,
    baseline_gaps: list[tuple[float, float]],
) -> tuple[np.ndarray | None, list[Diagnostic]]:
    """Interpolate baseline onto sample times; never extrapolate."""

    diagnostics: list[Diagnostic] = []
    if baseline is None:
        diagnostics.append(diag(
            "baseline_missing",
            "no baseline material supplied; integration cannot proceed",
            path="baseline",
        ))
        return None, diagnostics
    b_times, b_values, b_jump_gaps = _series(baseline)
    all_gaps = sorted(set(baseline_gaps + b_jump_gaps))
    finite = np.isfinite(b_values)
    if not finite.any():
        diagnostics.append(diag(
            "baseline_unusable",
            "baseline has no finite samples",
            path="baseline",
        ))
        return None, diagnostics
    valid_t = b_times[finite]
    valid_v = b_values[finite]
    if valid_t[0] > window_start + 1e-9 or valid_t[-1] < window_end - 1e-9:
        diagnostics.append(diag(
            "baseline_out_of_range",
            f"baseline [{valid_t[0]:g}, {valid_t[-1]:g}] s does not span window "
            f"[{window_start:g}, {window_end:g}] s; no extrapolation",
            path="baseline",
        ))
        return None, diagnostics
    relation, gap_diags = _gap_relation(window_start, window_end, all_gaps)
    if relation == "crosses":
        diagnostics.extend(gap_diags)
        diagnostics[-1] = diag(
            "baseline_crosses_gap",
            diagnostics[-1].message.replace("window", "baseline"),
            path="baseline",
        )
        return None, diagnostics
    diagnostics.extend(gap_diags)
    return np.interp(times, valid_t, valid_v), diagnostics


def _integrate_window(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    baseline: Material | None,
    gaps: list[tuple[float, float]],
    *,
    path_prefix: str,
) -> tuple[float | None, float | None, str, list[Diagnostic]]:
    """Trapezoid area of baseline-subtracted power over one window.

    Returns ``(energy, partial_energy, relation, diagnostics)`` where relation
    is ``clear`` / ``endpoint`` / ``crosses`` / ``beyond_trace`` /
    ``no_coverage``.
    """

    diagnostics: list[Diagnostic] = []
    relation, gap_diags = _gap_relation(start, end, gaps)
    diagnostics.extend(
        diag(d.code, d.message, path=f"{path_prefix}:{d.path}", ref=d.ref)
        for d in gap_diags
    )
    if relation == "crosses":
        return None, None, "crosses", diagnostics

    mask = (times >= start - 1e-12) & (times < end - 1e-9) | (
        np.isclose(times, end, atol=1e-12)
    )
    window_t = times[mask]
    window_v = values[mask]
    if window_t.size == 0:
        diagnostics.append(diag(
            "window_no_coverage",
            f"no power samples inside [{start:g}, {end:g}] s",
            path=f"{path_prefix}:integration_window",
        ))
        return None, None, "no_coverage", diagnostics

    last_t = times[-1]
    if last_t < end - 1e-9:
        diagnostics.append(diag(
            "window_end_beyond_trace",
            f"window ends at {end:g} s but trace ends at {last_t:g} s",
            path=f"{path_prefix}:integration_window",
        ))
        relation = "beyond_trace"

    base_values, base_diags = _baseline_values(
        window_t, start, min(end, last_t), baseline, gaps
    )
    for d in base_diags:
        diagnostics.append(
            diag(d.code, d.message, path=f"{path_prefix}:{d.path}", ref=d.ref)
        )
    if base_values is None:
        return None, None, relation, diagnostics

    finite = np.isfinite(window_v)
    corrected = window_v - base_values
    if not finite.all():
        # Interior NaNs should already be represented as gaps; guard anyway.
        diagnostics.append(diag(
            "window_contains_nan",
            "window contains unreadable samples not captured as gaps",
            path=f"{path_prefix}:integration_window",
        ))
        return None, None, "crosses", diagnostics
    area = float(np.trapezoid(corrected, window_t))
    if relation == "beyond_trace":
        return None, area, relation, diagnostics
    return area, area, relation, diagnostics


def _peak_end(
    times: np.ndarray,
    corrected_curve: np.ndarray,
    start: float,
    end: float,
    sigma: float,
) -> float | None:
    """Estimate when a post-injection peak returns to baseline.

    The signal is baseline-subtracted over the run; a peak ends at the first
    time after its maximum where the absolute signal stays within ``sigma``
    robust noise estimates through the rest of the window.  Purely
    deterministic (no random sampling) so equivalent inputs replay identically.
    """

    mask = (times >= start - 1e-12) & (times <= end + 1e-12)
    wt = times[mask]
    wv = corrected_curve[mask]
    finite = np.isfinite(wv)
    if not finite.any() or wt.size < 3:
        return None
    abs_wv = np.abs(wv[finite])
    peak_idx = int(np.argmax(abs_wv))
    peak_amp = abs_wv[peak_idx]
    if peak_amp <= 0:
        return float(wt[peak_idx])
    # Robust noise scale from the last 20% of the window.
    tail_count = max(3, int(0.2 * wt.size))
    tail = np.abs(wv[finite][-tail_count:])
    noise = float(np.median(tail)) if tail.size else 0.0
    threshold = max(noise * sigma, peak_amp * 0.02)
    after_peak = wt[finite][peak_idx:]
    after_v = wv[finite][peak_idx:]
    settled = np.abs(after_v) <= threshold
    # Require staying settled until window end (a later re-excursion means the
    # peak has not truly ended).
    settled_through_end = np.logical_and.accumulate(settled[::-1])[::-1]
    hits = np.flatnonzero(settled_through_end)
    if hits.size == 0:
        return None
    return float(after_peak[hits[0]])


def _match_blank(
    injection: dict[str, Any],
    injection_index: int,
    blank: Material | None,
    rule: str,
) -> tuple[int | None, Diagnostic | None]:
    if rule == "none" or blank is None:
        return None, None
    if rule == "by_index":
        if injection_index < len(blank.injections):
            return injection_index, None
        return None, diag(
            "blank_match_missing",
            f"no blank injection at index {injection_index} "
            f"(blank has {len(blank.injections)} injection(s)); "
            "dilution heat left un-subtracted under rule by_index",
            path=f"blank:injection_{injection_index}",
            ref="by_index",
        )
    if rule == "by_volume":
        target = injection["volume"]["value"]
        if target is None:
            return None, diag(
                "blank_match_missing",
                "injection volume unconvertible; cannot match blank by volume",
                path=f"blank:injection_{injection_index}",
                ref="by_volume",
            )
        candidates = [
            (i, b["volume"]["value"])
            for i, b in enumerate(blank.injections)
            if b["volume"]["value"] is not None
        ]
        if not candidates:
            return None, diag(
                "blank_match_missing",
                "blank schedule has no convertible volumes to match",
                path=f"blank:injection_{injection_index}",
                ref="by_volume",
            )
        # Deterministic tie-break: nearest volume, then lowest blank index.
        best_i, best_diff = min(
            ((i, abs(v - target)) for i, v in candidates),
            key=lambda pair: (pair[1], pair[0]),
        )
        relative = best_diff / target if target else float("inf")
        if relative > 0.02:
            return None, diag(
                "blank_match_missing",
                f"no blank injection within 2% of volume {target:g} L "
                f"(best relative gap {relative:.3g}) under rule by_volume",
                path=f"blank:injection_{injection_index}",
                ref="by_volume",
            )
        return best_i, None
    return None, diag(
        "unknown_blank_rule",
        f"unsupported blank correspondence rule {rule!r}",
        path="rules.blank_correspondence",
    )


def _blank_windows(blank: Material, rules: IntegrationRules) -> list[tuple[float, float]]:
    times = [b["time"]["value"] for b in blank.injections if b["time"]["value"] is not None]
    windows: list[tuple[float, float]] = []
    for i, t in enumerate(times):
        if i + 1 < len(times):
            windows.append((t, times[i + 1]))
        else:
            windows.append((t, t + rules.blank_tail_window_seconds))
    return windows


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def integrate_injections(
    *,
    injections: Material,
    power: Material,
    baseline: Material | None,
    blank: Material | None = None,
    rules: IntegrationRules | None = None,
    material_refs: dict[str, str] | None = None,
) -> IntegrationResult:
    """Integrate every injection window under a single versioned rule set.

    ``material_refs`` maps logical inputs (``injections``, ``power``,
    ``baseline``, ``blank``) to stored material version ids so the result can
    record exact parent dependencies.
    """

    rules = rules or IntegrationRules()
    refs = dict(material_refs or {})

    schedule: list[tuple[float, float, str]] = []
    schedule_diags: list[Diagnostic] = []
    for i, inj in enumerate(injections.injections):
        t_q = inj["time"]
        v_q = inj["volume"]
        t = t_q.get("value") if isinstance(t_q, dict) else t_q.value
        v = v_q.get("value") if isinstance(v_q, dict) else v_q.value
        if t is None or v is None:
            schedule_diags.append(diag(
                "injection_not_integrable",
                f"injection {inj.get('id', i)} has unconvertible time or volume",
                path=f"{injections.source_name}:row{i + 1}",
            ))
            continue
        schedule.append((float(t), float(v), str(inj["id"])))

    times, values, gaps = _series(power)
    corrected_curve = values.copy()
    # Baseline-subtracted full-trace view for peak-end detection.
    if baseline is not None:
        b_times, b_values, _ = _series(baseline)
        finite = np.isfinite(b_values)
        if finite.any():
            interp = np.interp(
                times,
                b_times[finite],
                b_values[finite],
                left=np.nan,
                right=np.nan,
            )
            with np.errstate(invalid="ignore"):
                corrected_curve = values - interp

    blank_windows = _blank_windows(blank, rules) if blank is not None else []
    _, blank_values, blank_gaps = _series(blank) if blank is not None else (
        np.array([]), np.array([]), []
    )

    results: list[InjectionIntegral] = []
    any_invalid = False
    all_invalid = True

    for i, (t_inj, vol, inj_id) in enumerate(schedule):
        path_prefix = f"injection[{inj_id}]"
        if i + 1 < len(schedule):
            window_end = schedule[i + 1][0]
        else:
            window_end = t_inj + rules.tail_window_seconds

        energy, partial, relation, window_diags = _integrate_window(
            times, values, t_inj, window_end, baseline, gaps,
            path_prefix=path_prefix,
        )
        integral = InjectionIntegral(
            index=i,
            injection_id=inj_id,
            window_start=float(t_inj),
            window_end=float(window_end),
            raw_energy=None,
            blank_energy=None,
            net_energy=None,
            molar_heat=None,
            injected_moles=None,
            status="invalid",
            excluded=False,
            overlapping_peaks=False,
            peak_end=None,
            diagnostics=list(window_diags),
            partial_energy=(
                quantity(partial, "energy") if partial is not None else None
            ),
        )

        # Overlap detection (energy still computed for both windows).
        peak_end = _peak_end(
            times, corrected_curve, t_inj, window_end, rules.peak_return_sigma
        )
        integral.peak_end = peak_end
        if peak_end is not None and i + 1 < len(schedule):
            if peak_end > schedule[i + 1][0] + 1e-9:
                integral.overlapping_peaks = True
                integral.diagnostics.append(diag(
                    "overlapping_peaks",
                    f"peak settles at {peak_end:g} s, after next injection at "
                    f"{schedule[i + 1][0]:g} s; window boundary is still honored",
                    path=f"{path_prefix}:peak",
                ))

        # Blank correspondence (explicit rule, missing counterpart diagnosed).
        matched_index, blank_diag = _match_blank(
            injections.injections[i], i, blank, rules.blank_correspondence
        )
        blank_energy: float | None = None
        if blank_diag is not None:
            integral.diagnostics.append(blank_diag)
        elif matched_index is not None and matched_index < len(blank_windows):
            b_start, b_end = blank_windows[matched_index]
            b_energy, _b_partial, b_relation, b_diags = _integrate_window(
                np.array(
                    [b["time"]["value"] for b in blank.injections
                     if b["time"]["value"] is not None]
                ) if False else _blank_full_times(blank),
                blank_values,
                b_start,
                b_end,
                baseline,
                blank_gaps,
                path_prefix=f"{path_prefix}:blank",
            )
            for d in b_diags:
                integral.diagnostics.append(d)
            if b_energy is not None:
                blank_energy = b_energy

        if energy is None:
            any_invalid = True
            integral.status = "invalid"
        else:
            all_invalid = False
            integral.raw_energy = quantity(energy, "energy")
            if blank_energy is not None:
                integral.blank_energy = quantity(blank_energy, "energy")
                integral.net_energy = quantity(energy - blank_energy, "energy")
            elif rules.blank_correspondence != "none" and blank is not None:
                # Missing counterpart: keep raw energy, mark un-normalized.
                integral.net_energy = quantity(energy, "energy")
            else:
                integral.net_energy = quantity(energy, "energy")
            integral.status = "valid"

        if rules.discard_first and i == 0:
            integral.excluded = True
            integral.status = "excluded" if integral.raw_energy is not None else "invalid"
            integral.diagnostics.append(diag(
                "first_injection_discarded",
                rules.discard_reason,
                path=f"{path_prefix}:policy",
                ref=rules.rule_version,
            ))

        results.append(integral)

    # Stable per-point identity: energy inputs only (no concentrations).
    point_inputs = {
        "rule_version": rules.rule_version,
        "blank_correspondence": rules.blank_correspondence,
        "tail_window_seconds": rules.tail_window_seconds,
        "discard_first": rules.discard_first,
        "refs": {k: refs.get(k) for k in ("injections", "power", "baseline", "blank")},
    }
    for integral in results:
        payload = dict(point_inputs)
        payload["window"] = [integral.window_start, integral.window_end]
        payload["injection_id"] = integral.injection_id
        integral.point_id = stable_id("point", payload)

    overall_status = "complete"
    if all_invalid:
        overall_status = "failed"
    elif any_invalid or schedule_diags:
        overall_status = "partial"

    energy_fp_payload = {
        "rules": {
            k: v for k, v in rules.to_dict().items()
            if k in (
                "rule_version", "tail_window_seconds", "discard_first",
                "blank_correspondence", "blank_tail_window_seconds",
            )
        },
        "refs": {k: refs.get(k) for k in ("injections", "power", "baseline", "blank")},
        "windows": [[r.window_start, r.window_end] for r in results],
    }
    energy_fingerprint = stable_id("energy", energy_fp_payload)

    result = IntegrationResult(
        result_id=stable_id("intg", energy_fp_payload),
        rules=rules,
        injections=results,
        inputs={
            "rules": rules.to_dict(),
            "materials": {
                k: {
                    "ref": refs.get(k),
                    "kind": m.kind if m else None,
                    "source_name": m.source_name if m else None,
                }
                for k, m in (
                    ("injections", injections),
                    ("power", power),
                    ("baseline", baseline),
                    ("blank", blank),
                )
            },
        },
        parent_dependencies={k: v for k, v in refs.items() if v},
        status=overall_status,
        diagnostics=list(schedule_diags),
        energy_fingerprint=energy_fingerprint,
    )
    return result


def _blank_full_times(blank: Material) -> np.ndarray:
    return np.array(
        [s.t.value for s in blank.samples if s.t.value is not None], dtype=float
    )
