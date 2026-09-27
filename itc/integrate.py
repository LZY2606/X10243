"""Injection window planning and peak integration.

Domain rules implemented here (each emits machine-readable evidence):

1. Post-injection peaks MAY overlap. Windows tile the time axis at the
   next injection start; a window whose peak physically overlaps the
   next one is flagged ``overlap_next`` instead of being rejected.
2. An integration window NEVER crosses a data gap. If the default
   window would cross a gap it is clamped at the gap start and flagged
   ``clamped_by_gap``; if the injection point itself lies inside a gap
   the injection is semantically unsolvable.
3. Dilution-heat (blank) subtraction must state its correspondence
   rule. A missing blank counterpart is a diagnostic + status, never a
   silently zero-filled value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .errors import Diagnostic, DiagBag, SemanticUnsolvable

# Peaks whose spacing is below this many decay constants are reported
# as physically overlapping (still legal, see rule 1).
OVERLAP_TAU_S = 75.0

BLANK_RULE_SAME_INDEX = "same_injection_index"
BLANK_RULES = (BLANK_RULE_SAME_INDEX,)


@dataclass
class Gap:
    start_s: float
    end_s: float
    reason: str = "unspecified"

    def to_dict(self):
        return {"start_s": self.start_s, "end_s": self.end_s,
                "reason": self.reason}


@dataclass
class WindowPlan:
    index: int
    start_s: float
    end_s: float
    overlap_next: bool = False
    clamped_by_gap: bool = False
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def to_dict(self):
        return {
            "index": self.index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "overlap_next": self.overlap_next,
            "clamped_by_gap": self.clamped_by_gap,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def plan_windows(inj_times_s: Sequence[float], gaps: Sequence[Gap],
                 trace_end_s: float, strict: bool = True) -> List[WindowPlan]:
    """Plan integration windows. Raises SemanticUnsolvable (with the
    injection position) when an injection point lies inside a gap."""
    plans: List[WindowPlan] = []
    n = len(inj_times_s)
    for i, t in enumerate(inj_times_s):
        pos = f"injection[{i}]"
        end = inj_times_s[i + 1] if i + 1 < n else trace_end_s
        if end <= t:
            raise SemanticUnsolvable(
                f"injection[{i}] window is empty (start={t}, end={end})",
                position=pos)
        clamped = False
        diags: List[Diagnostic] = []
        in_gap = None
        for g in gaps:
            if g.start_s <= t < g.end_s:
                in_gap = g
                break
        if in_gap is not None:
            err = SemanticUnsolvable(
                f"injection[{i}] at t={t}s lies inside data gap "
                f"[{in_gap.start_s}, {in_gap.end_s}] ({in_gap.reason})",
                position=pos, details={"gap": in_gap.to_dict()})
            if strict:
                raise err
            plans.append(WindowPlan(
                index=i, start_s=float(t), end_s=float(t),
                diagnostics=[Diagnostic(code="window.injection_in_gap",
                                        message=err.message, position=pos,
                                        severity="error")]))
            continue
        for g in gaps:
            if t < g.start_s < end:
                end = g.start_s
                clamped = True
                diags.append(Diagnostic(
                    code="window.clamped_by_gap",
                    message=(f"window for injection[{i}] clamped at gap "
                             f"start {g.start_s}s ({g.reason}); windows "
                             f"never cross data gaps"),
                    position=pos))
        overlap = (i + 1 < n) and (inj_times_s[i + 1] - t) < OVERLAP_TAU_S
        if overlap:
            diags.append(Diagnostic(
                code="window.overlap_next",
                message=(f"injection[{i}] peak overlaps injection[{i+1}] "
                         f"(spacing {inj_times_s[i+1]-t:.1f}s < "
                         f"{OVERLAP_TAU_S}s); overlapping post-injection "
                         f"peaks are legal, windows tile at the next "
                         f"injection start"),
                position=pos, severity="info"))
        plans.append(WindowPlan(index=i, start_s=float(t), end_s=float(end),
                                overlap_next=overlap, clamped_by_gap=clamped,
                                diagnostics=diags))
    return plans


@dataclass
class IntegralResult:
    index: int
    heat_uJ: Optional[float]
    n_samples: int
    status: str = "ok"               # ok | unsolvable
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def to_dict(self):
        return {"index": self.index, "heat_uJ": self.heat_uJ,
                "n_samples": self.n_samples, "status": self.status,
                "diagnostics": [d.to_dict() for d in self.diagnostics]}


def integrate_windows(t_s: Sequence[float], power_uJ_s: Sequence[float],
                      base_t_s: Sequence[float], base_uJ_s: Sequence[float],
                      plans: Sequence[WindowPlan],
                      strict: bool = False) -> List[IntegralResult]:
    """Baseline-subtracted trapezoid integration per window.

    strict=True raises SemanticUnsolvable on the first unsolvable
    injection; otherwise the failure is captured in the result status.
    """
    t = np.asarray(t_s, dtype=float)
    p = np.asarray(power_uJ_s, dtype=float)
    bt = np.asarray(base_t_s, dtype=float)
    bp = np.asarray(base_uJ_s, dtype=float)
    results: List[IntegralResult] = []
    for plan in plans:
        pos = f"injection[{plan.index}]"
        mask = (t >= plan.start_s) & (t < plan.end_s)
        n_samples = int(mask.sum())
        if n_samples < 2:
            err = SemanticUnsolvable(
                f"injection[{plan.index}] window "
                f"[{plan.start_s}, {plan.end_s}) contains {n_samples} "
                f"power samples; need >= 2 to integrate",
                position=pos)
            if strict:
                raise err
            results.append(IntegralResult(
                index=plan.index, heat_uJ=None, n_samples=n_samples,
                status="unsolvable",
                diagnostics=[Diagnostic(code="integral.unsolvable",
                                        message=err.message,
                                        position=pos, severity="error")]))
            continue
        base_here = np.interp(t[mask], bt, bp)
        heat = float(np.trapezoid(p[mask] - base_here, t[mask]))
        results.append(IntegralResult(
            index=plan.index, heat_uJ=heat, n_samples=n_samples,
            diagnostics=list(plan.diagnostics)))
    return results


@dataclass
class BlankCorrection:
    index: int
    heat_uJ: Optional[float]
    blank_heat_uJ: Optional[float]
    heat_corrected_uJ: Optional[float]
    status: str                      # corrected | unsubtracted | unsolvable
    rule: str
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def to_dict(self):
        return {"index": self.index, "heat_uJ": self.heat_uJ,
                "blank_heat_uJ": self.blank_heat_uJ,
                "heat_corrected_uJ": self.heat_corrected_uJ,
                "status": self.status, "rule": self.rule,
                "diagnostics": [d.to_dict() for d in self.diagnostics]}


def apply_blank(results: Sequence[IntegralResult],
                blank_heats_uJ: dict, rule: str = BLANK_RULE_SAME_INDEX,
                blank_ref: Optional[str] = None) -> List[BlankCorrection]:
    """Subtract blank (dilution heat) per the stated correspondence rule.

    Missing blank counterpart -> status 'unsubtracted' + diagnostic; the
    raw integral is preserved and never replaced by a default.
    """
    if rule not in BLANK_RULES:
        raise SemanticUnsolvable(
            f"unknown blank correspondence rule {rule!r}; "
            f"supported: {list(BLANK_RULES)}")
    out: List[BlankCorrection] = []
    for r in results:
        pos = f"injection[{r.index}]"
        if r.status != "ok":
            out.append(BlankCorrection(
                index=r.index, heat_uJ=None, blank_heat_uJ=None,
                heat_corrected_uJ=None, status="unsolvable", rule=rule,
                diagnostics=list(r.diagnostics)))
            continue
        blank = blank_heats_uJ.get(r.index)
        if blank is None:
            out.append(BlankCorrection(
                index=r.index, heat_uJ=r.heat_uJ, blank_heat_uJ=None,
                heat_corrected_uJ=None, status="unsubtracted", rule=rule,
                diagnostics=list(r.diagnostics) + [Diagnostic(
                    code="blank.counterpart_missing",
                    message=(f"blank rule '{rule}' has no counterpart for "
                             f"injection[{r.index}] in {blank_ref or 'blank'}; "
                             f"integral kept unsubtracted"),
                    position=pos)]))
            continue
        out.append(BlankCorrection(
            index=r.index, heat_uJ=r.heat_uJ, blank_heat_uJ=blank,
            heat_corrected_uJ=r.heat_uJ - blank, status="corrected",
            rule=rule, diagnostics=list(r.diagnostics)))
    return out
