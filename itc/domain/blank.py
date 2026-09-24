"""Dilution-heat (blank) subtraction.

Blank correspondence MUST be declared. When no blank injection corresponds to
a sample injection the dilution correction is unavailable: the net heat is
reported as not computable with a positioned diagnostic instead of silently
assuming zero dilution heat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MATCH_BY_ORDINAL = "BLANK_MATCH_BY_ORDINAL"
MATCH_BY_VOLUME = "BLANK_MATCH_BY_VOLUME"
NO_BLANK = "BLANK_NOT_APPLIED"

RULES = (MATCH_BY_ORDINAL, MATCH_BY_VOLUME)


@dataclass
class BlankParams:
    version: str
    rule: str = MATCH_BY_ORDINAL
    scale_by_volume: bool = True


@dataclass
class NetHeat:
    injection_index: int
    observed_ucal: float
    blank_ucal: float
    net_ucal: float
    matched_blank_index: int | None
    rule: str
    valid: bool
    diagnostics: list[dict] = field(default_factory=list)
    stable_id: str = ""

    def to_dict(self) -> dict:
        return {
            "stable_id": self.stable_id,
            "injection_index": self.injection_index,
            "observed_ucal": _f(self.observed_ucal),
            "blank_ucal": _f(self.blank_ucal),
            "net_ucal": _f(self.net_ucal),
            "matched_blank_index": self.matched_blank_index,
            "rule": self.rule,
            "valid": self.valid,
            "diagnostics": self.diagnostics,
        }


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def subtract(sample_results: list, blank_results: list,
             sample_injections: list, blank_injections: list | None,
             params: BlankParams) -> list[NetHeat]:
    sample_by = {r.injection_index: r for r in sample_results}
    blank_by = {r.injection_index: r for r in blank_results}
    blank_injections = blank_injections or []
    out: list[NetHeat] = []
    for inj in sample_injections:
        idx = inj.index
        res = sample_by.get(idx)
        observed = res.heat_ucal if res else math.nan
        nh = NetHeat(idx, observed, math.nan, math.nan, None, params.rule,
                     False)
        if res is None or not res.valid:
            nh.diagnostics.append({
                "code": "SAMPLE_INTEGRAL_INVALID",
                "message": "sample integral unavailable; blank not applied",
            })
            out.append(nh); continue
        match_idx = _match(idx, inj, sample_injections, blank_injections,
                           params.rule)
        if match_idx is None:
            nh.rule = params.rule
            nh.diagnostics.append({
                "code": "BLANK_CORRESPONDENCE_MISSING",
                "message": (f"no blank injection corresponds to sample "
                            f"injection {idx} under rule {params.rule}"),
                "location": {"injection_index": idx},
            })
            out.append(nh); continue
        bres = blank_by.get(match_idx)
        if bres is None or not bres.valid:
            nh.matched_blank_index = match_idx
            nh.diagnostics.append({
                "code": "BLANK_INTEGRAL_INVALID",
                "message": (f"matched blank injection {match_idx} has no valid "
                            "integral"),
                "location": {"injection_index": idx,
                             "blank_injection_index": match_idx},
            })
            out.append(nh); continue
        bheat = bres.heat_ucal
        if params.scale_by_volume:
            bvol = next((b.volume_uL for b in blank_injections
                         if b.index == match_idx), math.nan)
            if bvol and not math.isnan(bvol) and not math.isnan(inj.volume_uL):
                bheat = bheat * (inj.volume_uL / bvol)
        nh.blank_ucal = bheat
        nh.net_ucal = observed - bheat
        nh.matched_blank_index = match_idx
        nh.valid = True
        out.append(nh)
    return out


def _match(idx, inj, sample_injections, blank_injections, rule):
    if rule == MATCH_BY_ORDINAL:
        sample_ord = [i.index for i in sample_injections].index(idx)
        ordered = sorted(blank_injections,
                         key=lambda b: (b.nominal_time_s, b.index))
        return ordered[sample_ord].index if sample_ord < len(ordered) else None
    if rule == MATCH_BY_VOLUME:
        # nearest volume, requires <=5% relative difference
        best, best_diff = None, math.inf
        for b in blank_injections:
            if math.isnan(b.volume_uL) or inj.volume_uL == 0:
                continue
            rel = abs(b.volume_uL - inj.volume_uL) / inj.volume_uL
            if rel < best_diff:
                best, best_diff = b.index, rel
        return best if best_diff <= 0.05 else None
    return None
