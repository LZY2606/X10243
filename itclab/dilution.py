"""Blank correspondence and dilution-heat subtraction.

The rule that pairs sample injections with blank injections is part of the
derived record, never an implicit assumption. Two rules are supported:

* ``same_index``  - blank injection k corresponds to sample injection k;
* ``nearest_time`` - blank injection whose start is closest and within
  ``time_tolerance_s``.

A sample injection without a corresponding blank injection is kept, flagged
``blank_missing`` and excluded from fitting with a position-bearing
diagnostic. Blank areas that are themselves invalid also fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic


@dataclass(frozen=True)
class DilutionParams:
    rule: str = "same_index"  # same_index | nearest_time
    time_tolerance_s: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "time_tolerance_s": self.time_tolerance_s}


def _diag(code: str, message: str, **where: Any) -> dict[str, Any]:
    return Diagnostic(code, message, where).to_dict()


def subtraction(
    sample_set: dict[str, Any],
    blank_set: dict[str, Any],
    params: DilutionParams,
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    blank_results = blank_set["results"]
    by_index = {r["injection_index"]: r for r in blank_results}

    per_injection: list[dict[str, Any]] = []
    for r in sample_set["results"]:
        idx = r["injection_index"]
        match: dict[str, Any] | None = None
        rule_detail: dict[str, Any] = {"rule": params.rule}

        if params.rule == "same_index":
            match = by_index.get(idx)
        elif params.rule == "nearest_time":
            candidates = [
                (abs(b["injection_time_s"] - r["injection_time_s"]), b)
                for b in blank_results
                if abs(b["injection_time_s"] - r["injection_time_s"]) <= params.time_tolerance_s
            ]
            candidates.sort(key=lambda x: (x[0], x[1]["injection_index"]))
            if candidates:
                match = candidates[0][1]
                rule_detail["time_delta_s"] = candidates[0][0]
        else:
            diagnostics.append(
                _diag("UNKNOWN_CORRESPONDENCE_RULE", f"rule {params.rule!r} is not supported", rule=params.rule)
            )
            break

        entry: dict[str, Any] = {
            "injection_index": idx,
            "injection_time_s": r["injection_time_s"],
            "injection_volume_uL": r["injection_volume_uL"],
            "sample_area_uJ": r["area_uJ"],
            "sample_status": r["status"],
            "discarded": r["discarded"],
            "overlapping_peaks": r["overlapping_peaks"],
            "correspondence": rule_detail,
        }

        if r["status"] != "valid":
            entry.update(
                {
                    "blank_injection_index": match["injection_index"] if match else None,
                    "blank_area_uJ": match["area_uJ"] if match else None,
                    "corrected_area_uJ": None,
                    "status": "sample_invalid",
                }
            )
            entry["diagnostics"] = list(r.get("diagnostics", []))
        elif match is None:
            entry.update(
                {
                    "blank_injection_index": None,
                    "blank_area_uJ": None,
                    "corrected_area_uJ": None,
                    "status": "blank_missing",
                    "diagnostics": [
                        _diag(
                            "BLANK_CORRESPONDENCE_MISSING",
                            f"no blank injection corresponds to sample injection {idx} "
                            f"under rule {params.rule!r}",
                            injection=idx,
                            rule=params.rule,
                        )
                    ],
                }
            )
            diagnostics.extend(entry["diagnostics"])
        elif match["status"] != "valid":
            entry.update(
                {
                    "blank_injection_index": match["injection_index"],
                    "blank_area_uJ": None,
                    "corrected_area_uJ": None,
                    "status": "blank_invalid",
                    "diagnostics": [
                        _diag(
                            "BLANK_INTEGRAL_INVALID",
                            f"matched blank injection {match['injection_index']} has no valid integral",
                            injection=idx,
                            blank_injection=match["injection_index"],
                        )
                    ],
                }
            )
            diagnostics.extend(entry["diagnostics"])
        else:
            corrected = r["area_uJ"] - match["area_uJ"]
            entry.update(
                {
                    "blank_injection_index": match["injection_index"],
                    "blank_area_uJ": match["area_uJ"],
                    "corrected_area_uJ": corrected,
                    "status": "corrected",
                    "diagnostics": [],
                }
            )
        per_injection.append(entry)

    return {
        "rule": params.rule,
        "params": params.to_dict(),
        "per_injection": per_injection,
        "diagnostics": diagnostics,
        "summary": {
            "n": len(per_injection),
            "n_corrected": sum(e["status"] == "corrected" for e in per_injection),
            "n_blank_missing": sum(e["status"] == "blank_missing" for e in per_injection),
            "n_blank_invalid": sum(e["status"] == "blank_invalid" for e in per_injection),
        },
    }
