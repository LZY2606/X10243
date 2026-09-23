"""Blank-control correspondence and dilution-heat subtraction.

The dilution-heat subtraction must state its *correspondence rule*. Two rules
are supported and recorded on every result:

* ``matched`` - injection i uses the blank integrated for the same injection
  (optionally scaled by injected-volume ratio);
* ``none``    - no subtraction. A usable injection with no counterpart under
  ``matched`` is reported ``blank_missing`` (diagnostic), never silently zero.
"""

from __future__ import annotations

from typing import Any

from .canonical import stable_float
from .errors import diagnostic

USABLE = ("ok", "clipped_overlap")


def integrate_blanks(blank_runs: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index already-computed blank integration runs by injection number."""
    return dict(blank_runs)


def apply_blank_rule(integration: dict[str, Any],
                     blanks_by_injection: dict[int, dict[str, Any]],
                     rule: str = "matched",
                     *, scale_by_volume: bool = True) -> dict[str, Any]:
    if rule not in ("matched", "none"):
        raise SemanticImpasse(
            f"unknown blank correspondence rule {rule!r}",
            locations=[{"rule": rule, "allowed": ["matched", "none"]}])

    rows: list[dict[str, Any]] = []
    n_missing = 0
    n_subtracted = 0
    for shot in integration["injections"]:
        idx = int(shot["injection"])
        row: dict[str, Any] = {
            "injection": idx,
            "schedule_position": shot["schedule_position"],
            "status": shot["status"],
            "discarded": shot["discarded"],
            "first_injection": shot["first_injection"],
            "observed_heat_cal": shot.get("heat_cal"),
            "observed_uncertainty_cal": shot.get("heat_uncertainty_cal"),
            "injected_volume_l": shot.get("injected_volume_l"),
            "window_start_time": shot.get("window_start_time"),
            "window_end_time": shot.get("window_end_time"),
            "endpoint_mode": shot.get("endpoint_mode"),
            "overlaps_next": shot.get("overlaps_next", False),
            "blank_status": "not_applicable",
            "blank_heat_cal": None,
            "corrected_heat_cal": None,
            "diagnostics": list(shot.get("diagnostics", [])),
        }
        usable = shot["status"] in USABLE and not shot["discarded"]
        if shot["status"] not in USABLE:
            row["blank_status"] = "source_unresolved"
        elif shot["discarded"]:
            row["blank_status"] = "discarded"
        elif rule == "none":
            row["blank_status"] = "rule_none"
            row["corrected_heat_cal"] = shot["heat_cal"]
        else:  # matched
            blank = blanks_by_injection.get(idx)
            bheat = None if blank is None else blank.get("heat_cal")
            if blank is None or bheat is None:
                row["blank_status"] = "blank_missing"
                row["diagnostics"].append(diagnostic(
                    "blank_counterpart_missing",
                    "no usable blank curve corresponds to this injection; "
                    "corrected heat withheld (not set to zero)",
                    injection=idx))
                n_missing += 1
            else:
                factor = 1.0
                if scale_by_volume:
                    vv = shot.get("injected_volume_l")
                    bv = blank.get("injected_volume_l")
                    if vv and bv:
                        factor = vv / bv
                row["blank_heat_cal"] = stable_float(bheat * factor)
                row["blank_injection"] = blank.get("injection", idx)
                row["blank_scale_factor"] = stable_float(factor)
                row["corrected_heat_cal"] = stable_float(
                    shot["heat_cal"] - bheat * factor)
                row["blank_status"] = "subtracted"
                n_subtracted += 1
        if usable and row["corrected_heat_cal"] is not None:
            pass
        rows.append(row)

    usable_rows = [r for r in rows
                   if r["status"] in USABLE and not r["discarded"]]
    return {
        "kind": "corrected_heats",
        "rule": rule,
        "scale_by_volume": scale_by_volume,
        "rule_description": ("blank injection matched by injection number, "
                             "scaled by injected-volume ratio"
                             if rule == "matched" else
                             "no blank subtraction applied"),
        "integration_hash": integration["content_hash"]
        if "content_hash" in integration else None,
        "rows": rows,
        "n_usable": len(usable_rows),
        "n_subtracted": n_subtracted,
        "n_blank_missing": n_missing,
        "n_corrected": sum(1 for r in rows
                           if r["corrected_heat_cal"] is not None
                           and not r["discarded"]),
        "basis": dict(integration.get("basis", {})),
    }


def fit_table(corrected: dict[str, Any]) -> tuple[list[float], list[float], list[int]]:
    """Return ``(x_volume_l, q_cal, injection)`` rows usable for model fitting."""
    x: list[float] = []
    y: list[float] = []
    idx: list[int] = []
    for r in corrected["rows"]:
        if (r["status"] in USABLE and not r["discarded"]
                and r["corrected_heat_cal"] is not None
                and r["injected_volume_l"] is not None):
            x.append(float(r["injected_volume_l"]))
            y.append(float(r["corrected_heat_cal"]))
            idx.append(int(r["injection"]))
    return x, y, idx
