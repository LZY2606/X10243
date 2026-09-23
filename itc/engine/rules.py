"""Domain rule set.  Rules are data, versioned by content hash: changing a
rule (e.g. blank correspondence) creates a new rule version; decisions record
which version they used.  Rules never live only in the browser."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hashing import sha12

DEFAULT_RULES: dict[str, Any] = {
    "integration": {
        "end_of_window": "next_injection",   # integrate [t_i, t_{i+1})
        "baseline_interpolation": "linear",
        "gap_policy": "clip_before_gap",     # windows never cross gaps
        "return_to_baseline_sigma": 3.0,
        "return_to_baseline_dwell_s": 2.0,
    },
    "first_injection": {"discard": True, "reason": "first injection mixing artifact"},
    "blank_correspondence": {
        "mode": "ordinal",                   # correspondence MUST be declared
        "offset": 0,                         # blank ordinal = sample ordinal
        "required": False,
    },
    "fit": {
        "tie_aic_delta": 1.0,                # |delta AIC| <= this => tied
        "max_iterations": 4000,
        "x_tolerance": 1e-10,
    },
}


def rules_hash(rules: dict[str, Any]) -> str:
    return "rules_" + sha12(rules)


def with_overrides(base: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    out = deepcopy(base)
    node = out
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return out
