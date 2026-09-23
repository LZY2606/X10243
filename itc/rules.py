"""Versioned analysis rule sets.

Rules are data, versioned and referenced by derived results. Changing a rule
creates a new version; it never edits history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RULE_KIND = "rule_set"

# Rule version constants referenced across the code base.
V1 = "rules/itc-review/v1"
CURRENT_RULE_VERSION = V1


@dataclass(frozen=True)
class RuleSet:
    version: str
    # gap detection: consecutive samples separated by more than
    # gap_factor * median dt count as a data gap
    gap_factor: float = 4.0
    # a peak returns to baseline within return_tol * sigma
    return_tol: float = 2.5
    # peak detection threshold above baseline (sigma units)
    peak_threshold: float = 3.0
    # minimum samples inside an integration window
    min_window_samples: int = 3
    # BIC difference below which models are reported as co-optimal (tied)
    tie_bic_delta: float = 2.0
    # numeric floor on SSE to make log-likelihood finite (cal^2)
    sse_floor: float = 1e-20
    # bounded fitter budgets (these make "budget exhausted" real and testable)
    one_site_budget: int = 600
    two_site_budget: int = 900
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": RULE_KIND,
            "version": self.version,
            "gap_factor": self.gap_factor,
            "return_tol": self.return_tol,
            "peak_threshold": self.peak_threshold,
            "min_window_samples": self.min_window_samples,
            "tie_bic_delta": self.tie_bic_delta,
            "sse_floor": self.sse_floor,
            "one_site_budget": self.one_site_budget,
            "two_site_budget": self.two_site_budget,
            "description": self.description or "initial ITC review rule set",
        }


SETS: dict[str, RuleSet] = {
    V1: RuleSet(version=V1),
}


def get(version: str) -> RuleSet:
    try:
        return SETS[version]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown rule version {version!r}") from exc


def all_versions() -> list[str]:
    return sorted(SETS)
