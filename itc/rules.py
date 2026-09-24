"""Versioned computation rules.

Rules are data, never browser-only: each rule set has a content-addressed
``rule_id`` that every derived result records as a parent dependency.
"""

from __future__ import annotations

from .hashing import digest

RULE_KIND_INTEGRATION = "integration_rules"
RULE_KIND_BLANK = "blank_rules"
RULE_KIND_MODEL = "model_rules"

DEFAULT_INTEGRATION_RULES = {
    # Window for injection i: [t_i + start_offset, t_{i+1}+start_offset)
    # clipped to [start_offset, end_offset] relative to each injection time.
    "start_offset_s": 4.0,
    "end_offset_s": 150.0,
    # A new peak whose window starts before the previous integrated peak
    # returns to baseline flags OVERLAP_PEAKS (peaks may overlap; the
    # integration window itself never crosses a data gap).
    "settle_s": 60.0,
    "baseline_mode": "curve_anchors",  # piecewise-linear anchors
    "discard_first_injection": True,
}

DEFAULT_BLANK_RULES = {
    # correspondence rule must be explicit; missing correspondence is a
    # named diagnostic, never silently treated as zero.
    "correspondence": "by_index",  # or "by_volume"
    "required": True,
    "apply": True,
}

DEFAULT_MODEL_RULES = {
    # BIC/SSE tie tolerance; candidates within it are all kept.
    "tie_relative_tolerance": 5e-4,
    "evaluation_budget": 4000,
    "discard_first_injection": True,
}


def rule_id(kind, body):
    return digest(f"rule_{kind}", body)


def default_ruleset():
    return {
        RULE_KIND_INTEGRATION: dict(DEFAULT_INTEGRATION_RULES),
        RULE_KIND_BLANK: dict(DEFAULT_BLANK_RULES),
        RULE_KIND_MODEL: dict(DEFAULT_MODEL_RULES),
    }
