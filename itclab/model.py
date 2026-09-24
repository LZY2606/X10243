"""Domain records: immutable materials and hashing helpers.

Materials are the immutable layer: power sampling, injection schedules,
concentrations, conditions (temperature / cell volume), baselines and blank
power are each versioned independently. Nothing in a committed material is
ever mutated; a revision commits a new version that explicitly supersedes the
previous one.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Logical roles. Each maps to one independently versioned material stream.
ROLE_POWER = "power"
ROLE_INJECTIONS = "injections"
ROLE_CONCENTRATION = "concentration"
ROLE_CONDITIONS = "conditions"
ROLE_BASELINE = "baseline"

ROLES = {ROLE_POWER, ROLE_INJECTIONS, ROLE_CONCENTRATION, ROLE_CONDITIONS, ROLE_BASELINE}

# Default stable material identifiers per role/channel/target.
DEFAULT_MATERIAL_ID = {
    ("power", "sample"): "mat_power_sample",
    ("power", "blank"): "mat_power_blank",
    ("injections", "sample"): "mat_injections_sample",
    ("injections", "blank"): "mat_injections_blank",
    ("concentration", None): "mat_concentration",
    ("conditions", None): "mat_conditions",
    ("baseline", "sample"): "mat_baseline_sample",
    ("baseline", "blank"): "mat_baseline_blank",
}


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash8(obj: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{digest}" if prefix else digest


def content_hash(role: str, content: dict[str, Any]) -> str:
    return hash8({"role": role, "content": content})


def material_default_id(role: str, scope: str | None) -> str:
    if role in (ROLE_CONCENTRATION, ROLE_CONDITIONS):
        scope = None
    return DEFAULT_MATERIAL_ID[(role, scope)]


def make_material(
    role: str,
    scope: str | None,
    content: dict[str, Any],
    *,
    material_id: str | None,
    origin: dict[str, Any],
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mid = material_id or material_default_id(role, scope)
    return {
        "material_id": mid,
        "role": role,
        "scope": scope,
        "content": content,
        "origin": origin,
        "diagnostics": diagnostics or [],
        "content_hash": content_hash(role, content),
    }
