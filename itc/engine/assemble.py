"""Ingest parsed materials into an experiment bundle.

Material kinds are versioned separately: power trace, injection schedule,
concentration, temperature, cell volume, baseline trace and blank trace each
get their own immutable revision.  Incomplete bundles fail with located
diagnostics instead of silently accepting nulls/defaults.
"""
from __future__ import annotations

from typing import Any

from ..errors import Diagnostic, SemanticUnresolvable
from .hashing import sha12

POINT_KINDS = ("power", "baseline", "blank")


def material_id(kind: str, role: str, revision: int) -> str:
    return f"mat_{kind}_{role}_v{revision:02d}"


def build_material(parsed: dict[str, Any], revision: int) -> dict[str, Any]:
    kind = parsed["kind"]
    role = parsed.get("role", "main") if kind in POINT_KINDS else "main"
    mid = material_id(kind, role, revision)
    return {
        "id": mid,
        "kind": kind,
        "role": role,
        "revision": revision,
        "name": parsed.get("name", mid),
        "coordinate_basis": parsed.get("coordinate_basis"),
        "units": parsed.get("units", {}),
        "payload": parsed,
        "diagnostics": list(parsed.get("diagnostics", [])),
        "hash": "sha256:" + sha12(parsed),
    }


def assemble(materials: list[dict[str, Any]]) -> dict[str, Any]:
    """materials: list of build_material() outputs. Returns bundle descriptor."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for mat in materials:
        by_kind.setdefault(mat["kind"], []).append(mat)
        for d in mat.get("diagnostics", []):
            diagnostics.append(d)

    required = ["power", "injections", "concentrations", "conditions", "baseline"]
    for kind in required:
        if not by_kind.get(kind):
            raise SemanticUnresolvable(
                f"bundle is missing required material kind {kind!r}",
                [Diagnostic("missing_material",
                            f"{kind} trace/table is required", target=kind)])

    latest = {kind: sorted(rows, key=lambda m: m["revision"])[-1]
              for kind, rows in by_kind.items()}

    power = latest["power"]
    base = latest["baseline"]
    _check_trace(power, diagnostics)
    _check_trace(base, diagnostics, "baseline")
    for blank in by_kind.get("blank", []):
        _check_trace(blank, diagnostics, "blank")

    ref = power["payload"].get("baseline_ref")
    if ref and base["id"] != ref and base["name"] != ref:
        diagnostics.append(Diagnostic(
            "baseline_ref_mismatch",
            f"power references baseline {ref!r} but assembled baseline is {base['id']!r}",
            target=power["id"]).to_dict())

    inj = latest["injections"]["payload"]["injections"]
    power_tmax = _tmax(power)
    for row in inj:
        tv = row["time"]["canonical"]
        if tv is None or power_tmax is None:
            continue
        if tv["value"] < 0 or tv["value"] > power_tmax:
            diagnostics.append(Diagnostic(
                "injection_outside_trace",
                f"injection {row['ordinal']} at {tv['value']:g}s is outside power trace",
                target=f"injection:{row['ordinal']}").to_dict())

    bundle = {
        "materials": {m["id"]: _stripped(m) for m in materials},
        "latest": {kind: mat["id"] for kind, mat in latest.items()},
        "revisions": {kind: [m["id"] for m in sorted(rows, key=lambda x: x["revision"])]
                      for kind, rows in by_kind.items()},
        "diagnostics": diagnostics,
    }
    return bundle


def _stripped(mat: dict[str, Any]) -> dict[str, Any]:
    return {"id": mat["id"], "kind": mat["kind"], "role": mat["role"],
            "revision": mat["revision"], "name": mat["name"],
            "coordinate_basis": mat["coordinate_basis"], "units": mat["units"],
            "hash": mat["hash"], "diagnostics": mat["diagnostics"]}


def _tmax(trace: dict[str, Any]) -> float | None:
    pts = trace["payload"].get("points", [])
    vals = [p["t"]["canonical"]["value"] for p in pts
            if p["t"]["canonical"] is not None]
    return max(vals) if vals else None


def _check_trace(trace: dict[str, Any], diagnostics: list[dict], label: str | None = None):
    pts = trace["payload"].get("points", [])
    converted = [p for p in pts if p["t"]["canonical"] is not None
                 and p["p"]["canonical"] is not None]
    if len(converted) < 2:
        diagnostics.append(Diagnostic(
            "trace_unusable",
            f"{label or trace['id']} has fewer than 2 convertible samples",
            target=trace["id"]).to_dict())
