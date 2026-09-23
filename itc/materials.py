"""Build immutable, versioned *material* artifacts from parsed imports.

Materials are the immutable layer: power sampling, injection schedule
(volume + time), concentrations, conditions (temperature + cell volume),
baseline curve and blank curves. Each kind is versioned independently and
addressed by content hash. A revision adds a new version; nothing is mutated.
"""

from __future__ import annotations

from typing import Any

from .canonical import content_hash
from .errors import SemanticImpasse, diagnostic
from .units import CANONICAL_UNIT


def _series_doc(kind: str, parsed: Any, spec: dict[str, Any], *,
                is_blank: bool = False) -> dict[str, Any]:
    times: list[float] = []
    powers: list[float] = []
    kept_raw: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for i, (t, p) in enumerate(zip(parsed.times, parsed.powers)):
        if t is None or p is None:
            invalid.append({"sample": i, "time": t, "power": p,
                            "raw": parsed.raw_rows[i]})
            continue
        times.append(t)
        powers.append(p)
        kept_raw.append(parsed.raw_rows[i])
    if not times:
        raise SemanticImpasse(
            f"{kind}: no numeric samples remain after import",
            locations=[{"id": spec.get("id")}])
    unit_t = spec.get("time_unit", "s")
    unit_p = spec.get("power_unit", "µcal/s")
    doc: dict[str, Any] = {
        "kind": kind,
        "id_hint": spec.get("id"),
        "role": spec.get("role", "sample" if not is_blank else "blank"),
        "unit_time": unit_t,
        "unit_power": unit_p,
        "canonical_unit_time": CANONICAL_UNIT["time"],
        "canonical_unit_power": CANONICAL_UNIT["power"],
        "time_basis": spec.get("time_basis", "series_first_sample"),
        "times": times,
        "powers": powers,
        "n_samples": len(times),
        "raw_representation": {
            "headers": parsed.headers,
            "rows": kept_raw[:2000],
            "source_path": spec.get("path"),
        },
        "import_diagnostics": list(parsed.diagnostics),
        "invalid_samples": invalid,
    }
    if is_blank:
        doc["blank_id"] = spec.get("id")
        doc["corresponding_injection"] = spec.get("corresponding_injection")
        doc["blank_label"] = spec.get("label")
    return doc


def build_materials(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{material_id: artifact}`` for one initial case-pack load."""
    out: dict[str, dict[str, Any]] = {}
    for mid, item in pack["materials"].items():
        spec = item["spec"]
        kind = spec["kind"]
        if kind in ("power", "baseline", "blank"):
            doc = _series_doc(kind, item["series"], spec, is_blank=kind == "blank")
        elif kind in ("schedule", "concentration", "conditions"):
            doc = dict(item["doc"])
            doc["id_hint"] = spec.get("id")
        else:  # pragma: no cover - ingestion rejects earlier
            continue
        doc["material_id"] = f"{kind}:{mid}"
        doc["version"] = 1
        doc["content_hash"] = content_hash(
            {k: v for k, v in doc.items() if k != "content_hash"})
        out[mid] = doc
    return out


def revise_material(previous: dict[str, Any] | None, new_doc: dict[str, Any],
                    *, kind_hint: str) -> dict[str, Any]:
    """Create a new immutable version of one material kind.

    Only the revised material changes; other material identities are untouched.
    """
    doc = dict(new_doc)
    if previous is not None:
        doc["material_id"] = previous["material_id"]
        doc["version"] = int(previous["version"]) + 1
        doc["replaces"] = previous["content_hash"]
    else:
        doc.setdefault("material_id", f"{kind_hint}:{doc.get('id_hint', 'x')}")
        doc["version"] = 1
    doc["content_hash"] = content_hash(
        {k: v for k, v in doc.items() if k != "content_hash"})
    return doc


def revise_series(previous: dict[str, Any], *,
                  times: list[float] | None = None,
                  powers: list[float] | None = None,
                  note: str = "") -> dict[str, Any]:
    """Convenience: revise a series (used by local source-change scenarios)."""
    doc = dict(previous)
    if times is not None:
        doc["times"] = list(times)
    if powers is not None:
        doc["powers"] = list(powers)
    doc["n_samples"] = len(doc["times"])
    doc["revision_note"] = note
    return revise_material(previous, doc, kind_hint=previous["kind"])


def revise_concentration(previous: dict[str, Any], *, role: str,
                         new_value_canonical: float, new_unit: str,
                         raw_repr: str, reason: str) -> dict[str, Any]:
    """Revise one concentration entry (the 'concentration revision' scenario).

    The original integration is never modified by this: only model fits depend
    on concentrations, so they become stale/superseded.
    """
    doc = {
        "kind": "concentration",
        "id_hint": previous.get("id_hint"),
        "path": previous.get("path"),
        "headers": previous.get("headers"),
        "entries": [],
    }
    changed = False
    for entry in previous["entries"]:
        e = {
            "role": entry["role"],
            "raw": dict(entry.get("raw", {})),
            "concentration": dict(entry["concentration"]),
        }
        if entry["role"] == role and not changed:
            q = dict(e["concentration"])
            q["raw"] = raw_repr
            q["value"] = float(new_value_canonical)
            q["unit"] = new_unit
            q["original_unit"] = q.get("original_unit", new_unit)
            q["revision"] = {
                "previous_value": entry["concentration"].get("value"),
                "previous_unit": entry["concentration"].get("unit"),
                "reason": reason,
            }
            q["diagnostics"] = list(q.get("diagnostics", []))
            e["concentration"] = q
            changed = True
        doc["entries"].append(e)
    if not changed:
        raise SemanticImpasse(
            f"cannot revise concentration: role {role!r} not present",
            locations=[{"role": role}])
    return revise_material(previous, doc, kind_hint="concentration")


def concentration_of(material: dict[str, Any], role: str) -> float | None:
    """Canonical concentration (M) for a role, or None with diagnostics kept."""
    for entry in material.get("entries", []):
        if entry["role"] == role:
            return entry["concentration"].get("value")
    return None


def cell_volume(conditions: dict[str, Any]) -> float:
    v = conditions["cell_volume"]["value"]
    if v is None:
        raise SemanticImpasse(
            "cell volume did not convert; no default substituted",
            locations=[{"conditions": conditions.get("path"),
                        "diagnostics": conditions["cell_volume"].get("diagnostics")}])
    return v


def temperature(conditions: dict[str, Any]) -> float:
    v = conditions["temperature"]["value"]
    if v is None:
        raise SemanticImpasse(
            "temperature did not convert; no default substituted",
            locations=[{"conditions": conditions.get("path"),
                        "diagnostics": conditions["temperature"].get("diagnostics")}])
    return v
