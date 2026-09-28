"""Project export/import bundles.

The bundle is one self-describing JSON document carrying immutable
materials, results, events and branches. Identities are content-derived, so
re-import reconstructs the exact same ids, order and provenance relationships
even after the local database was deleted.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import FormatCorrupt, SemanticUnresolved

BUNDLE_FORMAT = "itc-bundle/v1"


def export_bundle(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": BUNDLE_FORMAT,
        "project_id": project["id"],
        "exported_schema": project.get("schema"),
        "seq": project["seq"],
        "materials": project["materials"],
        "results": project["results"],
        "events": project["events"],
        "branches": project["branches"],
    }


def encode_bundle(project: dict[str, Any]) -> bytes:
    return json.dumps(export_bundle(project), ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def decode_bundle(raw: bytes) -> dict[str, Any]:
    try:
        bundle = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise FormatCorrupt("bundle is not UTF-8", diagnostics=[{"reason": "ENCODING"}]) from exc
    except json.JSONDecodeError as exc:
        raise FormatCorrupt(
            "bundle JSON is structurally damaged",
            diagnostics=[{"reason": "JSON_PARSE", "detail": str(exc)}],
        ) from exc
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    problems: list[dict[str, Any]] = []
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise FormatCorrupt("not an itc-bundle/v1 document", diagnostics=[{"reason": "WRONG_FORMAT"}])
    for key in ("project_id", "materials", "results", "events", "branches"):
        if key not in bundle:
            problems.append({"reason": "MISSING_SECTION", "key": key})
    if problems:
        raise FormatCorrupt("bundle is missing required sections", diagnostics=problems)

    materials = bundle["materials"]
    events = bundle["events"]
    for event in events:
        payload = event.get("payload", {})
        if event.get("type") in ("IMPORT_MATERIAL", "SUPERSEDE_HEAD"):
            mid = payload.get("material", {}).get("id") or payload.get("material_id")
            if mid and mid not in materials:
                problems.append({"reason": "DANGLING_MATERIAL", "event": event.get("id"), "material_id": mid})
        if "id" not in event or "branch" not in event or "type" not in event:
            problems.append({"reason": "MALFORMED_EVENT", "event": event.get("id")})

    for branch, record in bundle.get("branches", {}).items():
        state = record.get("state", {})
        int_id = state.get("current_integration_id")
        if int_id and int_id not in bundle["results"]:
            problems.append({"reason": "DANGLING_INTEGRATION", "branch": branch, "integration_id": int_id})
        fit_id = state.get("current_fit_id")
        if fit_id and fit_id not in bundle["results"]:
            problems.append({"reason": "DANGLING_FIT", "branch": branch, "fit_id": fit_id})
    if problems:
        raise SemanticUnresolved("bundle references missing objects", diagnostics=problems)


def import_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the in-memory project exactly as exported (ids preserved)."""
    validate_bundle(bundle)
    project = {
        "id": bundle["project_id"],
        "schema": bundle.get("exported_schema", "itc-project/v1"),
        "seq": bundle.get("seq", len(bundle["events"])),
        "materials": dict(bundle["materials"]),
        "results": dict(bundle["results"]),
        "events": list(bundle["events"]),
        "branches": dict(bundle["branches"]),
    }
    _verify_lineage(project)
    return project


def _verify_lineage(project: dict[str, Any]) -> None:
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in sorted(project["events"], key=lambda e: e.get("seq", 0)):
        parent = event.get("parent_event_id")
        if parent and parent not in seen:
            problems.append({"reason": "BROKEN_PARENT", "event": event["id"]})
        seen.add(event["id"])
    if problems:
        raise SemanticUnresolved("event lineage is broken", diagnostics=problems)
