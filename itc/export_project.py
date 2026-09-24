"""Project export / import - identity, branch and decision history preserved.

An export is a deterministic ZIP bundle:

``manifest.json``  - schema, created time, content checksums, stable ordering
``project.json``   - full serialised event-sourced state

Import verifies every checksum before touching the database: a bad/partial
bundle is :class:`FormatCorrupt`, a structurally fine bundle that references
unknown parents is :class:`SemanticInfeasible`, and they stay distinct.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile

from .errors import Diagnostic, FormatCorrupt, SemanticInfeasible
from .storage import deserialize_state, serialize_state
from .engine import ProjectState

EXPORT_SCHEMA = "itc-export/1.0"


def export_zip(state: ProjectState, *, created_at: str = "") -> bytes:
    project_text = serialize_state(state)
    checksum = hashlib.sha256(project_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema": EXPORT_SCHEMA,
        "project_id": state.project_id,
        "title": state.title,
        "created_at": created_at,
        "entries": [
            {"name": "project.json", "sha256": checksum,
             "bytes": len(project_text.encode("utf-8"))},
        ],
        "stable_event_order": [e.event_id for e in state.events],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json",
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                               indent=2))
        zf.writestr("project.json", project_text)
    return buf.getvalue()


def read_export_bundle(data: bytes) -> tuple[dict, ProjectState]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise FormatCorrupt(
            "export bundle is not a valid zip",
            [Diagnostic("export:bad_zip", str(exc), "$", None)])
    names = set(zf.namelist())
    if not {"manifest.json", "project.json"} <= names:
        raise FormatCorrupt(
            "export bundle missing manifest.json or project.json",
            [Diagnostic("export:missing_entries",
                        f"present={sorted(names)}", "$", sorted(names))])
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except json.JSONDecodeError as exc:
        raise FormatCorrupt(
            "manifest.json is not valid JSON",
            [Diagnostic("export:bad_manifest", str(exc), "manifest.json", None)])
    raw_project = zf.read("project.json")
    expected = {e["name"]: e for e in manifest.get("entries", [])}
    if "project.json" in expected:
        actual = hashlib.sha256(raw_project).hexdigest()
        if actual != expected["project.json"]["sha256"]:
            raise FormatCorrupt(
                "project.json checksum mismatch",
                [Diagnostic("export:checksum",
                            f"expected {expected['project.json']['sha256'][:12]} "
                            f"got {actual[:12]}", "project.json", actual)])
    try:
        state = deserialize_state(raw_project.decode("utf-8"))
    except FormatCorrupt:
        raise
    # semantic validation: branch parent chains must resolve and match order
    ids = {e.event_id for e in state.events}
    for event in state.events:
        for parent in event.parent_ids:
            if parent not in ids:
                raise SemanticInfeasible(
                    f"event {event.event_id} references unknown parent {parent}",
                    [Diagnostic("export:dangling_parent",
                                "unknown parent event",
                                f"events/{event.event_id}/parents", parent)])
    order = manifest.get("stable_event_order")
    if order is not None and order != [e.event_id for e in state.events]:
        raise SemanticInfeasible(
            "event order does not match manifest stable order",
            [Diagnostic("export:event_order",
                        "order drift on import", "manifest.stable_event_order",
                        order)])
    return manifest, state


def import_zip(data: bytes) -> tuple[dict, ProjectState]:
    return read_export_bundle(data)

