"""Deterministic, content-derived stable identities.

* material ids are a content hash of the verbatim source bytes.
* result ids hash rule version + parent material versions + inputs.
* event ids form a tamper-evident hash chain per branch.

Everything is independent of wall-clock time and insertion order, so
"equivalent input -> stable ordering/identity" and export/import round trips
preserve ids exactly.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def material_id(kind: str, file_name: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(b"material\n")
    h.update(kind.encode())
    h.update(b"\n")
    h.update(file_name.encode("utf-8", "surrogateescape"))
    h.update(b"\n")
    h.update(content.encode("utf-8", "surrogateescape"))
    return "mat_" + h.hexdigest()[:16]


def result_id(kind: str, parents: dict[str, Any], inputs: Any,
              rule_version: str, run_id: str = "") -> str:
    payload = {"kind": kind, "parents": parents, "inputs": inputs,
               "rule_version": rule_version, "run_id": run_id}
    return "res_" + hashlib.sha256(
        _canonical(payload).encode()).hexdigest()[:16]


def event_id(seq: int, branch: str, event_type: str, payload: Any,
             parent_event_id: str | None) -> str:
    payload_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    h = hashlib.sha256()
    h.update(b"event\n")
    h.update(str(seq).encode())
    h.update(branch.encode())
    h.update(event_type.encode())
    h.update(payload_hash.encode())
    h.update((parent_event_id or "ROOT").encode())
    return "evt_" + h.hexdigest()[:16]


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
