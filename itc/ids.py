"""Stable, content-derived identities.

Materials and derived results are *content addressed*: the same bytes and the
same declared provenance always produce the same id, so an export -> delete ->
import round-trip preserves identity and order without any database rowid.

Events are decisions recorded on a branch; they use a chain hash
(previous-event-id + canonical content) which also makes event order tamper
evident.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=_default)


def _default(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, tuple)):
        return sorted(o) if all(isinstance(x, str) for x in o) else list(o)
    return str(o)


def content_id(prefix: str, payload: Any) -> str:
    h = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def event_id(prev_event_id: str | None, branch: str, seq: int,
             payload: Any) -> str:
    material = {"p": prev_event_id, "b": branch, "n": seq,
                "d": payload}
    return "evt_" + hashlib.sha256(
        canonical(material).encode("utf-8")).hexdigest()[:16]


def deterministic_key(obj: Any) -> str:
    """Sort key giving equivalent inputs a stable order."""
    return canonical(obj)
