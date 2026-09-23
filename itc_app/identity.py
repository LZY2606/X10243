"""Stable, deterministic identities and canonical serialisation.

Equivalent inputs always produce the same identifier and the same ordering.
Identifiers are content-addressed (sha256 over canonical JSON with sorted
keys) and are preserved across export/import round trips.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def content_hash(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def stable_sequence_key(item: dict[str, Any]) -> str:
    """Deterministic ordering key for equivalent unordered items."""
    # Prefer explicit natural coordinates, then identity, then full content.
    for key in ("index", "inject_no", "sequence", "time_s", "at"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return f"a:{value:020.6f}"
    for key in ("id", "version_id", "event_id"):
        value = item.get(key)
        if isinstance(value, str):
            return f"b:{value}"
    return f"c:{canonical_json(item)}"


def stable_dedupe_key(item: dict[str, Any]) -> str:
    for key in ("id", "version_id", "event_id", "code"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return canonical_json(item)
