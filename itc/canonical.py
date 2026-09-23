"""Canonical JSON serialisation and stable identities.

Equivalent inputs must produce equivalent bytes, so:

* dict keys are emitted in sorted order;
* floats use ``repr`` (round-trippable); ints are ints;
* None / bool / str round-trip naturally.

Stable ids are content hashes over canonical bytes.  Order-sensitive sequences
(lists) keep their order; order-insensitive collections (e.g. candidate model
results) are sorted via :func:`stable_order_key` before hashing/display.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, tuple)):
        return sorted(obj, key=lambda x: json.dumps(x, sort_keys=True,
                                                    default=_default,
                                                    ensure_ascii=False))
    raise TypeError(f"object of type {type(obj).__name__} is not canonicalisable")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=_default, allow_nan=False).encode("utf-8")


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def short_hash(obj: Any, length: int = 12) -> str:
    return content_hash(obj)[:length]


def stable_order_key(obj: Any) -> str:
    """Total, deterministic ordering key for domain payloads."""
    if isinstance(obj, dict):
        parts = []
        for key in sorted(obj):
            parts.append(f"{key}={stable_order_key(obj[key])}")
        return "{" + ";".join(parts) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(stable_order_key(v) for v in obj) + "]"
    return json.dumps(obj, ensure_ascii=False, default=_default, allow_nan=False)


def stable_sort(items: list[Any], key=None) -> list[Any]:
    if key is None:
        return sorted(items, key=stable_order_key)
    return sorted(items, key=lambda item: stable_order_key(key(item)))
