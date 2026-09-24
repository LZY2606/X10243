"""Deterministic identifiers and canonical serialization.

Equivalent inputs must produce equivalent identifiers and a stable ordering.
All hashing goes through :func:`canonical_json` so float handling and key
ordering are identical in every layer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not canonically serializable")


def canonical_json(payload: Any) -> str:
    """Serialize with sorted keys and a fixed float/NaN policy."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    )


def content_hash(payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


def stable_id(prefix: str, payload: Any) -> str:
    """Stable, content-derived identifier such as ``intg_8f31a2...``."""

    return f"{prefix}_{content_hash(payload)[:16]}"


def stable_order_key(payload: Any) -> str:
    """Deterministic ordering key for equivalent candidate results."""

    return canonical_json(payload)
