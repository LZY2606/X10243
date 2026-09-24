"""Deterministic serialisation + stable identity hashing.

Equivalent inputs must hash identically and sort identically, so everything
that participates in identity goes through :func:`canonical_json` first.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Stable JSON text: sorted keys, compact separators, no ASCII escaping."""

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any, *, prefix: str = "", length: int = 16) -> str:
    """Short stable id derived from the canonical representation of ``value``."""

    text = canonical_json(value)
    raw = hashlib.blake2b(text.encode("utf-8"), digest_size=length).hexdigest()
    return f"{prefix}{raw}" if prefix else raw


def digest_text(text: str, *, prefix: str = "", length: int = 16) -> str:
    raw = hashlib.blake2b(text.encode("utf-8"), digest_size=length).hexdigest()
    return f"{prefix}{raw}" if prefix else raw


def stable_key(value: Any) -> tuple:
    """Turn JSON-ish values into something orderable for deterministic sorts.

    Dict/list ordering is flattened into tagged tuples so that None, bool,
    numbers and strings never raise ``TypeError`` when mixed in a sort.
    """

    if value is None:
        return (0,)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (2, float(value))
    if isinstance(value, str):
        return (3, value)
    if isinstance(value, (list, tuple)):
        return (4, tuple(stable_key(v) for v in value))
    if isinstance(value, dict):
        return (5, tuple(sorted((k, stable_key(v)) for k, v in value.items())))
    return (6, str(value))


def stable_sorted(items: list[Any], *, key=None) -> list[Any]:
    if key is None:
        return sorted(items, key=stable_key)
    return sorted(items, key=lambda item: stable_key(key(item)))
