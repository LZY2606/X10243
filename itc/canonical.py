"""Deterministic canonical encoding and stable identity helpers.

Equivalent inputs must always serialise to the same bytes and therefore the
same identity, on any machine and across export/import round trips.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": obj.hex()}
    raise TypeError(f"cannot canonicalise object of type {type(obj).__name__}")


def encode(value: Any) -> bytes:
    """Return canonical JSON bytes (sorted keys, no insignificant spaces)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """Return a short, stable content digest for *value*."""
    return hashlib.sha256(encode(value)).hexdigest()[:16]


def identity(prefix: str, value: Any) -> str:
    """Stable identifier of the form ``prefix-digest`` for *value*."""
    return f"{prefix}-{digest(value)}"


def keyed(prefix: str, **parts: Any) -> str:
    """Stable identifier from an ordered (post-sort) mapping of parts."""
    return identity(prefix, parts)
