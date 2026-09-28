"""Deterministic canonical serialization and stable identity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> bytes:
    """Stable JSON bytes: sorted keys, no whitespace, allow_nan disabled."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(obj: Any, prefix: str = "") -> str:
    raw = hashlib.sha256(canonical(obj)).hexdigest()
    return f"{prefix}{raw[:16]}" if prefix else raw[:16]


def digest_full(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()
