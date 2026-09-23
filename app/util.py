"""Deterministic hashing / canonical-encoding helpers.

Derived identities are content-addressed so that replaying equivalent inputs
always yields the same ids and the same ordering.  Wall-clock time never
participates in an id.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default
    ).encode("utf-8")


def _default(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, tuple)):
        return sorted(o, key=lambda x: canonical(x))
    raise TypeError(f"not canonically encodable: {type(o)!r}")


def digest(obj: Any, *, length: int = 16, prefix: str = "") -> str:
    raw = hashlib.sha256(canonical(obj)).hexdigest()
    return f"{prefix}{raw[:length]}" if prefix else raw[:length]
