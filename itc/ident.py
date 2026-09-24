"""Stable, content-based identities and deterministic ordering helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
NAMESPACE = "itc"


def canonical(obj: Any) -> str:
    """Deterministic JSON encoding (sorted keys, no whitespace surprises)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    raise TypeError(f"not canonically serializable: {type(o)!r}")


def digest(data: Any, *, length: int = 16) -> str:
    if not isinstance(data, (str, bytes)):
        data = canonical(data)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def make_id(kind: str, *parts: Any, length: int = 16) -> str:
    """``kind_<hash(canonical(parts))>`` — same inputs always give same id."""
    body = digest([kind, *parts], length=max(length, 12))
    return f"{kind}_{body}"[:length + len(kind) + 1]


def event_id(case_id: str, seq: int, parent_id: str, payload: dict) -> str:
    return make_id("ev", case_id, seq, parent_id, payload, length=20)


def stable_key(value: Any) -> Any:
    """Sort key that tolerates mixed types and nested lists/dicts."""
    if isinstance(value, dict):
        return (1, tuple(sorted((str(k), stable_key(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (2, tuple(stable_key(v) for v in value))
    return (0, _scalar(value))


def _scalar(v: Any) -> Any:
    if v is None:
        return (0, "")
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, (int, float)):
        return (2, float(v))
    return (3, str(v))
