"""Stable, content-derived identities and deterministic ordering.

Equivalent inputs always produce the same id and the same sort order, in a
fresh database or after an export/import round trip.  Floats are rounded to
12 significant decimal places before hashing so hashes do not depend on
hardware float artifacts; full-precision values are still stored separately.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if obj == 0:
            return 0.0
        return float(f"{obj:.12g}")
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_canonicalize(obj), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = canonical_json(list(parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def stable_order_key(obj: dict[str, Any], *fields: str) -> tuple:
    key = []
    for f in fields:
        v = obj
        for piece in f.split("."):
            v = v.get(piece) if isinstance(v, dict) else None
        if isinstance(v, float):
            v = round(v, 12)
        key.append(("~" if v is None else v))
    return tuple(key)
