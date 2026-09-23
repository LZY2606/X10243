"""Deterministic canonicalisation + content hashing.

"Equivalent inputs get a stable order": JSON emitted here sorts keys, uses
fixed float formatting and never depends on wall-clock or dict insertion order.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _stable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stable(obj[k]) for k in sorted(obj, key=lambda x: str(x))}
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return repr(obj)
        return obj
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(_stable(obj), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode("utf-8")).hexdigest()


def stable_float(x: float, digits: int = 12) -> float:
    """Round-trip a float through a fixed precision for stable fingerprints."""
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")
