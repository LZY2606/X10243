"""Canonical JSON encoding and content-addressed stable identities."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, compact separators, no whitespace."""
    return json.dumps(
        _sanitize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False if False else True,  # NaN already sanitised above
    ).encode("utf-8")


def content_id(prefix: str, payload: Any) -> str:
    """Stable content-addressed identity, e.g. ``v_pwr_ab12cd34``."""
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return f"{prefix}_{digest[:12]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
