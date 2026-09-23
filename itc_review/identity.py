"""Stable identity and deterministic serialization.

Material ids are derived from their immutable content; computed-result ids are
derived from their input summary, parameter version and parent dependencies.
Equivalent inputs therefore hash to the same id and sort in a stable order.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical(obj: Any) -> str:
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_normalize(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return _float_key(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return _float_key(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _float_key(x: float) -> str:
    if x != x:  # NaN must not poison the hash
        return "NaN"
    if x == 0.0:
        return "0"
    return repr(float(np.round(x, 12)))


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def material_id(kind: str, payload: dict[str, Any]) -> str:
    return f"mat_{kind}_{digest(payload)[:16]}"


def result_id(kind: str, inputs: dict[str, Any], params: dict[str, Any],
              parents: list[str]) -> str:
    content = {
        "kind": kind,
        "inputs": inputs,
        "params": params,
        "parents": sorted(parents),
    }
    return f"res_{kind}_{digest(content)[:16]}"


def event_id(seq: int, branch: str, action: str, payload: dict[str, Any],
             parent: str | None) -> str:
    # Event identity is defined by content + position in its branch.
    content = {"seq": seq, "branch": branch, "action": action,
               "payload": payload, "parent": parent}
    return f"evt_{digest(content)[:16]}"


def stable_key(value_id: str, label: str) -> str:
    return f"{value_id}:{label}"
