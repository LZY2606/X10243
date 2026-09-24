"""Stable identity helpers.

IDs are content-derived so that equivalent inputs yield a stable ordering and
an export/import round-trip keeps the same identity. Wall-clock time is never
part of an ID.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def stable_id(prefix: str, obj: Any) -> str:
    return f"{prefix}_{content_hash(obj)[:16]}"


def child_id(prefix: str, parent_id: str, payload: Any) -> str:
    return stable_id(prefix, {"parent": parent_id, "payload": payload})
