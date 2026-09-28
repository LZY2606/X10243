"""Stable, content-derived identifiers.

Equivalent inputs must always produce the same identity and the same ordering,
so ids are hashes of canonical JSON (dictionaries sorted by key).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_id(prefix: str, obj: Any) -> str:
    digest = hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"
