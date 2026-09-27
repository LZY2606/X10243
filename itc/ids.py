"""Stable identity helpers.

Content identity (materials) is a pure content hash: importing the same bytes
twice yields the same id, so export/delete/import preserves identity and
"equivalent inputs get a stable order".

Result identity is deterministic over (rule-set, kind, input summaries,
parent dependency ids) - replay after export/import reproduces every id.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

RULESET_VERSION = "rules-v1"


def content_hash(blob: bytes | str) -> str:
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def digest(prefix: str, parts: Sequence[Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
