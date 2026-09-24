"""Stable identity helpers. Ids are deterministic content hashes so that
export -> delete -> import round-trips preserve identity."""
from __future__ import annotations

import hashlib
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj) -> str:
    return hashlib.sha1(canonical_json(obj).encode("utf-8")).hexdigest()


def material_version_id(project_id: str, slot: str, content: dict) -> str:
    return "mv_" + content_hash({"project": project_id, "slot": slot, "content": content})[:16]


def derived_id(kind: str, inputs: dict) -> str:
    return "dr_" + content_hash({"kind": kind, "inputs": inputs})[:16]


def named_id(prefix: str, *parts) -> str:
    return prefix + "_" + content_hash("|".join(str(p) for p in parts))[:16]
