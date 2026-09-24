"""Stable content identities and deterministic ordering."""

from __future__ import annotations

import hashlib
import json


def canonical(obj) -> str:
    """JSON canonicalization with sorted keys and stable float text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_default)


def _default(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not canonicalizable: {type(o)!r}")


def digest(prefix, obj) -> str:
    """Content-addressed identity ``<prefix>-<sha256[:16]>``.

    Equivalent inputs always produce the same identity, in every process and
    after an export/import round trip.
    """
    h = hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def stable_key(d):
    """Deterministic ordering key for equivalent candidate results."""
    return canonical(d)
