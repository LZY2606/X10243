"""Versioned rule-set parsing (integration / blank correction / models)."""
from __future__ import annotations

from typing import Any

_BOOL = {"true": True, "false": False}


class RuleError(ValueError):
    pass


def _coerce(value: str) -> Any:
    v = value.strip()
    low = v.lower()
    if low in _BOOL:
        return _BOOL[low]
    try:
        if any(c in v for c in ".eE"):
            return float(v)
        return int(v)
    except ValueError:
        if "," in v:
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


def parse_rule_text(text: str, *, source_name: str = "<rule>") -> dict[str, Any]:
    params: dict[str, Any] = {}
    notes: list[str] = []
    rule_set = None
    version = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if body:
                notes.append(body)
            continue
        if ":" not in s:
            raise RuleError(f"{source_name}:{lineno}: expected key: value, "
                            f"got {s!r}")
        key, _, value = s.partition(":")
        key = key.strip()
        if key == "rule_set":
            rule_set = value.strip()
        elif key == "rule_set_version":
            version = value.strip()
        else:
            params[key] = _coerce(value)
    if rule_set is None:
        raise RuleError(f"{source_name}: missing 'rule_set'")
    if version is None:
        raise RuleError(f"{source_name}: missing 'rule_set_version'")
    return {
        "rule_set": rule_set,
        "rule_version": version,
        "parameters": params,
        "notes": notes,
        "source_name": source_name,
    }
