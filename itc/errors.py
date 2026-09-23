"""Diagnostics and the three structurally different failure outcomes.

* :class:`InputIncomplete` -- semantic impossibility: the input is present but
  cannot be interpreted / made consistent for the requested computation.
* :class:`ResourceExhausted` -- the computation ran out of its declared budget
  (fit evaluations, integration points, material size ...).
* :class:`FormatCorrupt` -- the bytes / JSON structure itself is damaged and
  cannot be parsed at all.

Every diagnostic carries a *location* (JSON-pointer-ish path into the source
material) so callers never have to substitute empty values for failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Diagnostic:
    code: str
    message: str
    location: str = ""
    severity: str = "error"          # error | warning | info
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Diagnostic":
        return cls(**d)


class ITCError(Exception):
    """Base error; always carries structured diagnostics."""

    kind = "itc_error"

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": str(self),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class FormatCorrupt(ITCError):
    """Bytes / container cannot be parsed (truncated, wrong magic, bad JSON)."""

    kind = "format_corrupt"


class SemanticImpossibility(ITCError):
    """Input parses but is meaningless in domain terms."""

    kind = "semantic_impossibility"


# Alias used by the task statement: incomplete input -> positional diagnostics.
InputIncomplete = SemanticImpossibility


class ResourceExhausted(ITCError):
    """Declared resource budget ran out before a result was produced."""

    kind = "resource_exhausted"


def collect_diagnostics(items: list[Any]) -> list[Diagnostic]:
    """Flatten diagnostics embedded in derived payloads."""
    out: list[Diagnostic] = []
    for item in items or []:
        if isinstance(item, Diagnostic):
            out.append(item)
        elif isinstance(item, dict) and "code" in item and "message" in item:
            out.append(Diagnostic.from_dict(item))
    return out


def has_error(diagnostics: list[Diagnostic] | list[dict[str, Any]]) -> bool:
    for d in diagnostics:
        sev = getattr(d, "severity", None) or d.get("severity")
        if sev == "error":
            return True
    return False
