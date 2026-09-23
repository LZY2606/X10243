"""Distinct failure outcomes for the ITC pipeline.

Three *different* result kinds are required by the domain rules:

* ``format_corrupt``       - the byte/text representation itself is malformed
* ``semantic_unresolvable``- representation parsed, but meaning cannot be fixed
* ``resource_exhausted``   - a declared budget (points, iterations...) ran out

Incomplete input never silently becomes ``None``/defaults: callers get a
:class:`Diagnostic` carrying a position instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None
    target: str | None = None  # stable object id the diagnostic is attached to
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ITCError(Exception):
    code = "itc_error"
    http_status = 422

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorrupt(ITCError):
    code = "format_corrupt"
    http_status = 422


class SemanticUnresolvable(ITCError):
    code = "semantic_unresolvable"
    http_status = 422


class ResourceExhausted(ITCError):
    code = "resource_exhausted"
    http_status = 507
