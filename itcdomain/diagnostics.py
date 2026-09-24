"""Structured, position-bearing diagnostics.

Nothing is ever represented as a silent ``None``/default: incomplete inputs
return diagnostics that point back to the exact source location.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "error"
    location: str | None = None
    line: int | None = None
    column: int | None = None
    fragment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticBag:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def add(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def error(self, code: str, message: str, **kwargs: Any) -> Diagnostic:
        diag = Diagnostic(code=code, message=message, severity="error", **kwargs)
        self.diagnostics.append(diag)
        return diag

    def warning(self, code: str, message: str, **kwargs: Any) -> Diagnostic:
        diag = Diagnostic(code=code, message=message, severity="warning", **kwargs)
        self.diagnostics.append(diag)
        return diag

    def info(self, code: str, message: str, **kwargs: Any) -> Diagnostic:
        diag = Diagnostic(code=code, message=message, severity="info", **kwargs)
        self.diagnostics.append(diag)
        return diag

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> str:
        return json.dumps([d.to_dict() for d in self.diagnostics], ensure_ascii=False)
