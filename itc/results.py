from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostic:
    """A located diagnostic. Never silently dropped, never faked with defaults."""

    code: str
    message: str
    position: str

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "position": self.position}


class Result:
    ok: bool = False


@dataclass
class Ok(Result):
    value: Any = None
    diagnostics: list = field(default_factory=list)
    ok: bool = True


@dataclass
class ResourceExhausted(Result):
    """Budget (samples / model evaluations) ran out. Distinct from bad input."""

    detail: str = ""
    budget: str = ""


@dataclass
class FormatCorrupt(Result):
    """Bytes on disk could not be parsed. Position points at the corruption."""

    detail: str = ""
    position: str = ""


@dataclass
class SemanticUnsolvable(Result):
    """Input parsed fine but the request cannot be decided (missing materials,
    contradictory references, ...). Diagnostics carry positions."""

    diagnostics: list = field(default_factory=list)
