"""Parser for the laboratory ITC text format (``.itc``).

A document is a sectioned, line-oriented text file.  Example::

    # experiment IT-2026-09-24-A (free form comment)
    @meta
      kind: power
      run: IT-A
      temperature: 298.15 K
      power_unit: uW
      time_unit: s
      time_basis: elapsed seconds since injection 1 (t=0)
    @injections
      # index  time   volume
      1        0.0    2.0 uL
    @series
      # time  power
      0.0     12.1
    @gaps
      240.0 245.0    instrument feed paused
    @baseline
      0.0   10.0
    @blank_series
      0.0   4.0

Six material kinds are recognized: ``power``, ``injections``,
``concentration``, ``conditions``, ``baseline`` and ``blank``.  Every token
is preserved verbatim in the parsed document; conversion to canonical units
happens later (:mod:`itc.units`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic, FormatCorrupted
from .units import parse_quantity

_SECTIONS = {
    "meta",
    "injections",
    "series",
    "gaps",
    "baseline",
    "concentration",
    "conditions",
    "blank_series",
}
_KINDS = {"power", "injections", "concentration", "conditions", "baseline", "blank"}
_SERIES_SECTIONS = {"series", "baseline", "blank_series"}


@dataclass
class RawRow:
    line: int
    tokens: list[str]
    raw: str
    section: str
    comment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "tokens": self.tokens,
            "raw": self.raw,
            "section": self.section,
            "comment": self.comment,
        }


@dataclass
class Document:
    meta: dict[str, str] = field(default_factory=dict)
    meta_lines: dict[str, int] = field(default_factory=dict)
    sections: dict[str, list[RawRow]] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    file: str = "input.itc"

    def kind(self) -> str | None:
        kind = self.meta.get("kind")
        return kind if kind in _KINDS else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "meta_lines": dict(self.meta_lines),
            "sections": {k: [r.to_dict() for r in v] for k, v in self.sections.items()},
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "file": self.file,
        }


def _strip_comment(line: str) -> tuple[str, str | None]:
    hash_idx = line.find("#")
    if hash_idx == -1:
        return line, None
    return line[:hash_idx], line[hash_idx + 1 :].strip()


def parse_document(text: str, file: str = "input.itc") -> Document:
    """Structural parse.  Raises :class:`FormatCorrupted` on bad shape."""
    doc = Document(file=file)
    current: str | None = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        code, comment = _strip_comment(raw_line)
        stripped = code.strip()
        if not stripped:
            continue
        if stripped.startswith("@"):
            name = stripped[1:].strip()
            if name not in _SECTIONS:
                raise FormatCorrupted(
                    f"unknown section @{name}",
                    [
                        Diagnostic(
                            code="F001",
                            message=f"unknown section @{name}; expected one of "
                            f"{sorted(_SECTIONS)}",
                            file=file,
                            line=lineno,
                            section=name,
                            raw=raw_line.rstrip(),
                        )
                    ],
                )
            current = name
            doc.sections.setdefault(name, [])
            continue
        if current is None:
            if ":" in stripped:
                current = "meta"
                doc.sections.setdefault("meta", [])
            else:
                raise FormatCorrupted(
                    "content before any @section",
                    [
                        Diagnostic(
                            code="F002",
                            message="data line appears before any @section/@meta header",
                            file=file,
                            line=lineno,
                            raw=raw_line.rstrip(),
                        )
                    ],
                )
        if current == "meta":
            key, _, value = stripped.partition(":")
            key = key.strip()
            if not key or " " in key:
                raise FormatCorrupted(
                    "malformed meta line",
                    [
                        Diagnostic(
                            code="F003",
                            message="meta line must be 'key: value'",
                            file=file,
                            line=lineno,
                            raw=raw_line.rstrip(),
                        )
                    ],
                )
            doc.meta[key] = value.strip()
            doc.meta_lines[key] = lineno
        else:
            tokens = stripped.split()
            if len(tokens) < 2:
                doc.diagnostics.append(
                    Diagnostic(
                        code="F010",
                        message=f"@{current} row needs at least 2 columns, got {len(tokens)}",
                        severity="warning",
                        file=file,
                        line=lineno,
                        section=current,
                        raw=raw_line.rstrip(),
                    )
                )
                continue
            doc.sections.setdefault(current, []).append(
                RawRow(line=lineno, tokens=tokens, raw=stripped, section=current,
                       comment=comment)
            )
    if not doc.meta.get("kind"):
        raise FormatCorrupted(
            "missing kind in @meta",
            [
                Diagnostic(
                    code="F004",
                    message="meta 'kind' is required (power|injections|concentration|"
                    "conditions|baseline|blank)",
                    file=file,
                    section="meta",
                )
            ],
        )
    if doc.kind() is None:
        raise FormatCorrupted(
            f"unknown kind {doc.meta.get('kind')!r}",
            [
                Diagnostic(
                    code="F005",
                    message=f"kind must be one of {sorted(_KINDS)}",
                    file=file,
                    section="meta",
                    raw=doc.meta.get("kind"),
                )
            ],
        )
    return doc


def _q(tokens: list[str], start: int, dimension: str) -> Quantity:
    return parse_quantity(" ".join(tokens[start:]), dimension)


def _num(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None
