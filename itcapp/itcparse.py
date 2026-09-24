"""Parser for the .itc domain material format.

Grammar (line based):
    # comment
    key: value            (header field, before any @section)
    @section              (start of a table section)
    col1,col2,...         (first line of a section: column names)
    v1,v2,...             (data rows; blank cells allowed)

Format corruption (bad grammar, wrong header, ragged rows, non-numeric
cells in numeric columns) rejects the whole import with located
diagnostics. Unknown units are NOT format errors: they parse fine and the
derived layer reports UNIT_UNKNOWN while keeping the raw value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .errors import Diagnostic, FormatCorruptedError

REQUIRED_HEADER = ("format", "material", "version")


@dataclass
class Section:
    name: str
    columns: List[str]
    rows: List[List[str]]  # raw string cells
    line: int  # line number of the @section header


@dataclass
class ItcDocument:
    header: Dict[str, str]
    sections: Dict[str, Section]
    raw_text: str


def parse_itc(text: str, source_name: str = "<input>") -> ItcDocument:
    if not isinstance(text, str) or not text.strip():
        raise FormatCorruptedError(
            "empty material document",
            [Diagnostic("EMPTY_DOCUMENT", "error", f"{source_name}:1",
                        "document is empty")],
        )
    header: Dict[str, str] = {}
    sections: Dict[str, Section] = {}
    current: Optional[Section] = None
    diags: List[Diagnostic] = []

    lines = text.splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            name = line[1:].strip()
            if not name:
                diags.append(Diagnostic("BAD_SECTION", "error",
                                        f"{source_name}:{lineno}",
                                        "section name is empty"))
                continue
            if name in sections:
                diags.append(Diagnostic("DUP_SECTION", "error",
                                        f"{source_name}:{lineno}",
                                        f"duplicate section @{name}"))
                continue
            current = Section(name=name, columns=[], rows=[], line=lineno)
            sections[name] = current
            continue
        if current is None:
            if ":" not in line:
                diags.append(Diagnostic("BAD_HEADER_LINE", "error",
                                        f"{source_name}:{lineno}",
                                        f"expected 'key: value', got {line!r}"))
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if not key:
                diags.append(Diagnostic("BAD_HEADER_LINE", "error",
                                        f"{source_name}:{lineno}",
                                        "empty header key"))
                continue
            header[key] = value.strip()
            continue
        # inside a section
        cells = [c.strip() for c in line.split(",")]
        if not current.columns:
            if any(not c for c in cells):
                diags.append(Diagnostic("BAD_COLUMNS", "error",
                                        f"{source_name}:{lineno}",
                                        "column names must be non-empty"))
                continue
            current.columns = cells
        else:
            if len(cells) != len(current.columns):
                diags.append(Diagnostic("RAGGED_ROW", "error",
                                        f"{source_name}:{lineno}",
                                        f"row has {len(cells)} cells, expected "
                                        f"{len(current.columns)}"))
                continue
            current.rows.append(cells)

    for key in REQUIRED_HEADER:
        if key not in header:
            diags.append(Diagnostic("MISSING_HEADER", "error",
                                    f"{source_name}:1",
                                    f"required header field {key!r} missing"))
    if header.get("format") not in (None, "itc-material/1"):
        diags.append(Diagnostic("BAD_FORMAT", "error", f"{source_name}:1",
                                f"unsupported format {header.get('format')!r}"))
    if diags:
        raise FormatCorruptedError(
            f"material document {source_name} is corrupt "
            f"({len(diags)} problem(s))", diags)
    for sec in sections.values():
        if not sec.columns:
            raise FormatCorruptedError(
                f"section @{sec.name} has no columns",
                [Diagnostic("EMPTY_SECTION", "error",
                            f"{source_name}:{sec.line}",
                            f"section @{sec.name} declares no columns")])
    return ItcDocument(header=header, sections=sections, raw_text=text)


def require_sections(doc: ItcDocument, names: List[str], source_name: str) -> None:
    missing = [n for n in names if n not in doc.sections]
    if missing:
        raise FormatCorruptedError(
            f"material {source_name} misses required section(s): "
            + ", ".join(missing),
            [Diagnostic("MISSING_SECTION", "error", f"{source_name}:1",
                        f"required section @{n} missing") for n in missing],
        )
