"""Parser for the lab's `.itc` worksheet format (domain-specific, offline).

An `.itc` file is a sectioned text worksheet. Each section maps to one
material slot so that power sampling, injection schedule, concentrations,
temperature, cell volume, baseline and blank curves are versioned separately.

    ITC-WORKSHEET 1
    SECTION POWER
    META time_basis=seconds_since_run_start
    META filter_period_s=5
    TABLE time_s power_uJ_per_s
    0.0 1.02
    ...
    SECTION INJECTIONS
    TABLE index volume_uL time_s
    SECTION CONCENTRATIONS
    META cell_concentration_mM=0.05
    META syringe_concentration_mM=0.5
    SECTION TEMPERATURE
    META value_K=298.15
    SECTION CELL_VOLUME
    META value_uL=1400
    SECTION BASELINE
    TABLE time_s power_uJ_per_s
    SECTION BLANK
    META blank_id=blank-buffer-01
    TABLE index volume_uL time_s
    TABLE time_s power_uJ_per_s
    END

Raw text of every cell is preserved (raw_repr). Unconvertible values stay in
place and produce positioned diagnostics; they are never dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .quantities import Quantity, parse_header, parse_quantity

MAGIC = "ITC-WORKSHEET"
FORMAT_VERSION = "1"

# slot names (stable identities for materials)
SLOT_POWER = "power"
SLOT_INJECTIONS = "injections"
SLOT_CONCENTRATIONS = "concentrations"
SLOT_TEMPERATURE = "temperature"
SLOT_CELL_VOLUME = "cell_volume"
SLOT_BASELINE = "baseline"
SLOT_BLANK_PREFIX = "blank:"  # blank:<blank_id>

SECTION_TO_SLOT = {
    "POWER": SLOT_POWER,
    "INJECTIONS": SLOT_INJECTIONS,
    "CONCENTRATIONS": SLOT_CONCENTRATIONS,
    "TEMPERATURE": SLOT_TEMPERATURE,
    "CELL_VOLUME": SLOT_CELL_VOLUME,
    "BASELINE": SLOT_BASELINE,
}


class FormatCorruptError(Exception):
    """Raised when the file structure itself is broken (distinct from
    semantic problems, which become diagnostics)."""

    def __init__(self, message: str, line: Optional[int] = None):
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


@dataclass
class Diagnostic:
    code: str
    message: str
    pos: Optional[str] = None  # "SECTION:row" or "SECTION:META:key"
    severity: str = "error"  # error | warning

    def to_dict(self):
        return {"code": self.code, "message": self.message, "pos": self.pos,
                "severity": self.severity}


@dataclass
class TableBlock:
    headers: list  # list[str]
    header_units: list  # list[Optional[str]]
    rows: list = field(default_factory=list)  # list[list[Quantity]]
    start_line: int = 0


@dataclass
class SectionBlock:
    name: str
    meta: dict = field(default_factory=dict)      # key -> raw string
    meta_pos: dict = field(default_factory=dict)  # key -> "SEC:META:key"
    tables: list = field(default_factory=list)    # list[TableBlock]
    start_line: int = 0


@dataclass
class Worksheet:
    sections: dict = field(default_factory=dict)  # name -> SectionBlock
    diagnostics: list = field(default_factory=list)  # list[Diagnostic]
    source_name: str = ""


def _diag(diags, code, message, pos=None, severity="error"):
    diags.append(Diagnostic(code=code, message=message, pos=pos, severity=severity))


def parse_worksheet(text: str, source_name: str = "") -> Worksheet:
    """Parse text into a Worksheet. Raises FormatCorruptError on structural
    damage. Cell-level problems become diagnostics."""
    lines = text.splitlines()
    # find magic
    first = None
    for i, ln in enumerate(lines):
        if ln.strip() == "":
            continue
        first = (i, ln.strip())
        break
    if first is None:
        raise FormatCorruptError("empty file")
    i, content = first
    parts = content.split()
    if parts[0] != MAGIC:
        raise FormatCorruptError(f"missing magic header '{MAGIC}'", i + 1)
    if len(parts) < 2 or parts[1] != FORMAT_VERSION:
        raise FormatCorruptError(f"unsupported format version: {content}", i + 1)

    ws = Worksheet(source_name=source_name)
    current: Optional[SectionBlock] = None
    current_table: Optional[TableBlock] = None
    saw_end = False
    for idx in range(i + 1, len(lines)):
        ln = lines[idx]
        stripped = ln.strip()
        lineno = idx + 1
        if stripped == "" or stripped.startswith("#"):
            continue
        if stripped == "END":
            saw_end = True
            break
        if stripped.startswith("SECTION "):
            name = stripped.split(None, 1)[1].strip()
            if not name:
                raise FormatCorruptError("empty section name", lineno)
            if name in ws.sections:
                raise FormatCorruptError(f"duplicate section {name}", lineno)
            current = SectionBlock(name=name, start_line=lineno)
            ws.sections[name] = current
            current_table = None
            continue
        if current is None:
            raise FormatCorruptError("content outside any section", lineno)
        if stripped.startswith("META "):
            body = stripped[len("META "):]
            if "=" not in body:
                raise FormatCorruptError("META line must be key=value", lineno)
            key, _, val = body.partition("=")
            key = key.strip()
            current.meta[key] = val.strip()
            current.meta_pos[key] = f"{current.name}:META:{key}"
            continue
        if stripped.startswith("TABLE "):
            headers = stripped[len("TABLE "):].split()
            if not headers:
                raise FormatCorruptError("TABLE without headers", lineno)
            names, units = [], []
            for h in headers:
                n, u = parse_header(h)
                names.append(n)
                units.append(u)
            current_table = TableBlock(headers=names, header_units=units,
                                       start_line=lineno)
            current.tables.append(current_table)
            continue
        # data row
        if current_table is None:
            raise FormatCorruptError("data row before TABLE header", lineno)
        cells = stripped.split("\t") if "\t" in stripped else stripped.split()
        if len(cells) != len(current_table.headers):
            _diag(ws.diagnostics, "row_arity",
                  f"row has {len(cells)} cells, header has {len(current_table.headers)}",
                  pos=f"{current.name}:{lineno}")
            # pad/truncate but keep raw cells in place
        row = []
        for ci, cell in enumerate(cells):
            unit = current_table.header_units[ci] if ci < len(current_table.header_units) else None
            q = parse_quantity(cell, default_unit=unit, pos=f"{current.name}:{lineno}")
            if not q.ok:
                _diag(ws.diagnostics, "bad_value",
                      f"cannot convert value {cell!r}: {q.error}",
                      pos=f"{current.name}:{lineno}")
            row.append(q)
        current_table.rows.append(row)
    if not saw_end:
        raise FormatCorruptError("missing END terminator (file truncated?)")
    return ws


def worksheet_to_material_drafts(ws: Worksheet):
    """Convert a parsed worksheet into material drafts:
    list of {slot, content(dict, JSON-able), diagnostics}.
    Content keeps raw_repr for every value plus canonical normalisation hints.
    """
    drafts = []
    for name, sec in ws.sections.items():
        if name == "BLANK":
            blank_id = sec.meta.get("blank_id", "blank")
            slot = SLOT_BLANK_PREFIX + blank_id
        else:
            slot = SECTION_TO_SLOT.get(name)
            if slot is None:
                ws.diagnostics.append(Diagnostic(
                    code="unknown_section", message=f"unknown section {name}",
                    pos=f"{name}:0", severity="warning"))
                continue
        content = {
            "slot": slot,
            "source": ws.source_name,
            "meta": dict(sec.meta),
            "meta_pos": dict(sec.meta_pos),
            "tables": [],
        }
        for tb in sec.tables:
            content["tables"].append({
                "headers": tb.headers,
                "header_units": tb.header_units,
                "rows": [[{"raw": q.raw, "value": q.value, "unit": q.unit,
                           "ok": q.ok, "error": q.error, "pos": q.pos}
                          for q in row] for row in tb.rows],
            })
        drafts.append({"slot": slot, "content": content,
                       "diagnostics": [d.to_dict() for d in ws.diagnostics]})
    return drafts
