"""Raw series parsing for power / baseline / blank curves.

The parser preserves the original textual representation of every cell.
Missing samples become explicit *gaps* anchored at their time coordinate;
unconvertible values stay in place with a positional diagnostic. Nothing is
silently dropped for the sake of plotting.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic, FormatCorrupt, SemanticImpasse
from .units import Quantity, parse_quantity

GAP_MARKERS = {"", "na", "n/a", "nan", "null", "none", "gap", "missing", "--", "-"}


@dataclass
class RawTable:
    file: str
    header: list[str]
    rows: list[list[str]]           # original cells, one list per physical row

    def column_index(self, name: str) -> int:
        for i, h in enumerate(self.header):
            if h.strip().lower() == name.strip().lower():
                return i
        return -1


def parse_table(text: str, file: str) -> RawTable:
    """Parse CSV structure. Structural problems are :class:`FormatCorrupt`."""
    diagnostics: list[Diagnostic] = []
    reader = csv.reader(io.StringIO(text))
    rows: list[list[str]] = []
    for line_no, raw in enumerate(reader, start=1):
        if len(raw) == 0 or all(c.strip() == "" for c in raw):
            diagnostics.append(Diagnostic("empty_row", "empty row inside data block",
                                          file, line_no))
            continue
        rows.append(raw)
    if not rows:
        raise FormatCorrupt(f"{file}: file contains no rows",
                            [Diagnostic("empty_file", "no rows", file)])
    width = len(rows[0])
    bad = [d for d in diagnostics]
    for idx, r in enumerate(rows[1:], start=2):
        if len(r) != width:
            bad.append(Diagnostic("ragged_row",
                                  f"row has {len(r)} fields, header has {width}",
                                  file, idx, detail={"found": len(r), "expected": width}))
    if any(d.code == "ragged_row" for d in bad):
        raise FormatCorrupt(f"{file}: ragged CSV rows", bad)
    header = [c.strip() for c in rows[0]]
    if len(set(header)) != len(header):
        dupes = sorted({h for h in header if header.count(h) > 1})
        raise FormatCorrupt(f"{file}: duplicate header columns",
                            [Diagnostic("duplicate_column", "duplicate header",
                                        file, 1, detail={"columns": dupes})])
    return RawTable(file=file, header=header, rows=rows[1:])


@dataclass
class Sample:
    index: int                      # 0-based sample index in the source file
    row: int                        # 1-based physical row (header is row 1)
    raw: str                        # original text of the value cell
    t_raw: str
    t_s: float | None = None
    q: Quantity | None = None       # parsed value (may be unconvertible)
    kind: str = "sample"            # sample | gap | unconvertible | bad_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "row": self.row, "raw": self.raw,
            "t_raw": self.t_raw, "t_s": self.t_s, "kind": self.kind,
            "quantity": self.q.to_dict() if self.q else None,
        }


@dataclass
class SeriesData:
    file: str
    kind: str                       # power | baseline | blank
    time_column: str
    value_column: str
    time_unit: str
    value_unit: str | None
    samples: list[Sample]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    span: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file, "kind": self.kind,
            "time_column": self.time_column, "value_column": self.value_column,
            "time_unit": self.time_unit, "value_unit": self.value_unit,
            "span": list(self.span) if self.span else None,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "samples": [s.to_dict() for s in self.samples],
        }

    def time_array(self) -> tuple[list[float], list[int]]:
        """Return ``(times, indices)`` for samples with a valid time axis."""
        times, idx = [], []
        for s in self.samples:
            if s.kind != "bad_time" and s.t_s is not None:
                times.append(s.t_s)
                idx.append(s.index)
        return times, idx


def parse_series(text: str, file: str, kind: str, *,
                 time_column: str = "t", value_column: str = "power",
                 time_unit: str = "s", value_unit: str | None = "ucal/s") -> SeriesData:
    table = parse_table(text, file)
    diags: list[Diagnostic] = []
    ti = table.column_index(time_column)
    if ti < 0:
        raise SemanticImpasse(
            f"{file}: missing time column {time_column!r}",
            [Diagnostic("missing_column", "required time column absent",
                        file, 1, time_column, detail={"header": table.header})])
    vi = table.column_index(value_column)
    if vi < 0:
        raise SemanticImpasse(
            f"{file}: missing value column {value_column!r}",
            [Diagnostic("missing_column", "required value column absent",
                        file, 1, value_column, detail={"header": table.header})])
    samples: list[Sample] = []
    for i, row in enumerate(table.rows):
        physical_row = i + 2
        t_cell = row[ti].strip()
        v_cell = row[vi].strip()
        tq = parse_quantity(t_cell, "time", default_unit=time_unit,
                            file=file, row=physical_row, column=time_column)
        sample = Sample(index=i, row=physical_row, raw=row[vi], t_raw=t_cell)
        if not tq.convertible or tq.canonical is None:
            sample.kind = "bad_time"
            diags.append(Diagnostic(
                "bad_time", f"time value {t_cell!r} cannot be normalised",
                file, physical_row, time_column,
                detail={"raw": t_cell,
                        "nested": tq.diagnostic.to_dict() if tq.diagnostic else None}))
        else:
            sample.t_s = tq.canonical
        if v_cell.lower() in GAP_MARKERS:
            sample.kind = "gap" if sample.kind != "bad_time" else "bad_time"
        else:
            q = parse_quantity(v_cell, "power", default_unit=value_unit,
                               file=file, row=physical_row, column=value_column)
            sample.q = q
            if sample.kind == "sample" and not q.convertible:
                sample.kind = "unconvertible"
                diags.append(q.diagnostic or Diagnostic(
                    "unconvertible_sample", "value cannot be normalised",
                    file, physical_row, value_column,
                    detail={"t_s": sample.t_s, "raw": v_cell}))
        samples.append(sample)
    times = [s.t_s for s in self_check(samples)]
    span = (min(times), max(times)) if times else None
    return SeriesData(file=file, kind=kind, time_column=time_column,
                      value_column=value_column, time_unit=time_unit,
                      value_unit=value_unit, samples=samples,
                      diagnostics=diags, span=span)


def self_check(samples: list[Sample]) -> list[float]:
    return [s.t_s for s in samples if s.t_s is not None and s.kind != "bad_time"]


def numeric_segments(series: SeriesData) -> list[tuple[int, int]]:
    """Contiguous numeric (usable) sample-index ranges ``[a, b]`` inclusive.

    ``gap`` and ``unconvertible`` samples break segments; bad-time rows are
    excluded entirely.
    """
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for s in series.samples:
        usable = s.kind == "sample"
        if usable and start is None:
            start = s.index
        elif not usable and start is not None:
            segments.append((start, s.index - 1))
            start = None
    if start is not None:
        segments.append((start, series.samples[-1].index))
    return segments
