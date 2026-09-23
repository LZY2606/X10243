"""Domain model for raw ITC materials and parsed-but-not-normalized payloads.

Parsing keeps the original textual representation of every cell next to its
normalized numeric value.  Semantic evaluation (gaps, windows, binding) happens
in dedicated modules so that this layer never silently "fixes" data.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import FormatCorrupt
from .units import parse_quantity, parse_temperature, split_number_unit


MATERIAL_KINDS = (
    "power", "injections", "concentrations", "conditions",
    "baseline", "blank_power", "blank_injections",
)


@dataclass
class ParsedSeries:
    kind: str
    role: str
    source_filename: str
    time_basis: str
    power_unit: str
    samples: list[dict[str, Any]]
    header: list[str]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _read_rows(raw_text: str, source_filename: str):
    try:
        reader = csv.reader(io.StringIO(raw_text))
        rows = [row for row in reader]
    except csv.Error as exc:  # pragma: no cover - defensive
        raise FormatCorrupt(
            f"CSV tokenizer failed in {source_filename}",
            [{"code": "csv_error", "detail": str(exc),
              "position": {"file": source_filename}}])
    rows = [r for r in rows if not all(c.strip() == "" for c in r)]
    if not rows:
        raise FormatCorrupt(
            f"{source_filename} is empty",
            [{"code": "empty_file", "position": {"file": source_filename}}])
    return rows


def _header_units(header: list[str]):
    """Extract declared units from headers like ``time [s]``."""
    units = {}
    clean = []
    for col in header:
        if "[" in col and "]" in col:
            name = col[:col.index("[")].strip()
            unit = col[col.index("[") + 1:col.rindex("]")].strip()
        else:
            name, unit = col.strip(), None
        units[name] = unit
        clean.append(name)
    return clean, units


def parse_series(raw_text: str, kind: str, role: str,
                 source_filename: str) -> ParsedSeries:
    """Parse a power/baseline/blank power CSV.

    Expected header: ``time [<time unit>],power [<power unit>]``.  Every row
    keeps both its raw text and, where convertible, normalized values in
    seconds and µcal/s.
    """
    rows = _read_rows(raw_text, source_filename)
    header = rows[0]
    names, units = _header_units(header)
    if [n.lower() for n in names] != ["time", "power"]:
        raise FormatCorrupt(
            f"{source_filename}: expected header 'time [..],power [..]'",
            [{"code": "bad_header", "header": header,
              "position": {"file": source_filename, "row": 1}}])
    t_unit = units.get("time")
    p_unit = units.get("power")
    if not t_unit or not p_unit:
        raise FormatCorrupt(
            f"{source_filename}: units must be declared in the header",
            [{"code": "missing_header_unit", "header": header,
              "position": {"file": source_filename, "row": 1}}])

    samples: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    prev_t = None
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise FormatCorrupt(
                f"{source_filename} row {idx}: expected 2 columns",
                [{"code": "bad_arity", "columns": len(row),
                  "position": {"file": source_filename, "row": idx}}])
        t_raw, p_raw = row[0].strip(), row[1].strip()
        missing = t_raw == "" or p_raw == ""
        tq = parse_quantity(t_raw, t_unit, "time", "experiment_clock(s)",
                            {"file": source_filename, "row": idx, "column": "time"})
        pq = parse_quantity(p_raw, p_unit, "power", "sample_power(µcal/s)",
                            {"file": source_filename, "row": idx, "column": "power"})
        row_diag = list(tq.diagnostics) + list(pq.diagnostics)
        for d in row_diag:
            diagnostics.append(d)
        t_order_bad = (not missing and tq.value is not None and prev_t is not None
                       and tq.value < prev_t)
        if t_order_bad:
            diagnostics.append({
                "code": "non_monotonic_time",
                "row": idx,
                "position": {"file": source_filename, "row": idx, "column": "time"},
                "detail": "sample time is earlier than previous sample",
            })
        if not missing and tq.value is not None:
            prev_t = tq.value
        samples.append({
            "row": idx,
            "time_raw": t_raw,
            "power_raw": p_raw,
            "time_s": tq.value,
            "power_ucal_s": pq.value,
            "missing": missing,
            "diagnostics": row_diag,
        })
    return ParsedSeries(kind=kind, role=role, source_filename=source_filename,
                        time_basis="experiment_clock(s)", power_unit="µcal/s",
                        samples=samples, header=header, diagnostics=diagnostics)


@dataclass
class ParsedInjections:
    kind: str
    role: str
    source_filename: str
    injections: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def parse_injections(raw_text: str, kind: str, role: str,
                     source_filename: str) -> ParsedInjections:
    rows = _read_rows(raw_text, source_filename)
    header = rows[0]
    names, units = _header_units(header)
    expected = ["injection", "time", "volume"]
    if [n.lower() for n in names] != expected:
        raise FormatCorrupt(
            f"{source_filename}: expected header 'injection,time [..],volume [..]'",
            [{"code": "bad_header", "header": header,
              "position": {"file": source_filename, "row": 1}}])
    if not units.get("time") or not units.get("volume"):
        raise FormatCorrupt(
            f"{source_filename}: time and volume units required in header",
            [{"code": "missing_header_unit", "header": header,
              "position": {"file": source_filename, "row": 1}}])
    injections, diagnostics = [], []
    seen = set()
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            raise FormatCorrupt(
                f"{source_filename} row {idx}: expected 3 columns",
                [{"code": "bad_arity", "columns": len(row),
                  "position": {"file": source_filename, "row": idx}}])
        i_raw, t_raw, v_raw = (c.strip() for c in row)
        pos = {"file": source_filename, "row": idx}
        try:
            index = int(i_raw)
        except ValueError:
            raise FormatCorrupt(
                f"{source_filename} row {idx}: injection index not integral",
                [{"code": "bad_index", "raw": i_raw, "position": pos}])
        if index in seen:
            raise FormatCorrupt(
                f"{source_filename} row {idx}: duplicate injection index {index}",
                [{"code": "duplicate_index", "index": index, "position": pos}])
        seen.add(index)
        tq = parse_quantity(t_raw, units["time"], "time",
                            "experiment_clock(s)", {**pos, "column": "time"})
        vq = parse_quantity(v_raw, units["volume"], "volume", None,
                            {**pos, "column": "volume"})
        diagnostics.extend(tq.diagnostics)
        diagnostics.extend(vq.diagnostics)
        injections.append({
            "index": index,
            "time_raw": t_raw,
            "volume_raw": v_raw,
            "time_s": tq.value,
            "volume_ul": vq.value,
            "time_unit": tq.unit,
            "volume_unit": vq.unit,
            "diagnostics": list(tq.diagnostics) + list(vq.diagnostics),
        })
    injections.sort(key=lambda r: r["index"])
    return ParsedInjections(kind=kind, role=role, source_filename=source_filename,
                            injections=injections, diagnostics=diagnostics)


@dataclass
class ParsedConcentrations:
    kind: str = "concentrations"
    source_filename: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def lookup(self, role: str, compound: str | None = None):
        for rec in self.records:
            if rec["role"] == role and (compound is None or rec["compound"] == compound):
                return rec
        return None


def parse_concentrations(raw_text: str, source_filename: str) -> ParsedConcentrations:
    rows = _read_rows(raw_text, source_filename)
    header = rows[0]
    names, units = _header_units(header)
    if [n.lower() for n in names] != ["role", "compound", "concentration"]:
        raise FormatCorrupt(
            f"{source_filename}: expected 'role,compound,concentration [..]'",
            [{"code": "bad_header", "header": header,
              "position": {"file": source_filename, "row": 1}}])
    unit = units.get("concentration")
    if not unit:
        raise FormatCorrupt(
            f"{source_filename}: concentration unit required in header",
            [{"code": "missing_header_unit", "header": header,
              "position": {"file": source_filename, "row": 1}}])
    records, diagnostics = [], []
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            raise FormatCorrupt(
                f"{source_filename} row {idx}: expected 3 columns",
                [{"code": "bad_arity", "columns": len(row),
                  "position": {"file": source_filename, "row": idx}}])
        role, compound, c_raw = (c.strip() for c in row)
        cq = parse_quantity(c_raw, unit, "concentration", None,
                            {"file": source_filename, "row": idx,
                             "column": "concentration"})
        diagnostics.extend(cq.diagnostics)
        records.append({
            "role": role,
            "compound": compound,
            "raw": c_raw,
            "unit": cq.unit,
            "value_m": cq.value,
            "diagnostics": list(cq.diagnostics),
        })
    return ParsedConcentrations(source_filename=source_filename, records=records,
                                diagnostics=diagnostics)


@dataclass
class ParsedConditions:
    kind: str = "conditions"
    source_filename: str = ""
    fields_: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def get(self, name: str):
        for f in self.fields_:
            if f["name"] == name:
                return f
        return None


def parse_conditions(raw_text: str, source_filename: str) -> ParsedConditions:
    rows = _read_rows(raw_text, source_filename)
    header = rows[0]
    names, _units = _header_units(header)
    if [n.lower() for n in names] != ["field", "value", "note"]:
        raise FormatCorrupt(
            f"{source_filename}: expected 'field,value [..],note'",
            [{"code": "bad_header", "header": header,
              "position": {"file": source_filename, "row": 1}}])
    fields_, diagnostics = [], []
    declared = _units.get("value")
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            raise FormatCorrupt(
                f"{source_filename} row {idx}: expected 3 columns",
                [{"code": "bad_arity", "columns": len(row),
                  "position": {"file": source_filename, "row": idx}}])
        name, raw_val, note = (c.strip() for c in row)
        entry: dict[str, Any] = {"name": name, "raw": raw_val, "note": note}
        if name == "temperature":
            q = parse_temperature(raw_val, declared, position={
                "file": source_filename, "row": idx, "column": "value"})
            entry.update({"unit": q.unit, "value": q.value,
                          "canonical_unit": "K"})
            diagnostics.extend(q.diagnostics)
        elif name == "cell_volume":
            q = parse_quantity(raw_val, declared, "volume",
                               position={"file": source_filename, "row": idx,
                                         "column": "value"})
            entry.update({"unit": q.unit, "value": q.value,
                          "canonical_unit": "µL"})
            diagnostics.extend(q.diagnostics)
        else:
            num, tok = split_number_unit(raw_val)
            entry.update({"unit": tok, "value": float(num) if num else None,
                          "canonical_unit": tok})
            if num is None and raw_val != "":
                diagnostics.append({"code": "unparseable_number", "raw": raw_val,
                                    "position": {"file": source_filename, "row": idx}})
        fields_.append(entry)
    return ParsedConditions(source_filename=source_filename, fields_=fields_,
                            diagnostics=diagnostics)
