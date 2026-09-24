"""Versioned, immutable source materials for an ITC experiment.

Kinds, each versioned independently:

* ``power``      -- raw power sampling (with its time basis/origin)
* ``schedule``   -- injection volume and time table
* ``conditions`` -- cell/syringe concentration, temperature, cell volume
* ``baseline``   -- baseline curve (kept apart from the sample curve)
* ``blank``      -- blank control curve + its own injection table

Import preserves the original textual representation; normalization happens
only in the derived layer.  Rows that cannot be converted are kept in place
with diagnostics instead of being dropped.
"""

from __future__ import annotations

import csv
import io
import json

from .diagnostics import (
    BUDGET_EXHAUSTED,
    FORMAT_CORRUPT,
    SEMANTIC_INVALID,
    diagnostic,
)
from .hashing import digest
from .units import parse_quantity

DEFAULT_SAMPLE_BUDGET = 200_000
VALID_TIME_BASIS = ("run_elapsed",)


class Material:
    """One immutable version of one source material kind."""

    __slots__ = ("kind", "version_id", "payload", "origin_label")

    def __init__(self, kind, payload, origin_label="import"):
        self.kind = kind
        self.payload = payload
        self.origin_label = origin_label
        self.version_id = digest(f"mat_{kind}", payload)

    def to_dict(self):
        return {
            "kind": self.kind,
            "version_id": self.version_id,
            "origin_label": self.origin_label,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        m.kind = d["kind"]
        m.version_id = d["version_id"]
        m.origin_label = d.get("origin_label", "import")
        m.payload = d["payload"]
        return m


# --------------------------------------------------------------------------
# low level helpers
# --------------------------------------------------------------------------

def _read_table(text, expected, kind):
    """Parse a CSV with a unit-bearing header.

    expected: sequence of (column, dimension).  Returns (rows, diagnostics).
    Raises FormatCorrupt-like diagnostics via the list for shape problems.
    """
    diags = []
    try:
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
    except (csv.Error, UnicodeDecodeError) as exc:
        return None, [diagnostic(FORMAT_CORRUPT, f"{kind} CSV unreadable: {exc}")]
    if header is None:
        return None, [diagnostic(FORMAT_CORRUPT, f"{kind} CSV is empty")]
    header = [h.strip() for h in header]
    if header[: len(expected)] != [c for c, _ in expected]:
        return None, [
            diagnostic(
                FORMAT_CORRUPT,
                f"{kind} header mismatch: expected {[c for c, _ in expected]}, got {header}",
            )
        ]
    extra = header[len(expected):]
    rows = []
    try:
        for line_no, raw_row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw_row):
                continue
            if len(raw_row) < len(expected):
                diags.append(
                    diagnostic(
                        FORMAT_CORRUPT,
                        f"{kind} line {line_no}: expected {len(expected)} columns, got {len(raw_row)}",
                        line=line_no,
                    )
                )
                continue
            rows.append((line_no, raw_row, extra))
    except csv.Error as exc:
        diags.append(diagnostic(FORMAT_CORRUPT, f"{kind} CSV parse failure: {exc}"))
    return rows, diags


def _parse_cell(token, dimension, kind, line_no, col, diags):
    q, diag = parse_quantity(token, dimension, position=f"{kind}:{line_no}:{col}")
    if diag:
        diags.append(diag)
    if q is None:
        return {"raw": str(token).strip(), "number": None, "unit": None,
                "dimension": dimension, "canonical_unit": None, "value": None}
    d = q.to_dict()
    return d


# --------------------------------------------------------------------------
# material parsers
# --------------------------------------------------------------------------

def parse_power(text, time_basis="run_elapsed", origin_label="power.dat",
                budget=DEFAULT_SAMPLE_BUDGET):
    rows, diags = _read_table(
        text, [("time_s", "time"), ("power_uW", "power")], "power"
    )
    samples = []
    if rows is None:
        return None, diags
    for line_no, raw_row, _ in rows:
        samples.append({
            "line": line_no,
            "time": _parse_cell(raw_row[0], "time", "power", line_no, "time_s", diags),
            "power": _parse_cell(raw_row[1], "power", "power", line_no, "power_uW", diags),
        })
        if len(samples) > budget:
            return Material("power", {"time_basis": time_basis, "samples": samples,
                                      "origin_label": origin_label}), [
                diagnostic(BUDGET_EXHAUSTED,
                           f"power sample budget {budget} exceeded at line {line_no}",
                           line=line_no)]
    if time_basis not in VALID_TIME_BASIS:
        diags.append(diagnostic(SEMANTIC_INVALID,
                                f"unsupported time basis {time_basis!r}"))
    return Material("power", {"time_basis": time_basis, "samples": samples,
                              "origin_label": origin_label}), diags


def parse_schedule(text, origin_label="injections.csv", budget=100_000):
    rows, diags = _read_table(
        text, [("inject_index", None), ("time_s", "time"), ("volume_uL", "volume")],
        "schedule",
    )
    if rows is None:
        return None, diags
    injections = []
    seen_index = set()
    for line_no, raw_row, _ in rows:
        index_raw = raw_row[0].strip()
        try:
            index = int(index_raw)
        except ValueError:
            diags.append(diagnostic(SEMANTIC_INVALID,
                                    f"injection index not an integer: {index_raw!r}",
                                    line=line_no))
            index = None
        if index is not None:
            if index in seen_index:
                diags.append(diagnostic(SEMANTIC_INVALID,
                                        f"duplicate injection index {index}",
                                        line=line_no))
            seen_index.add(index)
        injections.append({
            "line": line_no,
            "index": index,
            "index_raw": index_raw,
            "time": _parse_cell(raw_row[1], "time", "schedule", line_no, "time_s", diags),
            "volume": _parse_cell(raw_row[2], "volume", "schedule", line_no, "volume_uL", diags),
        })
        if len(injections) > budget:
            return Material("schedule", {"injections": injections,
                                         "origin_label": origin_label}), [
                diagnostic(BUDGET_EXHAUSTED,
                           f"injection budget {budget} exceeded at line {line_no}",
                           line=line_no)]
    return Material("schedule", {"injections": injections,
                                 "origin_label": origin_label}), diags


def parse_conditions(text, origin_label="conditions.json"):
    diags = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [diagnostic(FORMAT_CORRUPT,
                                 f"conditions.json invalid JSON at position {exc.pos}: {exc.msg}")]
    if not isinstance(data, dict):
        return None, [diagnostic(FORMAT_CORRUPT, "conditions.json must be an object")]
    fields = {
        "cell_concentration": "concentration",
        "syringe_concentration": "concentration",
        "temperature": "temperature",
        "cell_volume": "volume",
    }
    out = {"origin_label": origin_label, "time_basis": data.get("time_basis", "run_elapsed")}
    for key, dim in fields.items():
        token = data.get(key)
        if token is None:
            diags.append(diagnostic(SEMANTIC_INVALID, f"missing {key}", field=key))
            out[key] = None
            continue
        q, diag = parse_quantity(token, dim, position=f"conditions:{key}")
        if diag:
            diags.append(diag)
        out[key] = None if q is None else q.to_dict()
    if out["time_basis"] not in VALID_TIME_BASIS:
        diags.append(diagnostic(SEMANTIC_INVALID,
                                f"unsupported time basis {out['time_basis']!r}"))
    return Material("conditions", out), diags


def parse_baseline(text, origin_label="baseline.csv"):
    """Baseline curve, versioned independently of the sample power curve.

    Supported forms expressed by the data:
      * single anchor            -> constant baseline
      * two anchors              -> linear drift
      * three or more anchors    -> piecewise-linear baseline curve
    """
    rows, diags = _read_table(
        text, [("time_s", "time"), ("baseline_uW", "power")], "baseline"
    )
    if rows is None:
        return None, diags
    anchors = []
    for line_no, raw_row, _ in rows:
        anchors.append({
            "line": line_no,
            "time": _parse_cell(raw_row[0], "time", "baseline", line_no, "time_s", diags),
            "value": _parse_cell(raw_row[1], "power", "baseline", line_no, "baseline_uW", diags),
        })
    if not anchors:
        diags.append(diagnostic(SEMANTIC_INVALID, "baseline has no anchors"))
    form = {1: "constant", 2: "linear"}.get(len(anchors), "curve") if anchors else "empty"
    payload = {"anchors": anchors, "form": form, "origin_label": origin_label}
    return Material("baseline", payload), diags


def parse_blank(text, origin_label="blank.csv", budget=DEFAULT_SAMPLE_BUDGET):
    """Blank control: a power curve plus its own injection/energy table.

    Injection energies may be given directly (integrated blank heats);
    rows with unconvertible entries are retained with diagnostics.
    """
    rows, diags = _read_table(
        text,
        [("inject_index", None), ("time_s", "time"), ("volume_uL", "volume"),
         ("energy_uJ", "energy")],
        "blank",
    )
    if rows is None:
        return None, diags
    blanks = []
    for line_no, raw_row, _ in rows:
        index_raw = raw_row[0].strip()
        try:
            index = int(index_raw)
        except ValueError:
            diags.append(diagnostic(SEMANTIC_INVALID,
                                    f"blank injection index not an integer: {index_raw!r}",
                                    line=line_no))
            index = None
        blanks.append({
            "line": line_no,
            "index": index,
            "index_raw": index_raw,
            "time": _parse_cell(raw_row[1], "time", "blank", line_no, "time_s", diags),
            "volume": _parse_cell(raw_row[2], "volume", "blank", line_no, "volume_uL", diags),
            "energy": _parse_cell(raw_row[3], "energy", "blank", line_no, "energy_uJ", diags),
        })
        if len(blanks) > budget:
            return Material("blank", {"injections": blanks,
                                      "origin_label": origin_label}), [
                diagnostic(BUDGET_EXHAUSTED,
                           f"blank budget {budget} exceeded at line {line_no}",
                           line=line_no)]
    return Material("blank", {"injections": blanks, "origin_label": origin_label}), diags


PARSERS = {
    "power": parse_power,
    "schedule": parse_schedule,
    "conditions": parse_conditions,
    "baseline": parse_baseline,
    "blank": parse_blank,
}
