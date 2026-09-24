"""Raw material import.

Supported on-disk formats are deliberately domain-specific (not generic
spreadsheets):

* power series CSV with a ``# units:`` header and ``time,power`` rows.
  Missing samples are written as ``<time>,GAP[: reason]`` (or an empty power
  cell) and become explicit gap intervals; unconvertible cells stay in place
  with a diagnostic rather than being dropped.
* injections CSV: ``ordinal,time,volume`` rows.
* conditions CSV: ``key,value`` rows (temperature, cell_volume).
* concentration CSV: ``role,value`` rows (cell, syringe).

The importer distinguishes ``FormatCorruptError`` (cannot even tokenise) from
in-place semantic diagnostics (unknown unit, unparseable number, ...).
"""
from __future__ import annotations

from typing import Any

from .diagnostics import Diagnostic, FormatCorruptError, Severity, warn_diag, err_diag
from .quantities import parse_quantity

RAW_EXCERPT_LIMIT = 4000


def _read_header(text: str, location: str) -> dict[str, str]:
    units: dict[str, str] = {}
    found = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if body.lower().startswith("units:"):
                found = True
                for item in body.split(":", 1)[1].split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        units[k.strip().lower()] = v.strip()
    if not found:
        raise FormatCorruptError("missing '# units:' header", location)
    return units


def _data_rows(text: str):
    header_seen = False
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        s = raw_line.strip()
        if not s or s.startswith("#"):
            continue
        cells = [c.strip() for c in s.split(",")]
        if not header_seen:
            header_seen = True
            continue
        yield lineno, raw_line, cells


def import_power(text: str, *, location: str = "power.csv") -> dict[str, Any]:
    """Parse a power series. Raises FormatCorruptError when undecodable."""
    if not isinstance(text, str):
        raise FormatCorruptError("power payload is not text", location)
    units = _read_header(text, location)
    tu = units.get("time")
    pu = units.get("power")
    if not tu or not pu:
        raise FormatCorruptError("units header must declare time and power", location)

    samples: list[dict[str, Any]] = []
    diagnostics: list[Diagnostic] = []
    times_internal: list[float] = []
    present_flags: list[bool] = []

    n_rows = 0
    for lineno, raw_line, cells in _data_rows(text):
        if len(cells) != 2:
            raise FormatCorruptError(
                f"expected 2 columns, got {len(cells)}", f"{location}:row{lineno}")
        n_rows += 1
        tcell, pcell = cells
        tq = parse_quantity(tcell, "time", default_unit=tu,
                            basis="seconds since experiment start")
        present = pcell == "" or pcell.upper().startswith("GAP")
        if present:
            reason = "missing sample"
            if pcell.upper().startswith("GAP"):
                rest = pcell.split(":", 1)
                reason = rest[1].strip() if len(rest) == 2 else "GAP marker"
            pq = parse_quantity("", "power")
            diagnostics.append(warn_diag(
                "power.sample_missing", f"power sample missing at t={tcell}",
                location=f"{location}:row{lineno}",
                subject=f"sample@{tcell}", detail={"reason": reason}))
        else:
            pq = parse_quantity(pcell, "power", default_unit=pu)
            reason = None
        if not tq.ok:
            diagnostics.append(err_diag(
                "power.bad_time", tq.diagnostic or "bad time",
                location=f"{location}:row{lineno}", subject=f"sample@{tcell}"))
        if not present and not pq.ok:
            diagnostics.append(err_diag(
                "power.bad_power", pq.diagnostic or "bad power",
                location=f"{location}:row{lineno}", subject=f"sample@{tcell}"))
        samples.append({
            "index": n_rows - 1,
            "row": lineno,
            "raw_row": raw_line,
            "time": tq.as_dict(),
            "power": pq.as_dict(),
            "present": present and tq.ok,
            "gap_reason": reason if present else None,
        })
        times_internal.append(tq.value)  # type: ignore[arg-type]
        present_flags.append(present and tq.ok)

    if n_rows < 2:
        raise FormatCorruptError("fewer than 2 data rows", location)
    if any(t is None for t in times_internal):
        # Unparseable times are semantic holes, not corruption: keep in place.
        pass

    gaps = _build_gaps(times_internal, present_flags)
    return {
        "format": "itc.power.v1",
        "location": location,
        "time_basis": "seconds since experiment start",
        "raw_unit_time": tu,
        "raw_unit_power": pu,
        "samples": samples,
        "gaps": gaps,
        "diagnostics": [d.as_dict() for d in diagnostics],
        "raw_fragment": text[:RAW_EXCERPT_LIMIT],
    }


def _build_gaps(times, present) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    start: float | None = None
    for t, ok in zip(times, present):
        if not ok and t is not None and start is None:
            start = t
        elif ok and start is not None:
            gaps.append({"start": start, "end": prev_t, "reason": "missing samples"})
            start = None
        prev_t = t
    if start is not None:
        gaps.append({"start": start, "end": prev_t, "reason": "missing samples"})
    return gaps


def import_injections(text: str, *, location: str = "injections.csv") -> dict[str, Any]:
    units = _read_header(text, location)
    tu = units.get("time")
    vu = units.get("volume")
    if not tu or not vu:
        raise FormatCorruptError("units header must declare time and volume", location)

    injections: list[dict[str, Any]] = []
    diagnostics: list[Diagnostic] = []
    for lineno, raw_line, cells in _data_rows(text):
        if len(cells) != 3:
            raise FormatCorruptError(
                f"expected 3 columns, got {len(cells)}", f"{location}:row{lineno}")
        ocell, tcell, vcell = cells
        try:
            ordinal = int(ocell)
        except ValueError:
            raise FormatCorruptError(
                f"non-integer ordinal {ocell!r}", f"{location}:row{lineno}")
        tq = parse_quantity(tcell, "time", default_unit=tu,
                            basis="seconds since experiment start")
        vq = parse_quantity(vcell, "volume", default_unit=vu)
        if not tq.ok:
            diagnostics.append(err_diag(
                "injection.bad_time", tq.diagnostic or "bad injection time",
                location=f"{location}:row{lineno}", subject=f"injection#{ordinal}"))
        if not vq.ok:
            diagnostics.append(err_diag(
                "injection.bad_volume", vq.diagnostic or "bad injection volume",
                location=f"{location}:row{lineno}", subject=f"injection#{ordinal}"))
        injections.append({
            "ordinal": ordinal,
            "identity": f"injection#{ordinal}",
            "row": lineno,
            "raw_row": raw_line,
            "time": tq.as_dict(),
            "volume": vq.as_dict(),
            "time_s": tq.value,
            "volume_l": vq.value,
            "endpoint_s": None,
            "endpoint_raw": None,
        })
    if not injections:
        raise FormatCorruptError("no injection rows", location)
    ordinals = [i["ordinal"] for i in injections]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        raise FormatCorruptError("injection ordinals must be unique and sorted",
                                 location)
    return {
        "format": "itc.injections.v1",
        "location": location,
        "time_basis": "seconds since experiment start",
        "raw_unit_time": tu,
        "raw_unit_volume": vu,
        "injections": injections,
        "diagnostics": [d.as_dict() for d in diagnostics],
        "raw_fragment": text[:RAW_EXCERPT_LIMIT],
    }


def import_conditions(text: str, *, location: str = "conditions.csv") -> dict[str, Any]:
    rows: dict[str, str] = {}
    raw_rows: list[str] = []
    diagnostics: list[Diagnostic] = []
    for lineno, raw_line, cells in _data_rows(text):
        if len(cells) != 2:
            raise FormatCorruptError(
                f"expected key,value columns, got {len(cells)}",
                f"{location}:row{lineno}")
        raw_rows.append(raw_line)
        rows[cells[0].strip().lower()] = cells[1].strip()
    required = {"temperature", "cell_volume"}
    missing = required - rows.keys()
    if missing:
        # Missing mandatory operating conditions: semantic, keep what we have.
        for name in sorted(missing):
            diagnostics.append(err_diag(
                "conditions.missing", f"missing mandatory condition {name}",
                location=location, subject=f"condition:{name}"))
    tq = parse_quantity(rows.get("temperature", ""), "temperature")
    vq = parse_quantity(rows.get("cell_volume", ""), "volume")
    if rows.get("temperature") and not tq.ok:
        diagnostics.append(err_diag("conditions.bad_temperature",
                                    tq.diagnostic or "bad temperature",
                                    location=location, subject="condition:temperature"))
    if rows.get("cell_volume") and not vq.ok:
        diagnostics.append(err_diag("conditions.bad_cell_volume",
                                    vq.diagnostic or "bad cell volume",
                                    location=location, subject="condition:cell_volume"))
    return {
        "format": "itc.conditions.v1",
        "location": location,
        "temperature": tq.as_dict(),
        "cell_volume": vq.as_dict(),
        "raw_rows": raw_rows,
        "diagnostics": [d.as_dict() for d in diagnostics],
        "raw_fragment": text[:RAW_EXCERPT_LIMIT],
    }


def import_concentration(text: str, *, location: str = "concentration.csv",
                         ) -> dict[str, Any]:
    rows: dict[str, str] = {}
    raw_rows: list[str] = []
    diagnostics: list[Diagnostic] = []
    for lineno, raw_line, cells in _data_rows(text):
        if len(cells) != 2:
            raise FormatCorruptError(
                f"expected role,value columns, got {len(cells)}",
                f"{location}:row{lineno}")
        raw_rows.append(raw_line)
        rows[cells[0].strip().lower()] = cells[1].strip()
    cell_q = parse_quantity(rows.get("cell", ""), "concentration")
    syr_q = parse_quantity(rows.get("syringe", ""), "concentration")
    for role, q in (("cell", cell_q), ("syringe", syr_q)):
        if rows.get(role) and not q.ok:
            diagnostics.append(err_diag(
                "concentration.bad", q.diagnostic or f"bad {role} concentration",
                location=location, subject=f"concentration:{role}"))
    if "cell" not in rows or "syringe" not in rows:
        for role in ("cell", "syringe"):
            if role not in rows:
                diagnostics.append(err_diag(
                    "concentration.missing", f"missing {role} concentration",
                    location=location, subject=f"concentration:{role}"))
    return {
        "format": "itc.concentration.v1",
        "location": location,
        "cell": cell_q.as_dict(),
        "syringe": syr_q.as_dict(),
        "raw_rows": raw_rows,
        "diagnostics": [d.as_dict() for d in diagnostics],
        "raw_fragment": text[:RAW_EXCERPT_LIMIT],
    }


IMPORTERS = {
    "power": import_power,
    "injections": import_injections,
    "conditions": import_conditions,
    "concentration": import_concentration,
    "blank_power": import_power,
    "blank_injections": import_injections,
}
