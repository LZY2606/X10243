"""Raw artifact import.

The import boundary keeps original representations (raw cell strings, headers,
file names) and only attaches normalized values in the derived layer. Values
that cannot be converted stay in place with located diagnostics; they are
never replaced by zero or a default.
"""
from __future__ import annotations

import csv
import io
import json

from .errors import CorruptFormat, Diagnostic, SemanticImpossibility
from .models import Material, stable_hash
from .units import parse_quantity


def _read_csv(blob: str, source: str, min_cols: int = 2):
    try:
        reader = csv.reader(io.StringIO(blob))
        rows = [r for r in reader]
    except (csv.Error, UnicodeDecodeError) as exc:
        raise CorruptFormat(f"{source}: unreadable CSV ({exc})") from exc
    if len(rows) < 2:
        raise CorruptFormat(f"{source}: needs a header row and >=1 data row")
    header = [h.strip() for h in rows[0]]
    if len(header) < min_cols:
        raise CorruptFormat(
            f"{source}: expected >= {min_cols} columns, got {header}")
    return header, rows[1:]


def parse_trace(blob: str, kind: str, source: str,
                gap_factor: float = 3.0) -> dict:
    """Parse a power/blank/baseline CSV.

    Required header: time, power (baseline may be 1 or 2 columns). Each cell is
    a '<number> <unit>' quantity so original text and units survive. A data gap
    is a sampling interval wider than ``gap_factor`` * median dt.
    """
    header, data_rows = _read_csv(
        blob, source, 1 if kind == "baseline" else 2)
    diags: list[Diagnostic] = []
    raw_rows: list[list[str]] = []
    times: list[float] = []
    values: list[float | None] = []

    for i, row in enumerate(data_rows):
        if not row or all(not c.strip() for c in row):
            diags.append(Diagnostic(
                "format_blank_row", "blank row skipped", "warning",
                f"{source}:row{i + 2}", None))
            continue
        if len(row) < (1 if kind == "baseline" else 2):
            raise CorruptFormat(
                f"{source}: row {i + 2} has {len(row)} columns")
        raw_rows.append([c.strip() for c in row])
        tq = parse_quantity(row[0], "time", f"{source}:row{i + 2}:time")
        diags.extend(tq.diagnostics or [])
        times.append(tq.value)
        if kind == "baseline":
            values.append(None)
        else:
            pq = parse_quantity(row[1], "power", f"{source}:row{i + 2}:power")
            diags.extend(pq.diagnostics or [])
            values.append(pq.value)

    good_t = [t for t in times if t is not None]
    if len(good_t) < 2:
        raise SemanticImpossibility(
            f"{source}: fewer than two usable time samples")

    ordered = sorted(good_t)
    deltas = [b - a for a, b in zip(ordered, ordered[1:])]
    med_dt = sorted(deltas)[len(deltas) // 2]
    if med_dt <= 0:
        raise SemanticImpossibility(f"{source}: non-positive median dt")

    gaps: list[dict] = []
    gap_after = set()
    for idx in range(len(times) - 1):
        if times[idx] is None or times[idx + 1] is None:
            continue
        dt = times[idx + 1] - times[idx]
        if dt > gap_factor * med_dt:
            gap_after.add(idx)
            gaps.append({"after_index": idx,
                         "from_s": times[idx], "to_s": times[idx + 1],
                         "missing_s": dt - med_dt})
    if any(b <= a for a, b in zip(good_t, good_t[1:])):
        diags.append(Diagnostic(
            "semantic_non_monotonic_time",
            "time column is not strictly increasing", "warning",
            f"{source}:time"))

    baseline_samples = None
    if kind == "baseline":
        bvals: list[float | None] = []
        for i, row in enumerate(raw_rows):
            if len(row) >= 2:
                bq = parse_quantity(row[1], "power",
                                    f"{source}:row{i + 2}:baseline")
                diags.extend(bq.diagnostics or [])
                bvals.append(bq.value)
            else:
                bvals.append(None)
        baseline_samples = bvals

    return {
        "header": header,
        "raw_rows": raw_rows,
        "time_s": times,
        "power_cal_s": values,
        "baseline_cal_s": baseline_samples,
        "median_dt_s": med_dt,
        "gap_after": sorted(gap_after),
        "gaps": gaps,
        "time_reference": "s after acquisition start",
        "power_unit": "cal/s",
        "diagnostics": diags,
    }


def parse_schedule(blob: str, source: str, kind: str = "schedule") -> dict:
    """Parse injections: ordinal, injection time, volume, optional label.

    Header must contain ``time`` and ``volume``. The first column is treated as
    ordinal when not named time.
    """
    header, rows = _read_csv(blob, source, 2)
    low = [h.lower() for h in header]
    if "time" not in low or "volume" not in low:
        raise CorruptFormat(
            f"{source}: schedule header needs 'time' and 'volume', got {header}")
    ti, vi = low.index("time"), low.index("volume")
    li = low.index("label") if "label" in low else -1
    diags: list[Diagnostic] = []
    injections = []
    for i, row in enumerate(rows):
        if not row or all(not c.strip() for c in row):
            continue
        loc = f"{source}:row{i + 2}"
        tq = parse_quantity(row[ti], "time", f"{loc}:time")
        vq = parse_quantity(row[vi], "volume", f"{loc}:volume")
        diags.extend(tq.diagnostics or [])
        diags.extend(vq.diagnostics or [])
        label = row[li] if li >= 0 and li < len(row) else f"#{i}"
        injections.append({
            "ordinal": i,
            "label": label.strip() or f"#{i}",
            "time_s": tq.value,
            "time_raw": tq.raw,
            "volume_l": vq.value,
            "volume_raw": vq.raw,
        })
    if not injections:
        raise SemanticImpossibility(f"{source}: no injections parsed")
    return {
        "header": header,
        "injections": injections,
        "time_reference": "s after acquisition start",
        "volume_unit": "L",
        "diagnostics": diags,
    }


def parse_concentration(blob: str, source: str) -> dict:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise CorruptFormat(f"{source}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise CorruptFormat(f"{source}: concentration payload must be object")
    diags: list[Diagnostic] = []

    def conc(field_name: str):
        raw = data.get(field_name)
        if raw is None:
            diags.append(Diagnostic(
                "semantic_missing_concentration",
                f"missing {field_name}", "error", f"{source}#{field_name}"))
            return None
        q = parse_quantity(str(raw), "concentration", f"{source}#{field_name}")
        diags.extend(q.diagnostics or [])
        return {"raw": q.raw, "value_m": q.value, "unit": q.canonical_unit}

    return {
        "syringe": conc("syringe_concentration"),
        "cell": conc("cell_concentration"),
        "syringe_label": data.get("syringe_label", "ligand"),
        "cell_label": data.get("cell_label", "macromolecule"),
        "diagnostics": diags,
    }


def parse_runmeta(blob: str, source: str) -> dict:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise CorruptFormat(f"{source}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise CorruptFormat(f"{source}: run metadata must be an object")
    diags: list[Diagnostic] = []

    tq = parse_quantity(str(data.get("temperature", "")), "temperature",
                        f"{source}#temperature")
    diags.extend(tq.diagnostics or [])
    cq = parse_quantity(str(data.get("cell_volume", "")), "volume",
                        f"{source}#cell_volume")
    diags.extend(cq.diagnostics or [])
    return {
        "temperature": {"raw": tq.raw, "value_k": tq.value,
                        "unit": tq.canonical_unit},
        "cell_volume": {"raw": cq.raw, "value_l": cq.value,
                        "unit": cq.canonical_unit},
        "experiment_id": data.get("experiment_id", ""),
        "diagnostics": diags,
    }


PARSERS = {
    "power": lambda b, s: parse_trace(b, "power", s),
    "blank": lambda b, s: parse_trace(b, "blank", s),
    "baseline": lambda b, s: parse_trace(b, "baseline", s),
    "schedule": lambda b, s: parse_schedule(b, s),
    "blank_schedule": lambda b, s: parse_schedule(b, s, "blank_schedule"),
    "concentration": parse_concentration,
    "runmeta": parse_runmeta,
}


def import_material(kind: str, blob: str, source: str,
                    version: int) -> Material:
    """Turn a raw artifact into an immutable, versioned Material."""
    if kind not in PARSERS:
        raise SemanticImpossibility(f"unknown material kind: {kind!r}")
    payload = PARSERS[kind](blob, source)
    diags = payload.pop("diagnostics")
    material_id = f"mat.{kind}.v{version}.{stable_hash(payload)[:12]}"
    return Material(id=material_id, kind=kind, version=version,
                    source=source, payload=payload, diagnostics=diags)
