"""Parsing of raw laboratory materials.

Six material kinds are versioned independently:

* ``power`` - raw power sampling ``(time, power)``
* ``injections`` - injection volumes and times
* ``concentration`` - ligand/macromolecule concentrations
* ``temperature`` - experiment temperature
* ``cell_volume`` - sample-cell volume
* ``baseline`` - instrument/baseline power (versioned separately from blank)
* ``blank`` - a blank/dilution run: its own power sampling plus injection timing

Original textual representations are retained verbatim. Normalisation happens
only when a derived computation requests it, and unconvertible values stay in
place with structured diagnostics instead of being dropped for plotting.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from .errors import FormatCorrupt, SemanticUnresolved
from .ids import content_id
from .units import normalize

KINDS = (
    "power",
    "injections",
    "concentration",
    "temperature",
    "cell_volume",
    "baseline",
    "blank",
)


def new_material(
    kind: str,
    role: str,
    raw_text: str,
    filename: str,
    payload: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    if kind not in KINDS:
        raise SemanticUnresolved(f"unknown material kind {kind!r}")
    canonical = {
        "kind": kind,
        "role": role,
        "filename": filename,
        "payload": payload,
    }
    return {
        "id": content_id("mat", canonical),
        "kind": kind,
        "role": role,
        "source": {"filename": filename, "raw": raw_text},
        "payload": payload,
        "diagnostics": diagnostics,
    }


def _read_rows(raw_text: str, filename: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else raw_text
    except UnicodeDecodeError as exc:
        raise FormatCorrupt(
            f"{filename}: file is not valid UTF-8",
            diagnostics=[{"position": filename, "reason": "ENCODING"}],
        ) from exc
    try:
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader]
    except csv.Error as exc:
        raise FormatCorrupt(
            f"{filename}: malformed CSV ({exc})",
            diagnostics=[{"position": filename, "reason": "CSV_PARSE"}],
        ) from exc
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise FormatCorrupt(
            f"{filename}: need a header row and at least one data row",
            diagnostics=[{"position": filename, "reason": "NO_DATA"}],
        )
    header = [h.strip() for h in rows[0]]
    if len(set(header)) != len(header):
        raise FormatCorrupt(
            f"{filename}: duplicate header labels {header}",
            diagnostics=[{"position": filename, "reason": "DUP_HEADER"}],
        )
    data = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise FormatCorrupt(
                f"{filename}: line {line_no} has {len(row)} fields, expected {len(header)}",
                diagnostics=[{"position": f"{filename}:{line_no}", "reason": "FIELD_COUNT"}],
            )
        data.append({h: v for h, v in zip(header, row)})
    return header, data


def _find_column(header: list[str], patterns: tuple[str, ...]) -> str | None:
    for h in header:
        token = h.strip().lower()
        for pat in patterns:
            if re.fullmatch(pat, token):
                return h
    return None


def _scan_series(
    rows: list[dict[str, str]],
    tcol: str,
    vcol: str,
    filename: str,
    *,
    time_unit: str,
    value_dim: str,
    value_unit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[float, float]]]:
    """Parse (time, value) rows, keeping raw text and recording every defect."""
    points: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []
    valid_times: list[float] = []
    for i, row in enumerate(rows):
        position = f"{filename}:row{i + 2}"
        t_raw = row[tcol].strip()
        v_raw = row[vcol].strip()
        point: dict[str, Any] = {
            "index": i,
            "time_raw": t_raw,
            "value_raw": v_raw,
            "position": position,
            "convertible": False,
        }
        t_quant = None
        v_quant = None
        try:
            t_quant = normalize(t_raw, time_unit, "time", position=position + ":time")
        except SemanticUnresolved as exc:
            point["time_error"] = exc.diagnostics
            diags.extend(exc.diagnostics)
        try:
            v_quant = normalize(v_raw, value_unit, value_dim, position=position + ":value")
        except SemanticUnresolved as exc:
            point["value_error"] = exc.diagnostics
            diags.extend(exc.diagnostics)
        if t_quant is not None and v_quant is not None:
            point["time_s"] = t_quant.value
            point["value"] = v_quant.value
            point["convertible"] = True
            valid_times.append(t_quant.value)
        points.append(point)

    gaps: list[tuple[float, float]] = []
    if len(valid_times) >= 3:
        deltas = [b - a for a, b in zip(valid_times, valid_times[1:]) if b > a]
        if deltas:
            ordered = sorted(deltas)
            median_dt = ordered[len(ordered) // 2]
            threshold = max(8.0 * median_dt, 1e-9)
            prev_t = valid_times[0]
            for t in valid_times[1:]:
                if t - prev_t > threshold:
                    gaps.append((prev_t, t))
                prev_t = t
    if not any(p["convertible"] for p in points):
        raise SemanticUnresolved(
            f"{filename}: no convertible samples",
            diagnostics=diags or [{"position": filename, "reason": "NO_CONVERTIBLE_POINTS"}],
        )
    return points, diags, gaps


def _parse_power_like(
    kind: str,
    role: str,
    raw_text: str,
    filename: str,
    *,
    value_dim: str,
    default_unit: str,
) -> dict[str, Any]:
    header, rows = _read_rows(raw_text, filename)
    tcol = _find_column(header, (r"t(ime)?(_?s)?", r"time\s*\(.*\)", r"t\s*\(.*\)"))
    vcol = _find_column(
        header,
        (r"p(ower)?(_?uw)?", r"power\s*\(.*\)", r"p\s*\(.*\)", r"dq.?dt", r"heat.?flow\s*\(.*\)"),
    ) if value_dim == "power" else _find_column(header, (r"base(line)?(_?w)?", r"power\s*\(.*\)", r"p(ower)?", r"p\s*\(.*\)"))
    if tcol is None or vcol is None:
        raise FormatCorrupt(
            f"{filename}: need time and value columns, found {header}",
            diagnostics=[{"position": filename, "reason": "MISSING_COLUMNS", "header": header}],
        )
    time_unit = re.search(r"\(([^)]*)\)", tcol)
    value_unit = re.search(r"\(([^)]*)\)", vcol)
    tu = (time_unit.group(1) if time_unit else "s")
    vu = (value_unit.group(1) if value_unit else default_unit)
    points, diags, gaps = _scan_series(
        rows, tcol, vcol, filename, time_unit=tu, value_dim=value_dim, value_unit=vu
    )
    payload = {
        "time_column": tcol,
        "value_column": vcol,
        "time_unit_raw": tu,
        "value_unit_raw": vu,
        "value_canonical_unit": "W" if value_dim == "power" else "W",
        "coordinate_basis": "elapsed seconds from first convertible sample",
        "points": points,
        "gaps_s": [{"start": a, "end": b} for a, b in gaps],
    }
    return new_material(kind, role, raw_text, filename, payload, diags)


def parse_power(role: str, raw_text: str, filename: str = "power.csv") -> dict[str, Any]:
    return _parse_power_like("power", role, raw_text, filename, value_dim="power", default_unit="µW")


def parse_baseline(role: str, raw_text: str, filename: str = "baseline.csv") -> dict[str, Any]:
    header, rows = _read_rows(raw_text, filename)
    if len(header) == 1:
        col = header[0]
        unit_match = re.search(r"\(([^)]*)\)", col)
        unit = unit_match.group(1) if unit_match else "µW"
        diags: list[dict[str, Any]] = []
        try:
            quant = normalize(rows[0][col], unit, "power", position=f"{filename}:{col}")
        except SemanticUnresolved as exc:
            diags = exc.diagnostics
            quant = None
        payload = {
            "mode": "constant",
            "value_column": col,
            "value_unit_raw": unit,
            "constant": quant.to_dict() if quant else None,
            "points": [
                {"index": 0, "value_raw": rows[0][col].strip(), "convertible": quant is not None}
            ],
        }
        return new_material("baseline", role, raw_text, filename, payload, diags)
    return _parse_power_like("baseline", role, raw_text, filename, value_dim="power", default_unit="µW")


def parse_injections(role: str, raw_text: str, filename: str = "injections.csv") -> dict[str, Any]:
    header, rows = _read_rows(raw_text, filename)
    icol = _find_column(header, (r"inj(?:ection)?(?:[_\s]?#?|no\.?|index|number)?", r"#", r"n"))
    tcol = _find_column(header, (r"t(ime)?(_?s)?", r"start(?:[_ ]?time)?(?:\s*\(.*\))?", r"time\s*\(.*\)"))
    vcol = _find_column(header, (r"v(ol)?(?:ume)?(?:\s*\(.*\))?", r"inj(?:ection)?[_ ]?vol(?:ume)?(?:\s*\(.*\))?"))
    if tcol is None or vcol is None:
        raise FormatCorrupt(
            f"{filename}: need injection time and volume columns, found {header}",
            diagnostics=[{"position": filename, "reason": "MISSING_COLUMNS", "header": header}],
        )
    tu_m = re.search(r"\(([^)]*)\)", tcol)
    vu_m = re.search(r"\(([^)]*)\)", vcol)
    tu = tu_m.group(1) if tu_m else "s"
    vu = vu_m.group(1) if vu_m else "µL"
    entries: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        position = f"{filename}:row{i + 2}"
        entry: dict[str, Any] = {
            "ordinal": i + 1,
            "label_raw": (row[icol].strip() if icol else str(i + 1)),
            "time_raw": row[tcol].strip(),
            "volume_raw": row[vcol].strip(),
            "convertible": False,
        }
        try:
            tq = normalize(entry["time_raw"], tu, "time", position=position + ":time")
            vq = normalize(entry["volume_raw"], vu, "volume", position=position + ":volume")
        except SemanticUnresolved as exc:
            diags.extend(exc.diagnostics)
            entry["error"] = exc.diagnostics
            entries.append(entry)
            continue
        entry["time_s"] = tq.value
        entry["volume_L"] = vq.value
        entry["convertible"] = True
        entries.append(entry)
    payload = {
        "index_column": icol,
        "time_column": tcol,
        "volume_column": vcol,
        "time_unit_raw": tu,
        "volume_unit_raw": vu,
        "coordinate_basis": "elapsed seconds from first power sample",
        "entries": entries,
    }
    return new_material("injections", role, raw_text, filename, payload, diags)


def parse_concentration(role: str, raw_text: str, filename: str = "concentration.csv") -> dict[str, Any]:
    header, rows = _read_rows(raw_text, filename)
    icol = _find_column(header, (r"inj(?:ection)?(?:[_\s]?#?|no\.?|index|number)?", r"#", r"n"))
    scol = _find_column(
        header,
        (r".*syringe.*", r".*ligand.*", r"c_?l.*", r"ligand.*", r".*injectant.*"),
    )
    mcol = _find_column(header, (r".*cell.*", r".*macromolecule.*", r"c_?m.*", r"protein.*"))
    if scol is None and mcol is None:
        raise FormatCorrupt(
            f"{filename}: need a ligand/syringe or macromolecule/cell column, found {header}",
            diagnostics=[{"position": filename, "reason": "MISSING_COLUMNS", "header": header}],
        )
    entries: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        position = f"{filename}:row{i + 2}"
        entry: dict[str, Any] = {
            "ordinal": i + 1,
            "label_raw": (row[icol].strip() if icol else str(i + 1)),
            "convertible": False,
        }
        ok = True
        if scol is not None:
            su = (re.search(r"\(([^)]*)\)", scol).group(1) if re.search(r"\(([^)]*)\)", scol) else "mM")
            entry["syringe_raw"] = row[scol].strip()
            entry["syringe_unit_raw"] = su
            try:
                entry["syringe_M"] = normalize(
                    entry["syringe_raw"], su, "concentration", position=position + ":syringe"
                ).value
            except SemanticUnresolved as exc:
                entry["syringe_error"] = exc.diagnostics
                diags.extend(exc.diagnostics)
                ok = False
        if mcol is not None:
            mu = (re.search(r"\(([^)]*)\)", mcol).group(1) if re.search(r"\(([^)]*)\)", mcol) else "µM")
            entry["macromolecule_raw"] = row[mcol].strip()
            entry["macromolecule_unit_raw"] = mu
            try:
                entry["macromolecule_M"] = normalize(
                    entry["macromolecule_raw"], mu, "concentration", position=position + ":macromolecule"
                ).value
            except SemanticUnresolved as exc:
                entry["macromolecule_error"] = exc.diagnostics
                diags.extend(exc.diagnostics)
                ok = False
        entry["convertible"] = ok
        entries.append(entry)
    payload = {
        "index_column": icol,
        "syringe_column": scol,
        "macromolecule_column": mcol,
        "coordinate_basis": "bulk molarity; syringe concentration is injectant concentration",
        "entries": entries,
    }
    return new_material("concentration", role, raw_text, filename, payload, diags)


def parse_scalar(kind: str, role: str, raw_text: str, filename: str, dimension: str, default_unit: str) -> dict[str, Any]:
    header, rows = _read_rows(raw_text, filename)
    if len(rows) != 1 or len(header) != 1:
        raise FormatCorrupt(
            f"{filename}: {kind} must be a single named column with one value",
            diagnostics=[{"position": filename, "reason": "SCALAR_SHAPE"}],
        )
    col = header[0]
    m = re.search(r"\(([^)]*)\)", col)
    unit = m.group(1) if m else default_unit
    raw_value = rows[0][col].strip()
    diags: list[dict[str, Any]] = []
    quant = None
    try:
        quant = normalize(raw_value, unit, dimension, position=f"{filename}:{col}")
    except SemanticUnresolved as exc:
        diags = exc.diagnostics
    payload = {
        "column": col,
        "unit_raw": unit,
        "raw": raw_value,
        "value": quant.to_dict() if quant else None,
        "canonical_unit": {"temperature": "K", "volume": "L"}[dimension],
    }
    return new_material(kind, role, raw_text, filename, payload, diags)


def parse_temperature(role: str, raw_text: str, filename: str = "temperature.csv") -> dict[str, Any]:
    return parse_scalar("temperature", role, raw_text, filename, "temperature", "°C")


def parse_cell_volume(role: str, raw_text: str, filename: str = "cell_volume.csv") -> dict[str, Any]:
    return parse_scalar("cell_volume", role, raw_text, filename, "volume", "µL")


def build_blank(
    role: str,
    power_material: dict[str, Any],
    injections_material: dict[str, Any],
) -> dict[str, Any]:
    if power_material["kind"] != "power":
        raise SemanticUnresolved("blank run requires a power sampling material")
    if injections_material["kind"] != "injections":
        raise SemanticUnresolved("blank run requires an injection timing material")
    payload = {
        "power_material_id": power_material["id"],
        "injections_material_id": injections_material["id"],
        "gaps_s": power_material["payload"].get("gaps_s", []),
    }
    raw = (
        f"BLANK COMPOSITE\npower={power_material['id']}\n"
        f"injections={injections_material['id']}\n"
    )
    return new_material("blank", role, raw, "blank.composite", payload, [])


_PARSERS = {
    "power": parse_power,
    "baseline": parse_baseline,
    "injections": parse_injections,
    "concentration": parse_concentration,
    "temperature": parse_temperature,
    "cell_volume": parse_cell_volume,
}


def parse_material(kind: str, role: str, raw_text: str, filename: str) -> dict[str, Any]:
    parser = _PARSERS.get(kind)
    if parser is None:
        raise SemanticUnresolved(f"unknown material kind {kind!r}")
    return parser(role, raw_text, filename)


def patch_power_region(
    material: dict[str, Any],
    start_s: float,
    end_s: float,
    multiplier: float,
) -> dict[str, Any]:
    """Create a new local power version by scaling samples in one time region.

    Used for source-data changes that must affect only dependent injections;
    the original material stays immutable and versioned.
    """
    if material["kind"] != "power":
        raise SemanticUnresolved("only power sampling can be region-patched")
    points = [dict(p) for p in material["payload"]["points"]]
    touched: list[int] = []
    for p in points:
        if p.get("convertible") and start_s <= p["time_s"] <= end_s:
            p["value"] = p["value"] * multiplier
            p["value_raw"] = f"{p['value']:.6g}"
            p["patched_from_raw"] = True
            touched.append(p["index"])
    if not touched:
        raise SemanticUnresolved(
            "patch region contains no convertible samples",
            diagnostics=[{"reason": "EMPTY_PATCH_REGION", "start_s": start_s, "end_s": end_s}],
        )
    payload = dict(material["payload"])
    payload["points"] = points
    payload["patches"] = list(payload.get("patches", [])) + [
        {"start_s": start_s, "end_s": end_s, "multiplier": multiplier, "sample_indexes": touched}
    ]
    return new_material(
        "power",
        material["role"],
        material["source"]["raw"],
        material["source"]["filename"],
        payload,
        list(material.get("diagnostics", [])),
    )


def cut_power_gap(
    material: dict[str, Any],
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """Introduce a data gap (unconvertible markers) without dropping rows."""
    points = [dict(p) for p in material["payload"]["points"]]
    removed: list[int] = []
    for p in points:
        if p.get("convertible") and start_s <= p["time_s"] <= end_s:
            p["value_raw"] = p["value_raw"]
            p["convertible"] = False
            p.setdefault("value_error", []).append(
                {"position": p.get("position", "?"), "reason": "GAP_INTRODUCED"}
            )
            removed.append(p["index"])
    valid_times = [p["time_s"] for p in points if p.get("convertible")]
    gaps = list(material["payload"].get("gaps_s", []))
    gaps.append({"start": start_s, "end": end_s, "introduced": True})
    payload = dict(material["payload"])
    payload["points"] = points
    payload["gaps_s"] = gaps
    return new_material(
        "power",
        material["role"],
        material["source"]["raw"],
        material["source"]["filename"],
        payload,
        list(material.get("diagnostics", [])),
    )


def points_list(material: dict[str, Any]) -> list[dict[str, Any]]:
    return material["payload"]["points"]
