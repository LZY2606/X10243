"""Material parsers: raw text -> normalized payload + positioned diagnostics.

Each material kind is versioned separately in the store. Raw text is always
kept; these parsers only build the derived, normalized view. Rows that cannot
be converted remain in place (valid=False, raw kept) with a diagnostic.
"""
from __future__ import annotations

import json
import re

from .results import Result, OK, DIAGNOSTICS, FORMAT_CORRUPT
from .units import (
    Diagnostic, convert_quantity,
    POWER_UNITS, TIME_UNITS, VOLUME_UNITS, CONC_UNITS, TEMP_UNITS,
)

MATERIAL_KINDS = [
    "power_trace", "injections", "concentrations", "temperature",
    "cell_volume", "baseline", "blank_trace", "blank_injections",
]


def _status(diags):
    return DIAGNOSTICS if diags else OK


def _unit_factor(raw_unit, table, what, diags):
    if raw_unit is None:
        diags.append(Diagnostic("missing_unit", f"missing unit header for {what}", "header", {}))
        return None
    if raw_unit not in table:
        diags.append(Diagnostic("unknown_unit", f"unknown {what} unit {raw_unit!r}", "header", {"raw": raw_unit}))
        return None
    return table[raw_unit]


def parse_power_trace(raw_text):
    """CSV with `# unit_time:` / `# unit_power:` headers.

    Empty power field = intentional data gap. Unparseable power = invalid
    sample kept in place with a diagnostic. Gaps never get filled.
    """
    diags = []
    unit_time_raw = None
    unit_power_raw = None
    samples = []
    for lineno, line in enumerate(raw_text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s[1:].strip()
            if ":" in body:
                key, val = [x.strip() for x in body.split(":", 1)]
                if key == "unit_time":
                    unit_time_raw = val
                elif key == "unit_power":
                    unit_power_raw = val
            continue
        parts = [p.strip() for p in s.split(",")]
        t_raw = parts[0]
        p_raw = parts[1] if len(parts) > 1 else ""
        try:
            t_val = float(t_raw)
        except ValueError:
            diags.append(Diagnostic("unconvertible_time", f"unparseable time {t_raw!r}", f"line {lineno}", {"raw": line}))
            samples.append({"t": None, "p": None, "valid": False, "gap": False, "raw": line, "line": lineno})
            continue
        if p_raw == "":
            samples.append({"t": t_val, "p": None, "valid": False, "gap": True, "raw": line, "line": lineno})
            continue
        try:
            p_val = float(p_raw)
        except ValueError:
            diags.append(Diagnostic("unconvertible_value", f"unparseable power {p_raw!r}", f"line {lineno}", {"raw": line}))
            samples.append({"t": t_val, "p": None, "valid": False, "gap": False, "raw": line, "line": lineno})
            continue
        samples.append({"t": t_val, "p": p_val, "valid": True, "gap": False, "raw": line, "line": lineno})

    units_unknown = False
    tu = _unit_factor(unit_time_raw, TIME_UNITS, "time", diags)
    pu = _unit_factor(unit_power_raw, POWER_UNITS, "power", diags)
    if tu is None or pu is None:
        units_unknown = True
    else:
        for smp in samples:
            if smp["t"] is not None:
                smp["t"] = smp["t"] * tu[1] + tu[2]
            if smp["p"] is not None:
                smp["p"] = smp["p"] * pu[1] + pu[2]

    if not any(smp["valid"] for smp in samples):
        diags.append(Diagnostic("no_usable_samples", "no usable samples in trace", "file", {}))
        return Result(FORMAT_CORRUPT, None, diags)

    gaps = []
    ts = [smp["t"] for smp in samples]
    i = 0
    while i < len(samples):
        if not samples[i]["valid"]:
            j = i
            reasons = set()
            while j < len(samples) and not samples[j]["valid"]:
                reasons.add("gap" if samples[j]["gap"] else "invalid")
                j += 1
            gaps.append({
                "start_index": i,
                "end_index": j - 1,
                "start_t": ts[i - 1] if i > 0 else ts[i],
                "end_t": ts[j] if j < len(ts) else ts[j - 1],
                "reasons": sorted(reasons),
                "position": f"lines {samples[i]['line']}-{samples[j - 1]['line']}",
            })
            i = j
        else:
            i += 1

    normalized = {
        "time_unit": None if units_unknown else "s",
        "power_unit": None if units_unknown else "uJ/s",
        "units_unknown": units_unknown,
        "samples": samples,
        "gaps": gaps,
        "coordinate_basis": "elapsed_time_from_trace_start",
    }
    return Result(_status(diags), normalized, diags)


def parse_injections(raw_text):
    """CSV `time,volume` with unit headers. Valid rows get stable ids inj-%02d
    assigned after sorting by (time, line), so equivalent input reordered in
    the file still yields the same ids and order."""
    diags = []
    unit_time_raw = None
    unit_volume_raw = None
    rows = []
    for lineno, line in enumerate(raw_text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s[1:].strip()
            if ":" in body:
                key, val = [x.strip() for x in body.split(":", 1)]
                if key == "unit_time":
                    unit_time_raw = val
                elif key == "unit_volume":
                    unit_volume_raw = val
            continue
        parts = [p.strip() for p in s.split(",")]
        if parts[0].lower() in ("time", "t"):
            continue
        t_raw = parts[0]
        v_raw = parts[1] if len(parts) > 1 else ""
        try:
            t_val = float(t_raw)
        except ValueError:
            diags.append(Diagnostic("unconvertible_time", f"unparseable time {t_raw!r}", f"line {lineno}", {"raw": line}))
            rows.append({"time": None, "volume_uL": None, "valid": False, "raw": line, "line": lineno})
            continue
        try:
            v_val = float(v_raw)
        except ValueError:
            diags.append(Diagnostic("unconvertible_value", f"unparseable volume {v_raw!r}", f"line {lineno}", {"raw": line}))
            rows.append({"time": t_val, "volume_uL": None, "valid": False, "raw": line, "line": lineno})
            continue
        rows.append({"time": t_val, "volume_uL": v_val, "valid": True, "raw": line, "line": lineno})

    units_unknown = False
    tu = _unit_factor(unit_time_raw, TIME_UNITS, "time", diags)
    vu = _unit_factor(unit_volume_raw, VOLUME_UNITS, "volume", diags)
    if tu is None or vu is None:
        units_unknown = True
    else:
        for row in rows:
            if row["time"] is not None:
                row["time"] = row["time"] * tu[1] + tu[2]
            if row["volume_uL"] is not None:
                row["volume_uL"] = row["volume_uL"] * vu[1] + vu[2]

    valid_sorted = sorted((r for r in rows if r["valid"]), key=lambda r: (r["time"], r["line"]))
    for rank, row in enumerate(valid_sorted):
        row["id"] = f"inj-{rank:02d}"
    invalid = [r for r in rows if not r["valid"]]
    for row in invalid:
        row["id"] = None
    normalized = {
        "time_unit": None if units_unknown else "s",
        "volume_unit": None if units_unknown else "uL",
        "units_unknown": units_unknown,
        "injections": valid_sorted + invalid,
    }
    if not valid_sorted:
        diags.append(Diagnostic("no_usable_injections", "no usable injection rows", "file", {}))
        return Result(FORMAT_CORRUPT, None, diags)
    return Result(_status(diags), normalized, diags)


def parse_concentrations(raw_text):
    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return Result(FORMAT_CORRUPT, None, [Diagnostic("json_corrupt", f"invalid JSON: {exc}", "file", {"pos": exc.pos})])
    diags = []
    out = {}
    raw_units = {}
    for role in ("syringe", "cell"):
        node = doc.get(role) if isinstance(doc, dict) else None
        if not isinstance(node, dict) or "value" not in node:
            diags.append(Diagnostic("missing_field", f"missing concentration for role {role!r}", f"concentrations.{role}", {}))
            out[role] = None
            continue
        val, _canon, diag = convert_quantity(node.get("value"), node.get("unit", ""), CONC_UNITS, f"concentrations.{role}")
        if diag is not None:
            diags.append(diag)
            out[role] = None
        else:
            out[role] = val
            raw_units[role] = node.get("unit")
    normalized = {"syringe_uM": out["syringe"], "cell_uM": out["cell"], "raw_units": raw_units}
    return Result(_status(diags), normalized, diags)


_SCALAR_RE = re.compile(r"^\s*([-+0-9.eE]+)\s*(\S+)\s*$")


def parse_temperature(raw_text):
    m = _SCALAR_RE.match(raw_text)
    if not m:
        return Result(FORMAT_CORRUPT, None, [Diagnostic("unparseable", "cannot parse temperature scalar", "line 1", {"raw": raw_text})])
    val, _canon, diag = convert_quantity(m.group(1), m.group(2), TEMP_UNITS, "temperature")
    if diag is not None:
        return Result(DIAGNOSTICS, {"degC": None, "raw": raw_text}, [diag])
    return Result(OK, {"degC": val, "K": val + 273.15, "unit": "degC", "raw": raw_text}, [])


def parse_cell_volume(raw_text):
    m = _SCALAR_RE.match(raw_text)
    if not m:
        return Result(FORMAT_CORRUPT, None, [Diagnostic("unparseable", "cannot parse cell volume scalar", "line 1", {"raw": raw_text})])
    val, _canon, diag = convert_quantity(m.group(1), m.group(2), VOLUME_UNITS, "cell_volume")
    if diag is not None:
        return Result(DIAGNOSTICS, {"uL": None, "raw": raw_text}, [diag])
    return Result(OK, {"uL": val, "unit": "uL", "raw": raw_text}, [])


def parse_baseline(raw_text):
    """Baseline scan CSV -> linear model p(t) = a + b*t (least squares)."""
    res = parse_power_trace(raw_text)
    if res.status == FORMAT_CORRUPT:
        return res
    norm = res.value
    pts = [(s["t"], s["p"]) for s in norm["samples"] if s["valid"]]
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else 0.0
    intercept = (sy - slope * sx) / n
    normalized = {
        "a_uJ_per_s": intercept,
        "b_uJ_per_s2": slope,
        "n_samples": n,
        "time_unit": norm["time_unit"],
        "power_unit": norm["power_unit"],
        "units_unknown": norm["units_unknown"],
    }
    return Result(res.status, normalized, res.diagnostics)


KIND_PARSERS = {
    "power_trace": parse_power_trace,
    "blank_trace": parse_power_trace,
    "injections": parse_injections,
    "blank_injections": parse_injections,
    "concentrations": parse_concentrations,
    "temperature": parse_temperature,
    "cell_volume": parse_cell_volume,
    "baseline": parse_baseline,
}
