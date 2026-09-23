"""Parsers for raw ITC materials.

Design rules enforced here:
* original textual representation is preserved by the caller (raw blob layer);
* every numeric field keeps both its original token and a normalised SI value;
* a value that cannot be converted stays at its original position with a
  diagnostic instead of being dropped or replaced by a default;
* structural damage raises FORMAT_CORRUPTED with a path/line; a parse budget
  running out raises BUDGET_EXHAUSTED.
"""

from __future__ import annotations

from typing import Optional

from .budget import Budget
from .errors import FormatCorrupted
from .units import convert_field


def _lines(text: str):
    for lineno, raw in enumerate(text.splitlines(), start=1):
        yield lineno, raw


def _tokenize(line: str) -> list[str]:
    return [tok.strip() for tok in line.split(",")]


def _header_block(text: str, path: str):
    """Return (metadata dict, first_data_lineno, header_cells_or_None)."""
    metadata: dict[str, str] = {}
    header: Optional[list[str]] = None
    header_lineno: Optional[int] = None
    for lineno, raw in _lines(text):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if ":" not in body:
                raise FormatCorrupted(
                    f"malformed metadata line {lineno}: expected '# key: value'",
                    path=path, line=lino_guard(lineno),
                )
            key, value = body.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
            continue
        header = _tokenize(stripped)
        header_lineno = lineno
        break
    if header is None:
        raise FormatCorrupted("no header row found", path=path)
    return metadata, header_lineno, header


def lino_guard(x):  # tiny helper kept explicit for positional diagnostics
    return x


def _data_rows(text: str, after_lineno: int, path: str, width: int, budget: Budget):
    for lineno, raw in _lines(text):
        if lineno <= after_lineno:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        budget.tick()
        cells = _tokenize(stripped)
        if len(cells) != width:
            raise FormatCorrupted(
                f"ragged row: expected {width} columns, got {len(cells)}",
                path=path, line=lineno,
            )
        yield lineno, cells


def _require(metadata: dict, key: str, path: str, line=1):
    if key not in metadata:
        raise FormatCorrupted(f"missing required metadata '{key}'", path=path, line=line)
    return metadata[key]


def parse_power(text: str, *, path: str = "power.csv", budget: Optional[Budget] = None,
                declared_time_unit: Optional[str] = None,
                declared_power_unit: Optional[str] = None):
    budget = budget or Budget("power_rows", 10_000_000)
    metadata, header_lineno, header = _header_block(text, path)
    required = ["time", "power"]
    for name in required:
        if not any(name == h.lower() for h in header):
            raise FormatCorrupted(
                f"power header must contain columns {required}, found {header}",
                path=path, line=header_lineno,
            )
    idx = {h.lower(): i for i, h in enumerate(header)}
    time_unit = declared_time_unit or _require(metadata, "time_unit", path)
    power_unit = declared_power_unit or _require(metadata, "power_unit", path)

    samples = []
    diagnostics = []
    seq = 0
    for lineno, cells in _data_rows(text, header_lineno, path, len(header), budget):
        seq += 1
        raw_t, raw_p = cells[idx["time"]], cells[idx["power"]]
        t_s, d1 = convert_field(raw_t, time_unit, "time", path=path, line=lineno)
        p_w, d2 = convert_field(raw_p, power_unit, "power", path=path, line=lineno)
        if d1:
            diagnostics.append(d1)
        if d2:
            diagnostics.append(d2)
        if t_s is not None:
            samples.append({
                "index": seq, "line": lineno,
                "raw_time": raw_t, "raw_power": raw_p,
                "time_s": t_s, "power_w": p_w,
                "valid": p_w is not None,
            })
        else:
            diagnostics.append({
                "code": "UNPLACED_SAMPLE", "path": path, "line": lineno,
                "index": seq, "raw_time": raw_t, "raw_power": raw_p,
                "message": "sample time could not be converted; point retained but unplaced",
            })

    samples.sort(key=lambda s: (s["time_s"], s["index"]))
    if not samples:
        raise FormatCorrupted("power material contains no placed samples", path=path)
    return {
        "kind": "power",
        "metadata": metadata,
        "units": {"time": time_unit, "power": power_unit},
        "samples": samples,
        "diagnostics": diagnostics,
    }


def parse_schedule(text: str, *, path: str = "injections.csv",
                   budget: Optional[Budget] = None,
                   time_unit: Optional[str] = None, volume_unit: Optional[str] = None):
    budget = budget or Budget("injection_rows", 1_000_000)
    metadata, header_lineno, header = _header_block(text, path)
    needed = ["ordinal", "time", "volume"]
    idx = {}
    for name in needed:
        if name not in [h.lower() for h in header]:
            raise FormatCorrupted(
                f"injection header must contain {needed}, found {header}",
                path=path, line=header_lineno,
            )
    idx = {h.lower(): i for i, h in enumerate(header)}
    tu = time_unit or _require(metadata, "time_unit", path)
    vu = volume_unit or _require(metadata, "volume_unit", path)

    injections = []
    diagnostics = []
    seen = set()
    for lineno, cells in _data_rows(text, header_lineno, path, len(header), budget):
        raw_ord, raw_t, raw_v = cells[idx["ordinal"]], cells[idx["time"]], cells[idx["volume"]]
        try:
            ordinal = int(raw_ord)
        except ValueError:
            try:
                as_float = float(raw_ord)
            except ValueError:
                as_float = None
            if as_float is None or as_float != int(as_float):
                raise FormatCorrupted(
                    f"injection ordinal must be an integer, found {raw_ord!r}",
                    path=path, line=lineno,
                )
            ordinal = int(as_float)
        if ordinal in seen:
            raise FormatCorrupted(f"duplicate injection ordinal {ordinal}", path=path, line=lineno)
        seen.add(ordinal)
        t_s, d1 = convert_field(raw_t, tu, "time", path=path, line=lineno)
        v_l, d2 = convert_field(raw_v, vu, "volume", path=path, line=lineno)
        diagnostics.extend(d for d in (d1, d2) if d)
        injections.append({
            "ordinal": ordinal, "line": lineno,
            "raw_time": raw_t, "raw_volume": raw_v,
            "time_s": t_s, "volume_l": v_l,
            "valid": t_s is not None and v_l is not None,
        })
    if not injections:
        raise FormatCorrupted("schedule contains no injections", path=path)
    injections.sort(key=lambda j: (j["ordinal"]))
    return {
        "kind": "schedule",
        "metadata": metadata,
        "units": {"time": tu, "volume": vu},
        "injections": injections,
        "diagnostics": diagnostics,
    }


def parse_concentrations(text: str, *, path: str = "concentrations.csv",
                         budget: Optional[Budget] = None):
    budget = budget or Budget("concentration_rows", 100_000)
    metadata, header_lineno, header = _header_block(text, path)
    cols = [h.lower() for h in header]
    for name in ("role", "value", "unit"):
        if name not in cols:
            raise FormatCorrupted(
                f"concentration header must contain role,value,unit, found {header}",
                path=path, line=header_lineno,
            )
    idx = {h.lower(): i for i, h in enumerate(header)}
    entries = {}
    diagnostics = []
    for lineno, cells in _data_rows(text, header_lineno, path, len(header), budget):
        role = cells[idx["role"]].strip().lower()
        raw_value, raw_unit = cells[idx["value"]], cells[idx["unit"]]
        si, diag = convert_field(raw_value, raw_unit, "concentration", path=path, line=lineno)
        if diag:
            diagnostics.append(diag)
        entries[role] = {
            "raw": f"{raw_value} {raw_unit}", "raw_value": raw_value, "raw_unit": raw_unit,
            "value_si": si, "line": lineno,
        }
    for role in ("ligand", "macromolecule"):
        if role not in entries:
            raise FormatCorrupted(
                f"concentration table must define '{role}'", path=path, line=header_lineno,
            )
    return {"kind": "concentrations", "metadata": metadata, "entries": entries,
            "diagnostics": diagnostics}


def parse_conditions(text: str, *, path: str = "conditions.csv",
                     budget: Optional[Budget] = None):
    budget = budget or Budget("condition_rows", 100_000)
    metadata, header_lineno, header = _header_block(text, path)
    cols = [h.lower() for h in header]
    for name in ("key", "value", "unit"):
        if name not in cols:
            raise FormatCorrupted(
                f"conditions header must contain key,value,unit, found {header}",
                path=path, line=header_lineno,
            )
    idx = {h.lower(): i for i, h in enumerate(header)}
    entries = {}
    diagnostics = []
    quantities = {"temperature": "temperature", "cell_volume": "volume"}
    for lineno, cells in _data_rows(text, header_lineno, path, len(header), budget):
        key = cells[idx["key"]].strip()
        raw_value, raw_unit = cells[idx["value"]], cells[idx["unit"]]
        entry = {"raw": f"{raw_value} {raw_unit}", "raw_value": raw_value,
                 "raw_unit": raw_unit, "line": lineno, "value_si": None}
        if key in quantities and raw_value:
            si, diag = convert_field(raw_value, raw_unit, quantities[key], path=path, line=lineno)
            entry["value_si"] = si
            if diag:
                diagnostics.append(diag)
        entries[key] = entry
    for key in ("temperature", "cell_volume"):
        if key not in entries:
            raise FormatCorrupted(f"conditions must define '{key}'", path=path, line=header_lineno)
        if entries[key]["value_si"] is None:
            diagnostics.append({
                "code": "UNCONVERTIBLE_CONDITION", "path": path,
                "line": entries[key]["line"], "key": key,
                "raw": entries[key]["raw"],
                "message": f"required condition {key} could not be converted; retained at origin",
            })
    return {"kind": "conditions", "metadata": metadata, "entries": entries,
            "diagnostics": diagnostics}


def parse_baseline(text: str, *, path: str = "baseline.csv",
                   budget: Optional[Budget] = None, power_unit: Optional[str] = None):
    budget = budget or Budget("baseline_rows", 1_000_000)
    metadata, header_lineno, header = _header_block(text, path)
    cols = [h.lower() for h in header]
    for name in ("time", "level"):
        if name not in cols:
            raise FormatCorrupted(
                f"baseline header must contain time,level, found {header}",
                path=path, line=header_lineno,
            )
    idx = {h.lower(): i for i, h in enumerate(header)}
    tu = metadata.get("time_unit", "s")
    pu = power_unit or _require(metadata, "power_unit", path)
    anchors = []
    diagnostics = []
    for lineno, cells in _data_rows(text, header_lineno, path, len(header), budget):
        raw_t, raw_l = cells[idx["time"]], cells[idx["level"]]
        t_s, d1 = convert_field(raw_t, tu, "time", path=path, line=lineno)
        l_w, d2 = convert_field(raw_l, pu, "power", path=path, line=lineno)
        diagnostics.extend(d for d in (d1, d2) if d)
        if t_s is not None:
            anchors.append({"time_s": t_s, "level_w": l_w,
                            "raw_time": raw_t, "raw_level": raw_l, "line": lineno})
    anchors.sort(key=lambda a: a["time_s"])
    return {"kind": "baseline", "metadata": metadata,
            "units": {"time": tu, "power": pu},
            "method": metadata.get("method", "linear_anchors"),
            "anchors": anchors, "diagnostics": diagnostics}
