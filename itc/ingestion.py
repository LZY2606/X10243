"""Import raw ITC materials.

The lab drops in: power curves, the injection schedule (volume + time),
concentrations, run conditions (temperature, cell volume), a baseline curve
and blank-control curves. Baseline and blank are *separately* versioned.

Import keeps the original representation on every cell (``raw``); normalised
values live beside it. Unparseable bytes -> CorruptFormat; parseable but
meaningless/incomplete input -> semantic diagnostics (never silent defaults).
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from typing import Any

from .errors import CorruptFormat, SemanticImpasse, diagnostic
from .units import parse_quantity

MATERIAL_KINDS = (
    "power",
    "schedule",
    "concentration",
    "conditions",
    "baseline",
    "blank",
)


@dataclass
class ParsedSeries:
    times: list[float | None]
    powers: list[float | None]
    raw_rows: list[dict[str, str]]
    headers: list[str]
    diagnostics: list[dict[str, Any]]


def _read_csv(text: str, path: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except csv.Error as exc:
        raise CorruptFormat(f"CSV could not be parsed: {exc}", path=path,
                            line=getattr(exc, "line", None)) from exc
    if not rows:
        raise CorruptFormat("file is empty", path=path)
    headers = [h.strip() for h in rows[0]]
    if len(set(headers)) != len(headers):
        raise CorruptFormat("duplicate column headers", path=path, line=1)
    out: list[dict[str, str]] = []
    for i, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue
        if len(row) != len(headers):
            raise CorruptFormat(
                f"row has {len(row)} fields, header has {len(headers)}",
                path=path, line=i)
        out.append({headers[j]: row[j].strip() for j in range(len(headers))})
    return headers, out


def _find_col(headers: list[str], *names: str) -> str | None:
    wanted = {n.lower() for n in names}
    for h in headers:
        if h.lower() in wanted:
            return h
    return None


def parse_series_csv(text: str, *, path: str = "power.csv",
                     time_unit: str = "s", power_unit: str = "µcal/s"
                     ) -> ParsedSeries:
    """Parse a two-column time/power CSV keeping raw tokens and diagnostics."""
    headers, raw_rows = _read_csv(text, path)
    tcol = _find_col(headers, "time_s", "time", "t", "t_s")
    pcol = _find_col(headers, "power_ucal_s", "power", "p", "dp",
                     "power_µcal_s")
    if tcol is None or pcol is None:
        raise SemanticImpasse(
            "series must expose time and power columns",
            locations=[{"path": path, "headers": headers}])
    times: list[float | None] = []
    powers: list[float | None] = []
    diags: list[dict[str, Any]] = []
    prev_t: float | None = None
    for i, row in enumerate(raw_rows, start=2):
        tok_t, tok_p = row[tcol], row[pcol]
        try:
            tv = float(tok_t)
        except ValueError:
            tv = None
            diags.append(diagnostic("unparseable_sample",
                                    f"time token {tok_t!r} left in place",
                                    path=path, line=i, column=tcol))
        try:
            pv = float(tok_p)
        except ValueError:
            pv = None
            diags.append(diagnostic("unparseable_sample",
                                    f"power token {tok_p!r} left in place",
                                    path=path, line=i, column=pcol))
        if tv is not None and prev_t is not None and tv < prev_t:
            diags.append(diagnostic("non_monotonic_time",
                                    "time decreases relative to previous sample",
                                    path=path, line=i, column=tcol))
        if tv is not None:
            prev_t = tv
        times.append(tv)
        powers.append(pv)
    return ParsedSeries(times, powers, raw_rows, headers, diags)


def parse_schedule_csv(text: str, *, path: str = "schedule.csv") -> dict[str, Any]:
    headers, rows = _read_csv(text, path)
    inj_col = _find_col(headers, "injection", "inj", "id", "n")
    tcol = _find_col(headers, "time", "t", "start", "start_time")
    vcol = _find_col(headers, "volume", "vol", "inj_volume", "dv")
    if not (inj_col and tcol and vcol):
        raise SemanticImpasse(
            "schedule needs injection, time and volume columns",
            locations=[{"path": path, "headers": headers}])
    injections: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=2):
        num = row[inj_col]
        inj_index: int | None
        try:
            inj_index = int(float(num))
        except ValueError:
            inj_index = None
        tq = parse_quantity(row[tcol], "time", location=f"{path}:L{i}:{tcol}")
        vq = parse_quantity(row[vcol], "volume", location=f"{path}:L{i}:{vcol}")
        injections.append({
            "index": inj_index,
            "raw_index": num,
            "time": tq,
            "volume": vq,
            "raw": {"injection": num, "time": row[tcol], "volume": row[vcol]},
        })
    return {"kind": "schedule", "path": path, "headers": headers,
            "injections": injections}


def parse_concentration_csv(text: str, *, path: str = "concentration.csv"
                            ) -> dict[str, Any]:
    headers, rows = _read_csv(text, path)
    role_col = _find_col(headers, "role", "component", "species", "analyte")
    conc_col = _find_col(headers, "concentration", "conc", "c")
    if not (role_col and conc_col):
        raise SemanticImpasse(
            "concentration file needs role and concentration columns",
            locations=[{"path": path, "headers": headers}])
    entries: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=2):
        q = parse_quantity(row[conc_col], "concentration",
                           location=f"{path}:L{i}:{conc_col}")
        entries.append({"role": row[role_col], "concentration": q,
                        "raw": dict(row)})
    return {"kind": "concentration", "path": path, "headers": headers,
            "entries": entries}


def parse_conditions_json(text: str, *, path: str = "conditions.json"
                          ) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CorruptFormat(f"invalid JSON: {exc.msg}", path=path,
                            line=exc.lineno, column=exc.colno) from exc
    if not isinstance(data, dict):
        raise CorruptFormat("conditions must be a JSON object", path=path)
    temp_raw = data.get("temperature")
    cell_raw = data.get("cell_volume")
    if temp_raw is None or cell_raw is None:
        raise SemanticImpasse(
            "conditions must provide temperature and cell_volume",
            locations=[{"path": path, "missing":
                        [k for k in ("temperature", "cell_volume")
                         if data.get(k) is None]}])
    tq = parse_quantity(temp_raw, "temperature", location=f"{path}:temperature")
    cq = parse_quantity(cell_raw, "volume", location=f"{path}:cell_volume")
    return {
        "kind": "conditions",
        "path": path,
        "temperature": tq,
        "cell_volume": cq,
        "raw": data,
        "note": data.get("note", ""),
    }


def read_case_pack(root: str) -> dict[str, Any]:
    """Read a case-pack directory; raise if the manifest itself is corrupt."""
    manifest_path = os.path.join(root, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError as exc:
        raise CorruptFormat("manifest.json not found", path=manifest_path) from exc
    except json.JSONDecodeError as exc:
        raise CorruptFormat(f"invalid manifest JSON: {exc.msg}",
                            path=manifest_path, line=exc.lineno,
                            column=exc.colno) from exc

    materials: dict[str, Any] = {}
    for spec in manifest.get("materials", []):
        rel = spec["path"]
        mpath = os.path.join(root, rel)
        try:
            with open(mpath, encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError as exc:
            raise CorruptFormat(f"material listed in manifest is missing: {rel}",
                                path=mpath) from exc
        kind = spec["kind"]
        if kind in ("power", "baseline", "blank"):
            materials[spec["id"]] = {
                "spec": spec,
                "series": parse_series_csv(
                    text, path=rel,
                    time_unit=spec.get("time_unit", "s"),
                    power_unit=spec.get("power_unit", "µcal/s")),
            }
        elif kind == "schedule":
            materials[spec["id"]] = {"spec": spec,
                                     "doc": parse_schedule_csv(text, path=rel)}
        elif kind == "concentration":
            materials[spec["id"]] = {"spec": spec,
                                     "doc": parse_concentration_csv(text, path=rel)}
        elif kind == "conditions":
            materials[spec["id"]] = {"spec": spec,
                                     "doc": parse_conditions_json(text, path=rel)}
        else:
            raise CorruptFormat(f"unknown material kind {kind!r}", path=manifest_path)
    return {"manifest": manifest, "materials": materials, "root": root}
