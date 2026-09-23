"""Parse case-pack material files into versioned immutable materials.

The original textual representation is always retained. Normalization to SI
happens only via the explicit view helpers (the derived layer). Values that
cannot be converted stay where they were with a positional diagnostic instead
of becoming zero / None / a plotting-friendly default.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import Diagnostic, Material
from .units import parse_quantity


def parse_material(doc: dict[str, Any]) -> Material:
    mtype = doc.get("material_type")
    if mtype not in ("power", "injections", "concentration", "conditions",
                     "baseline", "blank"):
        raise ValueError(f"unknown material_type {mtype!r}")
    version = int(doc.get("version", 1))
    diagnostics: list[dict[str, Any]] = []
    if mtype == "power":
        diagnostics = _check_series(doc, "rows", "t", "p", ("time", "power"))
    elif mtype == "blank":
        diagnostics = _check_series(doc, "rows", "t", "p", ("time", "power"))
        diagnostics += _check_injections(doc)
    elif mtype == "injections":
        diagnostics = _check_injections(doc)
    elif mtype == "concentration":
        diagnostics = _check_concentration(doc)
    elif mtype == "conditions":
        diagnostics = _check_conditions(doc)
    elif mtype == "baseline":
        diagnostics = _check_series(doc, "points", "t", "p", ("time", "power"))
    import json
    from .models import stable_id
    material_id = stable_id("mat", {"type": mtype, "body": doc})
    return Material(material_id=material_id, material_type=mtype, version=version,
                    raw_representation=doc, diagnostics=diagnostics,
                    supersedes=doc.get("supersedes"))


def _check_series(doc, table, tk, pk, kinds):
    out = []
    rows = doc.get(table, [])
    for i, row in enumerate(rows):
        if row.get("missing"):
            continue
        qt = parse_quantity(row.get(tk), doc.get("time_unit"), kinds[0])
        if qt.diagnostic:
            out.append(Diagnostic("unconvertible_value", qt.diagnostic,
                                  {table: i, "field": tk}, "format_corrupt").to_dict())
        qp = parse_quantity(row.get(pk), doc.get("power_unit"), kinds[1])
        if qp.diagnostic:
            out.append(Diagnostic("unconvertible_value", qp.diagnostic,
                                  {table: i, "field": pk}, "format_corrupt").to_dict())
    return out


def _check_injections(doc):
    out = []
    for i, inj in enumerate(doc.get("injections", [])):
        qt = parse_quantity(inj.get("time"), doc.get("time_unit"), "time")
        if qt.diagnostic:
            out.append(Diagnostic("unconvertible_value", qt.diagnostic,
                                  {"injections": i, "field": "time"}, "format_corrupt").to_dict())
        qv = parse_quantity(inj.get("volume"), doc.get("volume_unit"), "volume")
        if qv.diagnostic:
            out.append(Diagnostic("unconvertible_value", qv.diagnostic,
                                  {"injections": i, "field": "volume"}, "format_corrupt").to_dict())
    return out


def _check_concentration(doc):
    out = []
    for key, label in (("syringe_concentration", "syringe"), ("cell_concentration", "cell")):
        spec = doc.get(key)
        if not isinstance(spec, dict):
            out.append(Diagnostic("missing_field", f"{key} missing",
                                  {"field": key}, "format_corrupt").to_dict())
            continue
        q = parse_quantity(spec.get("raw"), spec.get("unit"), "concentration")
        if q.diagnostic:
            out.append(Diagnostic("unconvertible_value", q.diagnostic,
                                  {"field": label}, "format_corrupt").to_dict())
    return out


def _check_conditions(doc):
    out = []
    temp = doc.get("temperature")
    if not isinstance(temp, dict):
        out.append(Diagnostic("missing_field", "temperature missing",
                              {"field": "temperature"}, "format_corrupt").to_dict())
    else:
        q = parse_quantity(temp.get("raw"), temp.get("unit"), "temperature")
        if q.diagnostic:
            out.append(Diagnostic("unconvertible_value", q.diagnostic,
                                  {"field": "temperature"}, "format_corrupt").to_dict())
    vol = doc.get("cell_volume")
    if not isinstance(vol, dict):
        out.append(Diagnostic("missing_field", "cell_volume missing",
                              {"field": "cell_volume"}, "format_corrupt").to_dict())
    else:
        q = parse_quantity(vol.get("raw"), vol.get("unit"), "volume")
        if q.diagnostic:
            out.append(Diagnostic("unconvertible_value", q.diagnostic,
                                  {"field": "cell_volume"}, "format_corrupt").to_dict())
    return out


# ---- derived-layer views (normalization happens here only) -------------------
def power_view(material: Material):
    """Return (times_s, powers_w, missing_indices, diagnostics).

    Missing rows keep their raw entries; arrays carry NaN at those positions and
    gap intervals are derived for integration-window checks.
    """
    doc = material.raw_representation
    rows = doc.get("rows", [])
    times, powers, missing = [], [], []
    diags = list(material.diagnostics)
    for i, row in enumerate(rows):
        qt = parse_quantity(row.get("t"), doc.get("time_unit"), "time")
        qp = parse_quantity(row.get("p"), doc.get("power_unit"), "power")
        if row.get("missing"):
            # gaps still carry a time coordinate when available
            t = qt.normalized() if qt.value is not None else math.nan
            times.append(t)
            powers.append(math.nan)
            missing.append(i)
            continue
        if qt.value is None or qp.value is None:
            times.append(math.nan)
            powers.append(math.nan)
            missing.append(i)
            continue
        times.append(qt.normalized())
        powers.append(qp.normalized())
    return np.asarray(times, dtype=float), np.asarray(powers, dtype=float), missing, diags


def baseline_view(material: Material):
    doc = material.raw_representation
    pts = doc.get("points", [])
    times, powers = [], []
    for row in pts:
        qt = parse_quantity(row.get("t"), doc.get("time_unit"), "time")
        qp = parse_quantity(row.get("p"), doc.get("power_unit"), "power")
        times.append(qt.normalized())
        powers.append(qp.normalized())
    return np.asarray(times), np.asarray(powers)


def injections_view(material: Material):
    doc = material.raw_representation
    out = []
    for inj in doc.get("injections", []):
        qt = parse_quantity(inj.get("time"), doc.get("time_unit"), "time")
        qv = parse_quantity(inj.get("volume"), doc.get("volume_unit"), "volume")
        out.append({
            "index": int(inj["index"]),
            "time_s": qt.normalized(),
            "volume_l": qv.normalized(),
            "time_raw": inj.get("time"),
            "volume_raw": inj.get("volume"),
        })
    return out


def concentration_view(material: Material):
    doc = material.raw_representation
    syr = parse_quantity(doc["syringe_concentration"]["raw"],
                         doc["syringe_concentration"]["unit"], "concentration")
    cell = parse_quantity(doc["cell_concentration"]["raw"],
                          doc["cell_concentration"]["unit"], "concentration")
    return {"syringe_m": syr.normalized(), "cell_m": cell.normalized(),
            "syringe_raw": doc["syringe_concentration"], "cell_raw": doc["cell_concentration"]}


def conditions_view(material: Material):
    doc = material.raw_representation
    t = parse_quantity(doc["temperature"]["raw"], doc["temperature"]["unit"], "temperature")
    v = parse_quantity(doc["cell_volume"]["raw"], doc["cell_volume"]["unit"], "volume")
    return {"temperature_k": t.normalized(), "cell_volume_l": v.normalized(),
            "temperature_raw": doc["temperature"], "cell_volume_raw": doc["cell_volume"]}


def blank_view(material: Material):
    times, powers, missing, diags = power_view(material)
    return {"times_s": times, "powers_w": powers, "missing": missing,
            "injections": injections_view(material), "diagnostics": diags}


def gap_intervals(times: np.ndarray, missing: list[int]):
    """Closed coordinate gaps: from last good sample before a missing run to the
    first good sample after it."""
    gaps = []
    missing_set = set(missing)
    run = []
    for i in range(len(times)):
        if i in missing_set:
            run.append(i)
        elif run:
            lo = times[run[0] - 1] if run[0] > 0 and not np.isnan(times[run[0] - 1]) else times[run[0]]
            hi = times[i]
            gaps.append((float(lo), float(hi)))
            run = []
    if run:
        lo = times[run[0] - 1] if run[0] > 0 and not np.isnan(times[run[0] - 1]) else math.nan
        gaps.append((float(lo), math.inf))
    return gaps
