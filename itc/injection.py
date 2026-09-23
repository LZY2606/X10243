"""Injection schedule, concentrations, temperature and cell volume.

All imported scalars remain :class:`~itc.units.Measurement` objects in the
material layer; canonical numbers used by computations live in a parallel
``norm`` block with diagnostics pointing back into the source payload.
"""
from __future__ import annotations

from typing import Any

from .errors import Diagnostic
from .units import parse_measurement, convert, Measurement


def _norm_scalar(raw: Any, target_unit: str, location: str,
                 diagnostics: list[Diagnostic]) -> float | None:
    m, ds = parse_measurement(raw, location)
    diagnostics.extend(ds)
    if m is None:
        return None
    v, ds2 = convert(m, target_unit, location)
    diagnostics.extend(ds2)
    return v


def normalize_schedule(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    volume_unit = payload.get("volume_unit", "uL")
    time_unit = payload.get("time_unit", "s")
    time_basis = payload.get("time_basis", "experiment_clock:t=0 at acquisition start")
    rows_in = payload.get("injections")
    if rows_in is None:
        diagnostics.append(Diagnostic("missing_injections",
                                      "injection schedule has no rows",
                                      "/injections"))
        rows_in = []
    injections: list[dict[str, Any]] = []
    for idx, row in enumerate(rows_in):
        loc = f"/injections/{idx}"
        if not isinstance(row, dict):
            diagnostics.append(Diagnostic("bad_injection_shape",
                                          "injection must be an object",
                                          loc, "error", row))
            continue
        entry: dict[str, Any] = {
            "index": idx,
            "ordinal": int(row.get("ordinal", idx + 1)),
            "label": row.get("label", f"#{idx + 1}"),
            "original": row,
            "note": row.get("note"),
        }
        # injection time
        t_raw = row.get("time", row.get("t"))
        tm, ds = parse_measurement(t_raw, f"{loc}/time")
        diagnostics.extend(ds)
        if tm is not None and not tm.unit:
            tm = Measurement(value=tm.value, unit=time_unit,
                             original=tm.original, basis=time_basis)
        t_s = None
        if tm is not None:
            t_s, ds2 = convert(tm, "s", f"{loc}/time")
            diagnostics.extend(ds2)
        entry["time_s"] = t_s
        # duration
        dur = None
        if row.get("duration") is not None:
            dur, ds3 = _norm_with_unit(row["duration"], time_unit, "s",
                                       f"{loc}/duration", diagnostics)
        entry["duration_s"] = dur
        # volume
        v_raw = row.get("volume", row.get("v"))
        vm, ds = parse_measurement(v_raw, f"{loc}/volume")
        diagnostics.extend(ds)
        if vm is not None and not vm.unit:
            vm = Measurement(value=vm.value, unit=volume_unit,
                             original=vm.original)
        v_l = None
        if vm is not None:
            v_l, ds2 = convert(vm, "L", f"{loc}/volume")
            diagnostics.extend(ds2)
        entry["volume_l"] = v_l
        entry["measurement"] = {"time": tm.to_dict() if tm else None,
                                "volume": vm.to_dict() if vm else None}
        injections.append(entry)

    injections.sort(key=lambda e: ((e["time_s"] is None),
                                   e["time_s"] if e["time_s"] is not None else 0.0,
                                   e["ordinal"]))
    last_t = None
    for e in injections:
        if e["time_s"] is None:
            diagnostics.append(Diagnostic("missing_injection_time",
                                          f"injection {e['label']} has no time",
                                          f"/injections/{e['index']}/time"))
        elif last_t is not None and e["time_s"] < last_t:
            diagnostics.append(Diagnostic("non_monotonic_injection_time",
                                          f"injection {e['label']} out of order",
                                          f"/injections/{e['index']}/time",
                                          "error"))
        last_t = e["time_s"] if e["time_s"] is not None else last_t
        if e["volume_l"] is None:
            diagnostics.append(Diagnostic("missing_injection_volume",
                                          f"injection {e['label']} has no volume",
                                          f"/injections/{e['index']}/volume"))
    # overlap detection (delivery intervals intersect)
    for i, e in enumerate(injections):
        t0 = e["time_s"]
        if t0 is None:
            continue
        for j in range(i + 1, len(injections)):
            o = injections[j]
            if o["time_s"] is None:
                continue
            if e["duration_s"] and o["time_s"] < t0 + e["duration_s"]:
                diagnostics.append(Diagnostic(
                    "overlapping_delivery",
                    f"delivery interval of {e['label']} overlaps {o['label']}",
                    f"/injections/{o['index']}/time", "warning"))
    return {
        "schedule_id": payload.get("schedule_id", "schedule"),
        "time_basis": time_basis,
        "volume_unit_original": volume_unit,
        "time_unit_original": time_unit,
        "injections": injections,
        "diagnostics": diagnostics,
    }


def _norm_with_unit(raw: Any, fallback_unit: str, target_unit: str,
                    location: str, diagnostics: list[Diagnostic]) -> float | None:
    m, ds = parse_measurement(raw, location)
    diagnostics.extend(ds)
    if m is not None and not m.unit and fallback_unit:
        m = Measurement(value=m.value, unit=fallback_unit, original=m.original)
    if m is None:
        return None
    v, ds2 = convert(m, target_unit, location)
    diagnostics.extend(ds2)
    return v


def normalize_concentration(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    role = payload.get("role", "titrant")  # titrant | macromolecule
    raw = payload.get("concentration", payload)
    m, ds = parse_measurement(raw.get("concentration") if isinstance(raw, dict)
                              and "concentration" in raw else raw,
                              "/concentration")
    diagnostics.extend(ds)
    moles = None
    if m is not None:
        moles, ds2 = convert(m, "M", "/concentration")
        diagnostics.extend(ds2)
    return {
        "concentration_id": payload.get("concentration_id", f"concentration_{role}"),
        "role": role,
        "analyte": payload.get("analyte"),
        "molar": moles,
        "measurement": m.to_dict() if m else None,
        "diagnostics": diagnostics,
    }


def normalize_temperature(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    raw = payload.get("temperature", payload)
    m, ds = parse_measurement(raw, "/temperature")
    diagnostics.extend(ds)
    celsius = None
    if m is not None and m.unit:
        celsius, ds2 = convert(m, "C", "/temperature")
        diagnostics.extend(ds2)
    elif m is not None:
        diagnostics.append(Diagnostic("missing_unit",
                                      "temperature has no unit (C or K)",
                                      "/temperature"))
    return {
        "temperature_id": payload.get("temperature_id", "temperature"),
        "celsius": celsius,
        "kelvin": (celsius + 273.15) if celsius is not None else None,
        "measurement": m.to_dict() if m else None,
        "diagnostics": diagnostics,
    }


def normalize_cell_volume(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[Diagnostic] = []
    raw = payload.get("cell_volume", payload)
    m, ds = parse_measurement(raw, "/cell_volume")
    diagnostics.extend(ds)
    litres = None
    if m is not None:
        litres, ds2 = convert(m, "L", "/cell_volume")
        diagnostics.extend(ds2)
        if litres is not None and not (0 < litres < 0.1):
            diagnostics.append(Diagnostic("implausible_cell_volume",
                                          f"cell volume {litres*1e6:.4g} µL "
                                          "outside (0, 100 mL)",
                                          "/cell_volume", "warning"))
    return {
        "cell_volume_id": payload.get("cell_volume_id", "cell_volume"),
        "litres": litres,
        "measurement": m.to_dict() if m else None,
        "diagnostics": diagnostics,
    }
