from __future__ import annotations

from typing import Any

from .errors import FormatCorrupt, SemanticInvalid
from .units import convert_value

MATERIAL_TYPES = {
    "power_curve",
    "injection_schedule",
    "concentration",
    "temperature",
    "cell_volume",
    "baseline_curve",
    "blank_curve",
}


def diagnostic(severity: str, code: str, path: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {"severity": severity, "code": code, "path": path, "message": message}
    item.update(extra)
    return item


def _raw_scalar(raw: Any, unit: str | None, path: str, diagnostics: list[dict[str, Any]], required: bool = True) -> dict[str, Any]:
    value, canonical_unit, issue = convert_value(raw, unit, path)
    if issue:
        diagnostics.append(issue)
    out = {"raw": raw, "raw_unit": unit, "value": value, "unit": canonical_unit}
    if required and issue:
        out["valid"] = False
    return out


def _series_points(
    rows: Any,
    path: str,
    diagnostics: list[dict[str, Any]],
    *,
    allow_gap: bool,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise SemanticInvalid("series must be a list of sampled points", path)
    points = []
    previous_time = None
    for index, row in enumerate(rows):
        point_path = f"{path}[{index}]"
        if not isinstance(row, dict):
            raise FormatCorrupt("each sampled point must be an object so its raw representation is retained", point_path)
        if "t" not in row or "power" not in row:
            raise FormatCorrupt("each sampled point requires t and power", point_path)
        time_value, time_unit, time_issue = convert_value(row["t"], row.get("t_unit", "s"), f"{point_path}.t")
        if time_issue:
            diagnostics.append(time_issue)
        explicit_gap = bool(row.get("gap", False))
        missing_power = row.get("power") is None or row.get("power") == ""
        is_gap = explicit_gap or missing_power
        if is_gap and not allow_gap:
            diagnostics.append(diagnostic("error", "UNEXPECTED_GAP", f"{point_path}.power", "gap is not permitted here"))
        power_value = power_unit = None
        power_issue = None
        if not missing_power:
            power_value, power_unit, power_issue = convert_value(
                row["power"], row.get("power_unit", "µJ/s"), f"{point_path}.power"
            )
            if power_issue:
                diagnostics.append(power_issue)
        elif allow_gap:
            diagnostics.append(diagnostic("warning", "DATA_GAP", f"{point_path}.power", "power sample is absent; the gap is retained"))
        if time_value is not None:
            if previous_time is not None and time_value <= previous_time:
                diagnostics.append(diagnostic("error", "NON_MONOTONIC_TIME", f"{point_path}.t", "sample time must be strictly increasing"))
            previous_time = time_value
        points.append(
            {
                "raw": row,
                "t": {"raw": row["t"], "unit": time_unit, "value": time_value},
                "power": {"raw": row["power"], "unit": power_unit, "value": power_value},
                "gap": is_gap,
                "note": row.get("note"),
            }
        )
    return points


def _normalize_power(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    points = _series_points(doc.get("samples"), "$.samples", diagnostics, allow_gap=True)
    return {
        "time_base": doc.get("time_base", {"zero": "experiment", "unit": "s"}),
        "points": points,
    }


def _normalize_baseline(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    points = _series_points(doc.get("samples"), "$.samples", diagnostics, allow_gap=False)
    return {"time_base": doc.get("time_base", {"zero": "experiment", "unit": "s"}), "points": points}


def _injection_rows(rows: Any, path: str, diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise SemanticInvalid("injections must be a non-empty list", path)
    result = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        point_path = f"{path}[{index}]"
        if not isinstance(row, dict):
            raise FormatCorrupt("each injection must be an object", point_path)
        injection_id = str(row.get("injection_id") or f"I{index + 1}")
        if injection_id in seen:
            diagnostics.append(diagnostic("error", "DUPLICATE_INJECTION_ID", f"{point_path}.injection_id", "duplicate stable injection identity", injection_id=injection_id))
        seen.add(injection_id)
        result.append(
            {
                "injection_id": injection_id,
                "ordinal": int(row.get("ordinal", index + 1)),
                "time": _raw_scalar(row.get("time"), row.get("time_unit", "s"), f"{point_path}.time", diagnostics),
                "duration": _raw_scalar(row.get("duration", 6.0), row.get("duration_unit", "s"), f"{point_path}.duration", diagnostics, required=False),
                "volume": _raw_scalar(row.get("volume"), row.get("volume_unit", "µL"), f"{point_path}.volume", diagnostics),
                "raw": row,
            }
        )
    result.sort(key=lambda item: (item["ordinal"], item["injection_id"]))
    return result


def _normalize_schedule(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return {"injections": _injection_rows(doc.get("injections"), "$.injections", diagnostics)}


def _normalize_concentration(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cell_species": doc.get("cell_species", "macromolecule"),
        "syringe_species": doc.get("syringe_species", "ligand"),
        "cell_concentration": _raw_scalar(
            doc.get("cell_concentration"), doc.get("cell_concentration_unit", "mmol/L"),
            "$.cell_concentration", diagnostics,
        ),
        "syringe_concentration": _raw_scalar(
            doc.get("syringe_concentration"), doc.get("syringe_concentration_unit", "mmol/L"),
            "$.syringe_concentration", diagnostics,
        ),
    }


def _normalize_temperature(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return {"temperature": _raw_scalar(doc.get("temperature"), doc.get("temperature_unit", "K"), "$.temperature", diagnostics)}


def _normalize_cell_volume(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cell_volume": _raw_scalar(doc.get("cell_volume"), doc.get("cell_volume_unit", "µL"), "$.cell_volume", diagnostics)}


def _normalize_blank(doc: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    controls = _injection_rows(doc.get("controls"), "$.controls", diagnostics)
    points = _series_points(doc.get("samples"), "$.samples", diagnostics, allow_gap=True)
    return {
        "time_base": doc.get("time_base", {"zero": "blank_experiment", "unit": "s"}),
        "correspondence": doc.get("correspondence", "by_ordinal"),
        "controls": controls,
        "points": points,
    }


NORMALIZERS = {
    "power_curve": _normalize_power,
    "injection_schedule": _normalize_schedule,
    "concentration": _normalize_concentration,
    "temperature": _normalize_temperature,
    "cell_volume": _normalize_cell_volume,
    "baseline_curve": _normalize_baseline,
    "blank_curve": _normalize_blank,
}


def normalize_material(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if not isinstance(document, dict):
        raise FormatCorrupt("material document must be a JSON object")
    material_type = document.get("material_type")
    if material_type not in MATERIAL_TYPES:
        raise SemanticInvalid(
            "unknown ITC material type",
            "$.material_type",
            {"material_type": material_type, "supported": sorted(MATERIAL_TYPES)},
        )
    diagnostics: list[dict[str, Any]] = []
    canonical = NORMALIZERS[material_type](document, diagnostics)
    canonical["material_type"] = material_type
    canonical["source_id"] = document.get("source_id") or f"source:{material_type}"
    status = "invalid" if any(item.get("severity") == "error" for item in diagnostics) else "active"
    return canonical, diagnostics, status
