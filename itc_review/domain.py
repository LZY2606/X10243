from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DomainStatus(str, Enum):
    OK = "ok"
    DIAGNOSTIC = "diagnostic"
    INVALID = "invalid"
    STALE = "stale"


class DomainError(Exception):
    status_code = 422

    def __init__(self, code: str, message: str, diagnostics: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.diagnostics = diagnostics or []


class FormatCorrupt(DomainError):
    status_code = 400


class BudgetExhausted(DomainError):
    status_code = 507


class SemanticNoSolution(DomainError):
    status_code = 422


DIMENSION_FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "minute": 60.0},
    "power": {"uw": 1.0, "µw": 1.0, "μw": 1.0, "mw": 1000.0, "w": 1_000_000.0},
    "volume": {"ul": 1.0, "µl": 1.0, "μl": 1.0, "ml": 1000.0, "l": 1_000_000.0},
    "concentration": {"um": 1.0, "µm": 1.0, "μm": 1.0, "mm": 1000.0, "m": 1_000_000.0},
    "energy": {"uj": 1.0, "µj": 1.0, "μj": 1.0, "mj": 1000.0, "j": 1_000_000.0, "cal": 4_184_000.0},
}

MATERIAL_KINDS = {
    "power_series",
    "injection_schedule",
    "concentration",
    "temperature",
    "cell_volume",
    "baseline",
    "blank_curve",
}

MODEL_ORDER = ("dilution", "one_site", "two_site")
EPSILON = 1e-10


@dataclass
class Budget:
    max_points: int = 250_000
    max_operations: int = 5_000_000
    operations: int = 0

    def charge_points(self, count: int) -> None:
        if count > self.max_points:
            raise BudgetExhausted(
                "RESOURCE_BUDGET_POINTS",
                f"input contains {count} points; limit is {self.max_points}",
                [{"position": "$.rows", "expected": f"<= {self.max_points}", "actual": count}],
            )

    def charge(self, count: int) -> None:
        self.operations += count
        if self.operations > self.max_operations:
            raise BudgetExhausted(
                "RESOURCE_BUDGET_OPERATIONS",
                f"operation budget {self.max_operations} exhausted",
                [{"position": "$.calculation", "budget": self.max_operations}],
            )


@dataclass
class MaterialVersion:
    material_id: str
    version: int
    version_id: str
    kind: str
    payload: dict[str, Any]
    raw: dict[str, Any]
    unit_bindings: dict[str, Any]
    basis: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    status: DomainStatus = DomainStatus.OK
    content_hash: str = ""


def diagnostic(position: str, code: str, message: str, raw: Any = None, **extra: Any) -> dict[str, Any]:
    item = {"position": position, "code": code, "message": message, "raw": raw}
    item.update(extra)
    return item


def _number(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def convert_scalar(
    raw: Any,
    unit: str | None,
    dimension: str,
    position: str,
    *,
    temperature: bool = False,
) -> tuple[float | None, dict[str, Any] | None]:
    value = _number(raw)
    if value is None:
        return None, diagnostic(
            position,
            "UNCONVERTIBLE_VALUE",
            "value remains in raw form and was not used by derived calculations",
            raw,
            requested_unit=unit,
        )
    if temperature:
        if unit is None:
            return None, diagnostic(position, "UNIT_MISSING", "temperature requires K, C, or °C", raw)
        normalized = unit.strip().lower()
        if normalized in {"k", "kelvin"}:
            return value, None
        if normalized in {"c", "°c", "degc", "celsius"}:
            return value + 273.15, None
        return None, diagnostic(position, "UNCONVERTIBLE_UNIT", "unsupported temperature unit", raw, unit=unit)
    factors = DIMENSION_FACTORS[dimension]
    if unit is None:
        return None, diagnostic(position, "UNIT_MISSING", f"{dimension} requires a unit", raw)
    factor = factors.get(unit.strip().lower())
    if factor is None:
        return None, diagnostic(position, "UNCONVERTIBLE_UNIT", "unsupported unit; raw value retained", raw, unit=unit)
    return value * factor, None


def stable_hash(payload: Any) -> str:
    import hashlib
    import json

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_material(material_id: str, version: int, raw_event: dict[str, Any]) -> MaterialVersion:
    kind = raw_event.get("kind")
    if kind not in MATERIAL_KINDS:
        raise FormatCorrupt(
            "FORMAT_UNKNOWN_MATERIAL_KIND",
            f"unknown material kind: {kind!r}",
            [diagnostic("$.kind", "FORMAT_UNKNOWN_MATERIAL_KIND", "expected a versioned ITC material", kind)],
        )
    payload = raw_event.get("payload")
    if not isinstance(payload, dict):
        raise FormatCorrupt(
            "FORMAT_PAYLOAD_OBJECT",
            "material payload must be an object",
            [diagnostic("$.payload", "FORMAT_PAYLOAD_OBJECT", "object required", type(payload).__name__)],
        )
    version_id = raw_event.get("version_id") or f"{material_id}_v{version}"
    raw_repr = raw_event.get("raw", payload)
    unit_bindings = raw_event.get("units", payload.get("units", {}))
    basis = raw_event.get("basis", payload.get("basis", {}))
    material = MaterialVersion(
        material_id=material_id,
        version=version,
        version_id=version_id,
        kind=kind,
        payload=payload,
        raw=raw_repr if isinstance(raw_repr, dict) else {"value": raw_repr},
        unit_bindings=unit_bindings if isinstance(unit_bindings, dict) else {},
        basis=basis if isinstance(basis, dict) else {},
    )
    normalizers = {
        "power_series": _normalize_power,
        "injection_schedule": _normalize_schedule,
        "concentration": _normalize_concentration,
        "temperature": _normalize_temperature,
        "cell_volume": _normalize_cell_volume,
        "baseline": _normalize_baseline,
        "blank_curve": _normalize_blank,
    }
    normalizers[kind](material)
    material.diagnostics = sorted(material.diagnostics, key=lambda item: (item["position"], item["code"]))
    material.status = DomainStatus.INVALID if material.diagnostics else DomainStatus.OK
    material.content_hash = stable_hash(
        {
            "kind": kind,
            "payload": material.payload.get("normalized", material.payload),
            "raw": material.raw,
            "units": material.unit_bindings,
            "basis": material.basis,
        }
    )
    return material


def _require_rows(material: MaterialVersion) -> list[Any]:
    rows = material.payload.get("rows")
    if not isinstance(rows, list):
        material.diagnostics.append(diagnostic("$.payload.rows", "FORMAT_ROWS_MISSING", "rows must be a list"))
        return []
    return rows


def _normalize_power(material: MaterialVersion) -> None:
    rows = _require_rows(material)
    budget = Budget(max_points=250_000)
    budget.charge_points(len(rows))
    time_unit = material.unit_bindings.get("time", material.payload.get("time_unit"))
    power_unit = material.unit_bindings.get("power", material.payload.get("power_unit"))
    points: list[list[float | None]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            material.diagnostics.append(diagnostic(f"$.payload.rows[{index}]", "FORMAT_ROW_SHAPE", "expected [time,power]", row))
            continue
        t, d1 = convert_scalar(row[0], time_unit, "time", f"$.payload.rows[{index}][0]")
        p, d2 = convert_scalar(row[1], power_unit, "power", f"$.payload.rows[{index}][1]")
        if d1:
            material.diagnostics.append(d1)
        if d2:
            material.diagnostics.append(d2)
        points.append([t, p])
    sample_gap = _number(material.payload.get("gap_tolerance_s"))
    gap_tolerance = sample_gap if sample_gap and sample_gap > 0 else 2.5
    material.payload["normalized"] = {
        "points": points,
        "gap_tolerance_s": gap_tolerance,
        "segments": infer_segments(points, gap_tolerance),
    }
    material.basis.setdefault("time", time_unit or "s")
    material.basis.setdefault("power", power_unit or "uW")


def infer_segments(points: list[list[float | None]], gap_tolerance: float) -> list[dict[str, float]]:
    valid = [(t, p) for t, p in points if t is not None and p is not None]
    valid.sort(key=lambda item: item[0])
    segments: list[dict[str, Any]] = []
    start_t = prev_t = None
    observed: set[float] = set()
    for t, _ in valid:
        observed.add(t)
        if start_t is None:
            start_t = prev_t = t
            continue
        if t - prev_t > gap_tolerance:
            segments.append({"start": start_t, "end": prev_t})
            start_t = t
        prev_t = t
    if start_t is not None:
        segments.append({"start": start_t, "end": prev_t})
    for segment in segments:
        segment["start_observed"] = segment["start"] in observed
        segment["end_observed"] = segment["end"] in observed
    return segments


def _normalize_schedule(material: MaterialVersion) -> None:
    rows = _require_rows(material)
    time_unit = material.unit_bindings.get("time", material.payload.get("time_unit"))
    volume_unit = material.unit_bindings.get("volume", material.payload.get("volume_unit"))
    injections = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            material.diagnostics.append(diagnostic(f"$.payload.rows[{index}]", "FORMAT_ROW_SHAPE", "injection must be an object", row))
            continue
        injection_id = str(row.get("injection_id") or f"inj_{index + 1:03d}")
        if injection_id in seen:
            material.diagnostics.append(
                diagnostic(f"$.payload.rows[{index}].injection_id", "IDENTITY_DUPLICATE", "duplicate injection identity", injection_id)
            )
        seen.add(injection_id)
        t, d1 = convert_scalar(row.get("time"), time_unit, "time", f"$.payload.rows[{index}].time")
        volume, d2 = convert_scalar(row.get("volume"), volume_unit, "volume", f"$.payload.rows[{index}].volume")
        if d1:
            material.diagnostics.append(d1)
        if d2:
            material.diagnostics.append(d2)
        injections.append(
            {
                "injection_id": injection_id,
                "ordinal": int(row.get("ordinal", index + 1)),
                "time_s": t,
                "volume_ul": volume,
                "raw": {"time": row.get("time"), "volume": row.get("volume")},
            }
        )
    injections.sort(key=lambda item: (item["ordinal"], item["injection_id"]))
    material.payload["normalized"] = {"injections": injections}


def _normalize_concentration(material: MaterialVersion) -> None:
    unit = material.unit_bindings.get("concentration", material.payload.get("concentration_unit"))
    result: dict[str, Any] = {}
    for name in ("cell", "syringe"):
        raw = material.payload.get(f"{name}_concentration")
        value, diag = convert_scalar(raw, unit, "concentration", f"$.payload.{name}_concentration")
        result[f"{name}_um"] = value
        if diag:
            material.diagnostics.append(diag)
    material.payload["normalized"] = result


def _normalize_temperature(material: MaterialVersion) -> None:
    unit = material.unit_bindings.get("temperature", material.payload.get("temperature_unit"))
    raw = material.payload.get("temperature")
    value, diag = convert_scalar(raw, unit, "temperature", "$.payload.temperature", temperature=True)
    if diag:
        material.diagnostics.append(diag)
    material.payload["normalized"] = {"temperature_k": value, "display_c": None if value is None else value - 273.15}


def _normalize_cell_volume(material: MaterialVersion) -> None:
    unit = material.unit_bindings.get("volume", material.payload.get("volume_unit"))
    value, diag = convert_scalar(material.payload.get("cell_volume"), unit, "volume", "$.payload.cell_volume")
    if diag:
        material.diagnostics.append(diag)
    material.payload["normalized"] = {"cell_volume_ul": value}


def _normalize_baseline(material: MaterialVersion) -> None:
    energy_unit = material.unit_bindings.get("energy", material.payload.get("energy_unit"))
    raw = material.payload.get("offset")
    value, diag = convert_scalar(raw, energy_unit, "energy", "$.payload.offset")
    if diag:
        material.diagnostics.append(diag)
    material.payload["normalized"] = {
        "offset_uj": value,
        "model": material.payload.get("model", "constant_offset"),
    }


def _normalize_blank(material: MaterialVersion) -> None:
    rows = _require_rows(material)
    energy_unit = material.unit_bindings.get("energy", material.payload.get("energy_unit"))
    entries = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            material.diagnostics.append(diagnostic(f"$.payload.rows[{index}]", "FORMAT_ROW_SHAPE", "blank row must be an object", row))
            continue
        injection_id = str(row.get("injection_id") or f"blank_{index + 1:03d}")
        heat, diag = convert_scalar(row.get("heat"), energy_unit, "energy", f"$.payload.rows[{index}].heat")
        if diag:
            material.diagnostics.append(diag)
        entries.append({"injection_id": injection_id, "heat_uj": heat, "raw": row.get("heat")})
    entries.sort(key=lambda item: item["injection_id"])
    material.payload["normalized"] = {"entries": entries}


def _interp(points: np.ndarray, x: float) -> float:
    return float(np.interp(x, points[:, 0], points[:, 1]))


def integrate_window(points: list[list[float | None]], start: float, end: float) -> tuple[float, int]:
    valid = np.array([(t, p) for t, p in points if t is not None and p is not None], dtype=float)
    if valid.size == 0:
        raise SemanticNoSolution("SEMANTIC_NO_POWER_POINTS", "no usable power samples", [])
    valid = valid[valid[:, 0].argsort()]
    grid_t = np.concatenate(([start], valid[(valid[:, 0] > start) & (valid[:, 0] < end), 0], [end]))
    grid_p = np.interp(grid_t, valid[:, 0], valid[:, 1])
    return float(np.trapz(grid_p, grid_t)), int(len(grid_t))


def segment_for_window(segments: list[dict[str, Any]], start: float, end: float) -> dict[str, Any] | None:
    for segment in segments:
        if segment["start"] - EPSILON <= start and end <= segment["end"] + EPSILON:
            return segment
    return None


def compute_integrations(
    power: MaterialVersion,
    schedule: MaterialVersion,
    baseline: MaterialVersion,
    blank: MaterialVersion | None,
    *,
    window_s: float = 120.0,
    budget: Budget | None = None,
) -> dict[str, Any]:
    budget = budget or Budget()
    diagnostics: list[dict[str, Any]] = []
    if power.status != DomainStatus.OK:
        diagnostics.append(
            diagnostic("$.materials.power_series", "INPUT_INCOMPLETE", "power series has unresolved diagnostics", power.version_id)
        )
    if schedule.status != DomainStatus.OK:
        diagnostics.append(
            diagnostic("$.materials.injection_schedule", "INPUT_INCOMPLETE", "injection schedule has unresolved diagnostics", schedule.version_id)
        )
    if baseline.status != DomainStatus.OK:
        diagnostics.append(
            diagnostic("$.materials.baseline", "INPUT_INCOMPLETE", "baseline has unresolved diagnostics", baseline.version_id)
        )
    if diagnostics:
        return {"status": DomainStatus.DIAGNOSTIC.value, "diagnostics": diagnostics, "results": []}

    normalized_power = power.payload["normalized"]
    points = normalized_power["points"]
    segments = normalized_power["segments"]
    injections = schedule.payload["normalized"]["injections"]
    offset = baseline.payload["normalized"]["offset_uj"]
    blank_entries = {e["injection_id"]: e["heat_uj"] for e in blank.payload["normalized"]["entries"]} if blank else {}
    results: list[dict[str, Any]] = []
    for injection in injections:
        budget.charge(10)
        injection_id = injection["injection_id"]
        if injection["time_s"] is None or injection["volume_ul"] is None:
            results.append(
                {
                    "injection_id": injection_id,
                    "ordinal": injection["ordinal"],
                    "status": DomainStatus.DIAGNOSTIC.value,
                    "reason_code": "INJECTION_INPUT_INCOMPLETE",
                    "raw": injection["raw"],
                }
            )
            continue
        start = float(injection["time_s"])
        end = start + window_s
        segment = segment_for_window(segments, start, end)
        if segment is None:
            results.append(
                {
                    "injection_id": injection_id,
                    "ordinal": injection["ordinal"],
                    "window": {"start_s": start, "end_s": end},
                    "status": DomainStatus.INVALID.value,
                    "reason_code": "WINDOW_CROSSES_DATA_GAP",
                    "segments": segments,
                }
            )
            continue
        endpoint_flags = {
            "start_at_gap_edge": bool(segment.get("start_observed")) and abs(start - segment["start"]) <= EPSILON,
            "end_at_gap_edge": bool(segment.get("end_observed")) and abs(end - segment["end"]) <= EPSILON,
        }
        gross_uj, sample_count = integrate_window(points, start, end)
        blank_heat = blank_entries.get(injection_id)
        if blank is None:
            blank_status = "blank_not_provided"
            net_uj = gross_uj - (offset or 0.0)
        elif blank_heat is None:
            blank_status = "blank_mapping_missing"
            net_uj = gross_uj - (offset or 0.0)
        else:
            blank_status = "blank_subtracted"
            net_uj = gross_uj - (offset or 0.0) - blank_heat
        results.append(
            {
                "injection_id": injection_id,
                "ordinal": injection["ordinal"],
                "window": {"start_s": start, "end_s": end},
                "segment": segment,
                "endpoint_flags": endpoint_flags,
                "gross_uj": gross_uj,
                "baseline_offset_uj": offset,
                "blank_heat_uj": blank_heat,
                "blank_status": blank_status,
                "net_uj": net_uj,
                "volume_ul": injection["volume_ul"],
                "sample_count": sample_count,
                "status": DomainStatus.OK.value,
            }
        )
    results.sort(key=lambda item: (item["ordinal"], item["injection_id"]))
    return {
        "status": DomainStatus.OK.value,
        "diagnostics": diagnostics,
        "results": results,
        "window_rule": {
            "window_s": window_s,
            "overlap_allowed": True,
            "gap_rule": "integration window must lie within one observed segment; endpoints at observed gap edges are valid",
        },
    }


def dilution_rule_payload(blank: MaterialVersion | None, integration_output: dict[str, Any]) -> dict[str, Any]:
    mapped = sum(1 for item in integration_output.get("results", []) if item.get("blank_status") == "blank_subtracted")
    missing = [item["injection_id"] for item in integration_output.get("results", []) if item.get("blank_status") == "blank_mapping_missing"]
    return {
        "rule": "per_injection_identity",
        "blank_version_id": None if blank is None else blank.version_id,
        "mapped_injections": mapped,
        "missing_mappings": missing,
        "statement": "dilution heat is subtracted only when the blank curve maps the same injection_id; missing mappings are explicit",
    }


def _one_site_heat(volumes: np.ndarray, cell0: float, syringe: float, cell_volume: float, n: float, kd: float, dh: float) -> np.ndarray:
    total_ligand = np.cumsum(volumes) * syringe / cell_volume
    heat = np.zeros(len(volumes), dtype=float)
    prev_complex = 0.0
    for i, ligand in enumerate(total_ligand):
        total = cell0 + ligand + 1.0 / max(kd, EPSILON)
        root = math.sqrt(max(total * total - 4.0 * cell0 * ligand, 0.0))
        complex_um = 0.5 * (total - root)
        heat[i] = (complex_um - prev_complex) * cell_volume * dh / 1000.0
        prev_complex = complex_um
    return heat


def _two_site_heat(volumes: np.ndarray, cell0: float, syringe: float, cell_volume: float, params: np.ndarray) -> np.ndarray:
    n1, kd1, dh1, n2, kd2, dh2 = params
    return _one_site_heat(volumes, cell0 * n1, syringe, cell_volume, 1.0, kd1, dh1) + _one_site_heat(
        volumes, cell0 * n2, syringe, cell_volume, 1.0, kd2, dh2
    )


def _model_prediction(model: str, params: np.ndarray, x: dict[str, np.ndarray | float]) -> np.ndarray:
    volumes = x["volumes"]  # type: ignore[assignment]
    cell0 = float(x["cell0"])
    syringe = float(x["syringe"])
    cell_volume = float(x["cell_volume"])
    if model == "dilution":
        return np.full(len(volumes), params[0], dtype=float)
    if model == "one_site":
        return _one_site_heat(volumes, cell0, syringe, cell_volume, params[0], params[1], params[2])
    return _two_site_heat(volumes, cell0, syringe, cell_volume, params)


def _initial_grids(model: str) -> list[np.ndarray]:
    if model == "dilution":
        return [np.array([v], dtype=float) for v in (-5.0, -1.0, 0.0, 1.0, 5.0)]
    if model == "one_site":
        grids = []
        for n in (0.5, 1.0, 1.5):
            for kd in (0.1, 1.0, 10.0, 100.0):
                for dh in (-50.0, -20.0, -5.0, 5.0, 20.0, 50.0):
                    grids.append(np.array([n, kd, dh], dtype=float))
        return grids
    grids = []
    for kd1 in (0.1, 1.0, 10.0):
        for kd2 in (10.0, 100.0):
            for dh1 in (-30.0, -10.0, 10.0):
                for dh2 in (-30.0, -10.0, 10.0):
                    grids.append(np.array([1.0, kd1, dh1, 1.0, kd2, dh2], dtype=float))
    return grids


def _bounds(model: str) -> tuple[np.ndarray, np.ndarray]:
    if model == "dilution":
        return np.array([-100.0]), np.array([100.0])
    if model == "one_site":
        return np.array([0.05, 1e-3, -500.0]), np.array([5.0, 1e4, 500.0])
    return (
        np.array([0.05, 1e-3, -500.0, 0.05, 1e-3, -500.0]),
        np.array([5.0, 1e4, 500.0, 5.0, 1e4, 500.0]),
    )


def _fit_model(model: str, x: dict[str, Any], y: np.ndarray, budget: Budget) -> dict[str, Any]:
    lower, upper = _bounds(model)
    best_params: np.ndarray | None = None
    best_sse = float("inf")
    for start in _initial_grids(model):
        params = start.copy()
        step = np.maximum(np.abs(params) * 0.5, 0.05)
        for _ in range(24):
            budget.charge(len(y) * len(params))
            improved = False
            for j in range(len(params)):
                for direction in (-1.0, 1.0):
                    trial = params.copy()
                    trial[j] = np.clip(trial[j] + direction * step[j], lower[j], upper[j])
                    pred = _model_prediction(model, trial, x)
                    sse = float(np.sum((y - pred) ** 2))
                    if sse + 1e-12 < best_sse or (best_params is not None and sse < float(np.sum((y - _model_prediction(model, params, x)) ** 2)) - 1e-12):
                        params = trial
                        improved = True
            if not improved:
                step *= 0.5
                if float(np.max(step)) < 1e-4:
                    break
        pred = _model_prediction(model, params, x)
        sse = float(np.sum((y - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_params = params.copy()
    if best_params is None or not np.isfinite(best_sse):
        raise SemanticNoSolution("SEMANTIC_NO_SOLUTION", f"model {model} produced no finite candidate", [])
    return {"params": best_params, "sse": best_sse}


def fit_models(
    integration_output: dict[str, Any],
    concentration: MaterialVersion,
    cell_volume_material: MaterialVersion,
    decisions: list[dict[str, Any]],
    *,
    budget: Budget | None = None,
) -> dict[str, Any]:
    budget = budget or Budget()
    if integration_output.get("status") != DomainStatus.OK.value:
        return {"status": DomainStatus.DIAGNOSTIC.value, "diagnostics": integration_output.get("diagnostics", []), "candidates": []}
    discarded = {d.get("injection_id") for d in decisions if d.get("decision_type") == "discard_first_injection"}
    included = [
        item
        for item in integration_output["results"]
        if item.get("status") == DomainStatus.OK.value and item["injection_id"] not in discarded
    ]
    if len(included) < 3:
        return {
            "status": DomainStatus.DIAGNOSTIC.value,
            "diagnostics": [diagnostic("$.integrations", "INPUT_INCOMPLETE", "at least three valid non-discarded injections are required")],
            "candidates": [],
        }
    conc = concentration.payload["normalized"]
    cell_volume = cell_volume_material.payload["normalized"]["cell_volume_ul"]
    if conc.get("cell_um") is None or conc.get("syringe_um") is None or cell_volume is None:
        return {
            "status": DomainStatus.DIAGNOSTIC.value,
            "diagnostics": [diagnostic("$.materials.concentration", "INPUT_INCOMPLETE", "concentration or cell volume missing")],
            "candidates": [],
        }
    volumes = np.array([item["volume_ul"] for item in included], dtype=float)
    y = np.array([item["net_uj"] for item in included], dtype=float)
    x = {"volumes": volumes, "cell0": conc["cell_um"], "syringe": conc["syringe_um"], "cell_volume": cell_volume}
    candidates = []
    for model in MODEL_ORDER:
        fitted = _fit_model(model, x, y, budget)
        params = fitted["params"]
        k = len(params)
        sse = fitted["sse"]
        aic = len(y) * math.log(max(sse / len(y), EPSILON)) + 2 * k
        pred = _model_prediction(model, params, x)
        residuals = (y - pred).tolist()
        candidates.append(
            {
                "model": model,
                "params": params.tolist(),
                "sse": sse,
                "aic": aic,
                "residuals": residuals,
                "included_injections": [item["injection_id"] for item in included],
                "discarded_injections": sorted(discarded),
            }
        )
    best_aic = min(item["aic"] for item in candidates)
    for item in candidates:
        item["equally_good"] = item["aic"] <= best_aic + 2.0
    candidates.sort(key=lambda item: (item["aic"], MODEL_ORDER.index(item["model"])))
    return {"status": DomainStatus.OK.value, "diagnostics": [], "candidates": candidates}
