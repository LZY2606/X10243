from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .units import Diagnostic, measurement, parse_number, parse_temperature


@dataclass
class ParsedPower:
    points: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    status: str = "ok"
    budget_exhausted: bool = False


@dataclass
class ParsedInjections:
    injections: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    status: str = "ok"


@dataclass
class ParsedConcentration:
    records: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    status: str = "ok"


@dataclass
class ParsedConditions:
    conditions: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    status: str = "ok"


@dataclass
class ParsedSeries:
    points: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    status: str = "ok"


def _rows(text: str, filename: str) -> tuple[list[dict[str, str]], Diagnostic | None]:
    try:
        reader = csv.DictReader(io.StringIO(text), strict=True)
        rows = list(reader)
    except csv.Error as exc:
        return [], Diagnostic("format_corrupt", f"CSV cannot be parsed: {exc}", {"file": filename})
    if reader.fieldnames is None:
        return [], Diagnostic("format_corrupt", "CSV has no header.", {"file": filename})
    return rows, None


def parse_power_series(text: str, *, filename: str = "power.csv", budget: int = 200000) -> ParsedPower:
    rows, diag = _rows(text, filename)
    if diag:
        return ParsedPower([], [diag], "corrupt")
    required = {"time", "time_unit", "power", "power_unit"}
    if not rows or not required.issubset(rows[0].keys()):
        return ParsedPower([], [Diagnostic("format_corrupt", "Power CSV requires time,time_unit,power,power_unit.", {"file": filename})], "corrupt")
    points = []
    diagnostics = []
    previous_time = None
    for index, row in enumerate(rows):
        location = {"file": filename, "row": index + 2, "kind": "power_sample"}
        time_unit = (row.get("time_unit") or "").strip()
        power_unit = (row.get("power_unit") or "").strip()
        time_m, time_d = measurement("time", row.get("time"), time_unit, location=location | {"field": "time"})
        power_m, power_d = measurement("power", row.get("power"), power_unit, location=location | {"field": "power"})
        if time_d:
            diagnostics.append(time_d)
        if power_d:
            diagnostics.append(power_d)
        if time_m and power_m:
            if previous_time is not None and time_m.normalized_value <= previous_time:
                diagnostics.append(Diagnostic("semantic_impossible", "Power samples must have strictly increasing timestamps.", location | {"field": "time"}, row.get("time")))
            else:
                points.append({"index": index, "raw_time": time_m.raw_text, "raw_power": power_m.raw_text, "time_s": time_m.normalized_value, "power_ucal_s": power_m.normalized_value, "time_unit": time_m.unit, "power_unit": power_m.unit, "coordinate_basis": "elapsed_seconds_from_series_start"})
                previous_time = time_m.normalized_value
        if len(points) > budget:
            return ParsedPower(points[:budget], diagnostics, "budget_exhausted", True)
    return ParsedPower(points, diagnostics, "corrupt" if diagnostics else "ok")


def parse_injection_schedule(text: str, *, filename: str = "injections.csv") -> ParsedInjections:
    rows, diag = _rows(text, filename)
    if diag:
        return ParsedInjections([], [diag], "corrupt")
    required = {"injection", "time", "time_unit", "volume", "volume_unit", "blank_ref"}
    if not rows or not required.issubset(rows[0].keys()):
        return ParsedInjections([], [Diagnostic("format_corrupt", "Injection CSV requires injection,time,time_unit,volume,volume_unit,blank_ref.", {"file": filename})], "corrupt")
    injections = []
    diagnostics = []
    for index, row in enumerate(rows):
        location = {"file": filename, "row": index + 2, "kind": "injection"}
        ident = (row.get("injection") or "").strip()
        time_m, time_d = measurement("time", row.get("time"), (row.get("time_unit") or "").strip(), location=location | {"field": "time"})
        volume_m, volume_d = measurement("volume", row.get("volume"), (row.get("volume_unit") or "").strip(), location=location | {"field": "volume"})
        if time_d:
            diagnostics.append(time_d)
        if volume_d:
            diagnostics.append(volume_d)
        if time_m and volume_m:
            injections.append({"id": ident, "sequence": index + 1, "raw_time": time_m.raw_text, "raw_volume": volume_m.raw_text, "time_s": time_m.normalized_value, "volume_ul": volume_m.normalized_value, "time_unit": time_m.unit, "volume_unit": volume_m.unit, "blank_ref": (row.get("blank_ref") or "").strip() or None, "coordinate_basis": "elapsed_seconds_from_series_start"})
    ordered = sorted(injections, key=lambda item: item["time_s"])
    if [item["sequence"] for item in ordered] != [item["sequence"] for item in injections]:
        diagnostics.append(Diagnostic("semantic_impossible", "Injection rows are not ordered by time.", {"file": filename, "field": "time"}))
    return ParsedInjections(injections, diagnostics, "corrupt" if diagnostics else "ok")


def parse_concentrations(text: str, *, filename: str = "concentrations.csv") -> ParsedConcentration:
    rows, diag = _rows(text, filename)
    if diag:
        return ParsedConcentration([], [diag], "corrupt")
    required = {"role", "value", "unit"}
    if not rows or not required.issubset(rows[0].keys()):
        return ParsedConcentration([], [Diagnostic("format_corrupt", "Concentration CSV requires role,value,unit.", {"file": filename})], "corrupt")
    records = []
    diagnostics = []
    for index, row in enumerate(rows):
        location = {"file": filename, "row": index + 2, "role": row.get("role"), "field": "value"}
        measure, measure_diag = measurement("concentration", row.get("value"), (row.get("unit") or "").strip(), location=location)
        if measure_diag:
            diagnostics.append(measure_diag)
        if measure:
            records.append({"role": (row.get("role") or "").strip(), "raw_value": measure.raw_text, "raw_unit": measure.unit, "value_um": measure.normalized_value, "unit": "µM"})
    return ParsedConcentration(records, diagnostics, "corrupt" if diagnostics else "ok")


def parse_baseline(text: str, *, filename: str = "baseline.csv") -> ParsedSeries:
    rows, diag = _rows(text, filename)
    if diag:
        return ParsedSeries([], [diag], "corrupt")
    required = {"start_time", "start_time_unit", "end_time", "end_time_unit", "power", "power_unit"}
    if not rows or not required.issubset(rows[0].keys()):
        return ParsedSeries([], [Diagnostic("format_corrupt", "Baseline CSV requires start_time,start_time_unit,end_time,end_time_unit,power,power_unit.", {"file": filename})], "corrupt")
    points = []
    diagnostics = []
    for index, row in enumerate(rows):
        location = {"file": filename, "row": index + 2, "kind": "baseline"}
        start, sd = measurement("time", row.get("start_time"), (row.get("start_time_unit") or "").strip(), location=location | {"field": "start_time"})
        end, ed = measurement("time", row.get("end_time"), (row.get("end_time_unit") or "").strip(), location=location | {"field": "end_time"})
        power, pd_ = measurement("power", row.get("power"), (row.get("power_unit") or "").strip(), location=location | {"field": "power"})
        for item in (sd, ed, pd_):
            if item:
                diagnostics.append(item)
        if start and end and power:
            if end.normalized_value <= start.normalized_value:
                diagnostics.append(Diagnostic("semantic_impossible", "Baseline segment end must be after start.", location))
            else:
                points.append({"start_s": start.normalized_value, "end_s": end.normalized_value, "power_ucal_s": power.normalized_value, "raw": dict(row)})
    return ParsedSeries(points, diagnostics, "corrupt" if diagnostics else "ok")


def parse_conditions(text: str, *, filename: str = "conditions.json") -> ParsedConditions:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedConditions({}, [Diagnostic("format_corrupt", f"JSON cannot be parsed: {exc}", {"file": filename})], "corrupt")
    if not isinstance(raw, dict):
        return ParsedConditions({}, [Diagnostic("format_corrupt", "Conditions must be a JSON object.", {"file": filename})], "corrupt")
    diagnostics = []
    pool, pool_diag = measurement("volume", raw.get("cell_volume_value"), str(raw.get("cell_volume_unit", "")), location={"file": filename, "field": "cell_volume_value"})
    if pool_diag:
        diagnostics.append(pool_diag)
    temperature, temp_diag = parse_temperature(raw.get("temperature_value"), str(raw.get("temperature_unit", "")), location={"file": filename, "field": "temperature_value"})
    if temp_diag:
        diagnostics.append(temp_diag)
    if not diagnostics and pool and temperature:
        conditions = {"cell_volume_l": pool.normalized_value / 1000000.0, "cell_volume_ul": pool.normalized_value, "cell_volume_raw": pool.raw_text, "cell_volume_raw_unit": pool.unit, "temperature_k": temperature.normalized_value, "temperature_display_value": temperature.value, "temperature_display_unit": temperature.coordinate, "temperature_raw": temperature.raw_text, "temperature_raw_unit": temperature.unit, "raw": raw}
        return ParsedConditions(conditions, [], "ok")
    return ParsedConditions({}, diagnostics, "corrupt")


def diagnostics_to_dict(items: list[Diagnostic]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
