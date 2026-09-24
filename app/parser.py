from __future__ import annotations

import csv
import io

from .errors import FormatCorrupted
from .quantities import quantity


CURVE_HEADERS = {
    "power": {"time": "time", "value": "power"},
    "baseline": {"time": "time", "value": "baseline"},
    "blank": {"time": "time", "value": "blank"},
}


def _read_csv(text):
    if not isinstance(text, str):
        raise FormatCorrupted("CSV 载荷必须是字符串", path="data.csv")
    stream = io.StringIO(text)
    reader = csv.DictReader(stream)
    required = {"time", "time_unit", "value_unit"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise FormatCorrupted(
            "CSV 缺少必要表头 time,time_unit,value_unit",
            path="data.csv",
            details={"found": reader.fieldnames or []},
        )
    value_col = reader.fieldnames[3] if len(reader.fieldnames) > 3 else "value"
    rows = []
    for line, row in enumerate(reader, start=2):
        if None in row:
            raise FormatCorrupted("CSV 存在多余未命名列", path=f"data.csv:{line}")
        clean = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
        clean["_line"] = line
        clean["_value_column"] = value_col
        rows.append(clean)
    if not rows:
        raise FormatCorrupted("CSV 没有数据行", path="data.csv")
    return rows


def _read_schedule_csv(text):
    if not isinstance(text, str):
        raise FormatCorrupted("CSV 载荷必须是字符串", path="schedule.csv")
    stream = io.StringIO(text)
    reader = csv.DictReader(stream)
    required = {"time", "time_unit", "volume", "volume_unit"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise FormatCorrupted(
            "注射 CSV 缺少必要表头 time,time_unit,volume,volume_unit",
            path="schedule.csv",
            details={"found": reader.fieldnames or []},
        )
    rows = []
    for line, row in enumerate(reader, start=2):
        clean = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
        clean["_line"] = line
        rows.append(clean)
    if not rows:
        raise FormatCorrupted("注射 CSV 没有数据行", path="schedule.csv")
    return rows


def _parse_curve(kind, payload):
    records = payload.get("records")
    if records is None:
        records = _read_csv(payload.get("csv", ""))
    diagnostics = list(payload.get("diagnostics", []))
    points = []
    for ordinal, raw in enumerate(records):
        line = raw.get("_line", ordinal + 1)
        base_path = f"record[{ordinal}]"
        raw_time = raw.get("time", "")
        raw_value = raw.get(raw.get("_value_column", CURVE_HEADERS[kind]["value"]), raw.get("value", ""))
        time_unit = raw.get("time_unit", payload.get("time_unit", "s"))
        value_unit = raw.get("value_unit", payload.get("value_unit", {"power": "uW"}.get(kind, "uW")))
        missing = str(raw_value).strip() == "" or str(raw_time).strip() == ""
        point = {
            "ordinal": ordinal,
            "source_line": line,
            "raw": {"time": raw_time, "value": raw_value, "time_unit": time_unit, "value_unit": value_unit},
            "available": not missing,
        }
        if not missing:
            t = quantity(raw_time, time_unit, "time", path=f"{base_path}.time", diagnostics=diagnostics)
            v = quantity(raw_value, value_unit, "power", path=f"{base_path}.value", diagnostics=diagnostics)
            point["time"] = t
            point["value"] = v
            if t.get("available") is False or v.get("available") is False:
                point["available"] = False
        else:
            diagnostics.append(
                {
                    "code": "missing_sample",
                    "message": "采样值留空，作为数据缺口保留",
                    "path": base_path,
                    "source_line": line,
                }
            )
        points.append(point)
    return {
        "material_id": payload["material_id"],
        "kind": kind,
        "name": payload.get("name", kind),
        "coordinate_basis": {"time_origin": "experiment_t0", "time_unit": "s", "value_unit": "uW"},
        "raw_representation": payload.get("raw_representation", {"format": "structured"}),
        "points": points,
        "diagnostics": diagnostics,
    }


def _parse_schedule(payload):
    records = payload.get("records")
    if records is None:
        records = _read_schedule_csv(payload.get("csv", ""))
    diagnostics = list(payload.get("diagnostics", []))
    injections = []
    for ordinal, raw in enumerate(records):
        path = f"injections[{ordinal}]"
        idx_raw = raw.get("index", str(ordinal + 1))
        time_q = quantity(raw.get("time"), raw.get("time_unit", "s"), "time", path=f"{path}.time", diagnostics=diagnostics)
        vol_q = quantity(raw.get("volume"), raw.get("volume_unit", "uL"), "volume", path=f"{path}.volume", diagnostics=diagnostics)
        injections.append(
            {
                "ordinal": ordinal,
                "index": int(float(idx_raw)) if str(idx_raw).strip() else None,
                "raw": {"index": idx_raw, "time": raw.get("time"), "time_unit": raw.get("time_unit", "s"),
                        "volume": raw.get("volume"), "volume_unit": raw.get("volume_unit", "uL")},
                "time": time_q,
                "volume": vol_q,
                "available": time_q.get("available", True) and vol_q.get("available", True) and str(idx_raw).strip() != "",
            }
        )
    return {
        "material_id": payload["material_id"],
        "kind": "schedule",
        "name": payload.get("name", "注射时序"),
        "coordinate_basis": {"time_origin": "experiment_t0", "time_unit": "s", "volume_unit": "uL"},
        "raw_representation": payload.get("raw_representation", {"format": "structured"}),
        "injections": injections,
        "diagnostics": diagnostics,
    }


def _parse_blank_mapping(payload):
    records = payload.get("records", [])
    diagnostics = list(payload.get("diagnostics", []))
    mappings = []
    for ordinal, raw in enumerate(records):
        injection_index = raw.get("injection_index", raw.get("index"))
        blank_index = raw.get("blank_index")
        if injection_index in (None, "") or blank_index in (None, ""):
            diagnostics.append({
                "code": "blank_pair_incomplete",
                "message": "空白对应缺少注射或空白编号",
                "path": f"blank_mapping[{ordinal}]",
            })
        mappings.append({
            "ordinal": ordinal,
            "injection_index": int(injection_index) if str(injection_index).strip() else None,
            "blank_index": int(blank_index) if str(blank_index).strip() else None,
            "raw": dict(raw),
            "available": injection_index not in (None, "") and blank_index not in (None, ""),
        })
    return {
        "material_id": payload["material_id"],
        "kind": "blank_mapping",
        "name": payload.get("name", "空白对应"),
        "mappings": mappings,
        "raw_representation": payload.get("raw_representation", {"format": "structured"}),
        "coordinate_basis": {"rule": "explicit_blank_injection_index"},
        "diagnostics": diagnostics,
    }


def _parse_concentration(payload):
    diagnostics = list(payload.get("diagnostics", []))
    raw = payload.get("raw", payload)
    cell = quantity(raw.get("cell_value"), raw.get("cell_unit", "uM"), "concentration", path="cell", diagnostics=diagnostics)
    syringe = quantity(raw.get("syringe_value"), raw.get("syringe_unit", "uM"), "concentration", path="syringe", diagnostics=diagnostics)
    return {
        "material_id": payload["material_id"],
        "kind": "concentration",
        "name": payload.get("name", "浓度"),
        "cell": cell,
        "syringe": syringe,
        "raw": {"cell_value": raw.get("cell_value"), "cell_unit": raw.get("cell_unit", "uM"),
                "syringe_value": raw.get("syringe_value"), "syringe_unit": raw.get("syringe_unit", "uM")},
        "coordinate_basis": {"concentration_unit": "M"},
        "diagnostics": diagnostics,
    }


def _parse_conditions(payload):
    diagnostics = list(payload.get("diagnostics", []))
    raw = payload.get("raw", payload)
    temp = quantity(raw.get("temperature_value"), raw.get("temperature_unit", "K"), "temperature", path="temperature", diagnostics=diagnostics)
    cell_volume = quantity(raw.get("cell_volume_value"), raw.get("cell_volume_unit", "uL"), "volume", path="cell_volume", diagnostics=diagnostics)
    return {
        "material_id": payload["material_id"],
        "kind": "conditions",
        "name": payload.get("name", "实验条件"),
        "temperature": temp,
        "cell_volume": cell_volume,
        "raw": raw,
        "coordinate_basis": {"temperature_unit": "K", "volume_unit": "uL"},
        "diagnostics": diagnostics,
    }


def parse_material(payload):
    if "material_id" not in payload or "kind" not in payload:
        raise FormatCorrupted("材料缺少 material_id 或 kind", path="material")
    kind = payload["kind"]
    if kind in ("power", "baseline", "blank"):
        return _parse_curve(kind, payload)
    if kind == "schedule":
        return _parse_schedule(payload)
    if kind == "concentration":
        return _parse_concentration(payload)
    if kind == "conditions":
        return _parse_conditions(payload)
    if kind == "blank_mapping":
        return _parse_blank_mapping(payload)
    raise FormatCorrupted(f"未知材料类型 {kind}", path="kind")
