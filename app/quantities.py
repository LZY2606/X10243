from __future__ import annotations

import math


CONVERSIONS = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "minute": 60.0},
    "power": {"uW": 1.0, "µW": 1.0, "mcal/s": 4184.0},
    "heat": {"uJ": 1.0, "µJ": 1.0, "cal": 4.184, "mcal": 4184.0},
    "volume": {"uL": 1.0, "µL": 1.0, "mL": 1000.0, "L": 1_000_000.0},
    "concentration": {"uM": 1e-6, "µM": 1e-6, "mM": 1e-3, "M": 1.0},
    "temperature": {"K": "K", "C": "C", "°C": "C"},
}

SI_UNITS = {
    "time": "s",
    "power": "uW",
    "heat": "uJ",
    "volume": "uL",
    "concentration": "M",
}


def _float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("not finite")
    return value


def convert_quantity(raw_value, raw_unit, kind, *, path=None):
    try:
        value = _float(raw_value)
    except (TypeError, ValueError):
        from .errors import SemanticNoSolution

        raise SemanticNoSolution(
            "无法转换数值，原始表示已保留",
            path=path,
            details={"raw_value": raw_value, "raw_unit": raw_unit, "kind": kind},
        )
    unit = str(raw_unit).strip()
    table = CONVERSIONS[kind]
    if unit not in table:
        from .errors import SemanticNoSolution

        raise SemanticNoSolution(
            "单位没有已知坐标基准，原始表示已保留",
            path=path,
            details={"raw_value": raw_value, "raw_unit": raw_unit, "kind": kind},
        )
    if kind == "temperature":
        if table[unit] == "K":
            normalized = value
        else:
            normalized = value + 273.15
        return {"raw": {"value": raw_value, "unit": raw_unit}, "value": normalized, "unit": "K"}
    factor = table[unit]
    return {
        "raw": {"value": raw_value, "unit": raw_unit},
        "value": value * factor,
        "unit": SI_UNITS[kind],
        "factor": factor,
    }


def quantity(raw_value, raw_unit, kind, *, path=None, diagnostics=None):
    try:
        return convert_quantity(raw_value, raw_unit, kind, path=path)
    except Exception as exc:  # diagnostics collection keeps import going
        if diagnostics is None:
            raise
        if hasattr(exc, "to_diagnostic"):
            diagnostics.append(exc.to_diagnostic())
        else:
            diagnostics.append({"code": "quantity_error", "message": str(exc), "path": path})
        return {"raw": {"value": raw_value, "unit": raw_unit}, "available": False, "kind": kind}


def display_quantity(q, digits=6):
    if not isinstance(q, dict):
        return q
    if q.get("available") is False:
        raw = q.get("raw", {})
        return f"{raw.get('value')} {raw.get('unit', '')}".strip()
    value = q.get("value")
    unit = q.get("unit", "")
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g} {unit}".strip()
    return f"{value} {unit}".strip()

