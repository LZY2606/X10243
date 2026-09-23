"""单位与坐标基准。

规则: 导入时保留原始表示 (value/unit/basis); 规范化只发生在派生层。
无法转换的值留在原位置并附诊断, 不得用空值或默认值伪装成功。
"""
from __future__ import annotations

# dimension -> {unit: 到基准单位的倍率}
UNIT_TABLE = {
    "time": {"s": 1.0, "min": 60.0},
    "power": {"µJ/s": 1.0, "uJ/s": 1.0, "µcal/s": 4.184, "ucal/s": 4.184, "nW": 1e-3},
    "volume": {"µL": 1.0, "uL": 1.0, "mL": 1e3},
    "concentration": {"mM": 1.0, "µM": 1e-3, "uM": 1e-3, "M": 1e3},
    "temperature": {"K": 1.0},
    "heat": {"µJ": 1.0, "uJ": 1.0, "µcal": 4.184, "ucal": 4.184},
}
BASE_UNIT = {
    "time": "s", "power": "µJ/s", "volume": "µL",
    "concentration": "mM", "temperature": "K", "heat": "µJ",
}
CELSIUS = ("°C", "degC", "C")


def qv(value, unit, basis=None):
    """构造携带单位/坐标基准的量。"""
    q = {"value": value, "unit": unit}
    if basis:
        q["basis"] = basis
    return q


def diag(code, message, position=None, severity="error"):
    d = {"code": code, "message": message, "severity": severity}
    if position is not None:
        d["position"] = position
    return d


def normalize(q, dimension, position=None):
    """把量 q 规范化到基准单位。

    返回 (value, diag): 成功时 diag 为 None; 失败时 value 为 None 且 diag 带位置。
    原始表示不被修改。
    """
    if not isinstance(q, dict) or "value" not in q or "unit" not in q:
        return None, diag("malformed_quantity", f"量缺少 value/unit: {q!r}", position)
    unit = q["unit"]
    value = q["value"]
    if dimension == "temperature" and unit in CELSIUS:
        return value + 273.15, None
    table = UNIT_TABLE.get(dimension, {})
    if unit not in table:
        return None, diag(
            "unknown_unit",
            f"单位 {unit!r} 无法转换到 {BASE_UNIT.get(dimension, '?')} (维度 {dimension})",
            position,
        )
    if not isinstance(value, (int, float)):
        return None, diag("non_numeric", f"数值不可解析: {value!r}", position)
    return value * table[unit], None
