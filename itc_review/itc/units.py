"""领域量的单位 / 坐标基准。

导入层保留原始表示（raw + raw_unit + 来源位置）；派生层才做规范化。
无法转换的值留在原位置并附 Diagnostic，绝不为画图丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 目标（规范）单位
TARGET = {
    "power": "uW",            # 微瓦
    "time": "s",              # 秒
    "volume": "uL",           # 微升
    "concentration": "uM",    # 微摩尔/升
    "temperature": "degC",    # 摄氏度
    "heat": "uJ",             # 微焦
    "enthalpy": "kJ/mol",
}

# 乘到目标单位的系数
CONVERSIONS: dict[tuple[str, str, str], float] = {
    ("power", "W", "uW"): 1e6,
    ("power", "mW", "uW"): 1e3,
    ("power", "uW", "uW"): 1.0,
    ("power", "µW", "uW"): 1.0,
    ("time", "s", "s"): 1.0,
    ("time", "sec", "s"): 1.0,
    ("time", "min", "s"): 60.0,
    ("time", "m", "s"): 60.0,
    ("volume", "uL", "uL"): 1.0,
    ("volume", "µL", "uL"): 1.0,
    ("volume", "mL", "uL"): 1e3,
    ("volume", "L", "uL"): 1e6,
    ("concentration", "uM", "uM"): 1.0,
    ("concentration", "µM", "uM"): 1.0,
    ("concentration", "mM", "uM"): 1e3,
    ("concentration", "nM", "uM"): 1e-3,
    ("concentration", "M", "uM"): 1e6,
    ("temperature", "degC", "degC"): 1.0,
    ("temperature", "C", "degC"): 1.0,
    ("temperature", "°C", "degC"): 1.0,
    ("temperature", "K", "degC"): 1.0,  # 特殊处理见 convert_temperature
}


@dataclass
class Diagnostic:
    """带位置的诊断。kind: format_corrupt | budget_exhausted | semantic_unresolved"""
    code: str
    kind: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "kind": self.kind,
            "message": self.message,
            "location": self.location,
            "raw": self.raw,
        }


def convert(quantity: str, value: float, unit: str) -> float:
    if (quantity, unit, TARGET[quantity]) not in CONVERSIONS:
        raise ValueError(f"无法把 {unit} 转换为 {TARGET[quantity]}")
    return value * CONVERSIONS[(quantity, unit, TARGET[quantity])]


def convert_temperature(value: float, unit: str) -> float:
    if unit in ("degC", "C", "°C"):
        return value
    if unit == "K":
        return value - 273.15
    raise ValueError(f"无法把 {unit} 转换为 degC")


def try_convert(
    quantity: str,
    raw: Any,
    unit: str | None,
    location: dict[str, Any],
    diagnostics: list[Diagnostic],
) -> float | None:
    """规范化单个标量；失败时追加诊断并返回 None（原值留在调用方记录中）。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        diagnostics.append(Diagnostic(
            "non_numeric_value", "format_corrupt",
            f"{quantity} 值无法解析为数字: {raw!r}", location, raw))
        return None
    if unit is None:
        diagnostics.append(Diagnostic(
            "missing_unit", "semantic_unresolved",
            f"{quantity} 缺少单位，无法规范化", location, raw))
        return None
    try:
        if quantity == "temperature":
            return convert_temperature(value, unit)
        return convert(quantity, value, unit)
    except ValueError:
        diagnostics.append(Diagnostic(
            "inconvertible_unit", "semantic_unresolved",
            f"{quantity} 单位 {unit!r} 无法转换为 {TARGET[quantity]}；原值保留",
            location, raw))
        return None
