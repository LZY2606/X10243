"""领域量、单位与坐标基准。

导入时保留原始表示（original 字符串 + 声明单位），派生层才规范化为内部单位。
无法转换的值由 ``convert`` 抛出 ``Unconvertible``，留在原位置并附诊断。
"""
from __future__ import annotations

from dataclasses import dataclass

# 内部规范单位（所有计算在此基准上进行）
CANONICAL = {
    "power": "uW",
    "time": "s",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "energy": "uJ",
    "energy_per_mole": "cal/mol",
}

# 线性单位 -> (规范单位, 乘子)；temperature 单独处理
_LINEAR = {
    "power": {"uW": ("uW", 1.0), "mW": ("uW", 1000.0), "W": ("uW", 1e6)},
    "time": {"s": ("s", 1.0), "sec": ("s", 1.0), "min": ("s", 60.0),
             "m": ("s", 60.0), "ms": ("s", 1e-3)},
    "volume": {"uL": ("uL", 1.0), "ul": ("uL", 1.0), "mL": ("uL", 1000.0),
               "ml": ("uL", 1000.0), "L": ("uL", 1e6)},
    "concentration": {
        "mM": ("mM", 1.0), "mmol/L": ("mM", 1.0),
        "uM": ("mM", 1e-3), "um": ("mM", 1e-3), "µM": ("mM", 1e-3),
        "M": ("mM", 1000.0), "mol/L": ("mM", 1000.0),
        "nM": ("mM", 1e-6),
    },
    "energy": {"uJ": ("uJ", 1.0), "mJ": ("uJ", 1000.0), "J": ("uJ", 1e6),
               "cal": ("uJ", 4.184e6), "mcal": ("uJ", 4184.0)},
    "energy_per_mole": {
        "cal/mol": ("cal/mol", 1.0), "kcal/mol": ("cal/mol", 1000.0),
        "J/mol": ("cal/mol", 0.2390057361376673),
        "kJ/mol": ("cal/mol", 239.0057361376673),
    },
}

_KIND_FOR_UNIT: dict[str, str] = {}
for _kind, _units in _LINEAR.items():
    for _u in _units:
        _KIND_FOR_UNIT[_u] = _kind
_KIND_FOR_UNIT.update({"K": "temperature", "C": "temperature",
                       "degC": "temperature", "°C": "temperature"})


class Unconvertible(Exception):
    """三种失败之一：语义无解（值/单位无法解释）。携带定位位置。"""

    def __init__(self, message: str, position: "Position | None" = None):
        super().__init__(message)
        self.position = position


@dataclass(frozen=True)
class Position:
    """输入中的位置基准，诊断永远带位置，不用空值伪装。"""
    source: str
    kind: str = "file"
    row: int | None = None
    column: str | None = None

    def as_dict(self) -> dict:
        return {"source": self.source, "kind": self.kind,
                "row": self.row, "column": self.column}


def kind_of(unit: str) -> str | None:
    return _KIND_FOR_UNIT.get(unit.strip())


def convert(value: float, unit: str, declared_kind: str,
            position: "Position | None" = None) -> float:
    """把 (value, unit) 规范化到 declared_kind 的内部单位。

    单位与声明量纲不一致、或单位未知时抛 Unconvertible（语义无解），
    调用方负责把原始表示留在原位置并附诊断。
    """
    unit = unit.strip()
    actual_kind = _KIND_FOR_UNIT.get(unit)
    if actual_kind is None:
        raise Unconvertible(f"未知单位 {unit!r}（量纲 {declared_kind}）", position)
    if actual_kind != declared_kind:
        raise Unconvertible(
            f"单位 {unit!r} 属于 {actual_kind}，与声明量纲 {declared_kind} 不符",
            position)
    if declared_kind == "temperature":
        if unit in ("C", "degC", "°C"):
            return value + 273.15
        return value
    _, factor = _LINEAR[declared_kind][unit]
    return value * factor


def display(value: float, kind: str) -> str:
    unit = CANONICAL[kind]
    if kind == "temperature":
        return f"{value - 273.15:.6g} °C ({value:.6g} K)"
    return f"{value:.6g} {unit}"
