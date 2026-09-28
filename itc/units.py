"""单位、坐标/时间基准与原始表示保留。

导入层只做“登记”：记录原始字符串、声明单位与基准，不强制转换。
派生层通过 convert() 取规范单位数值；无法转换时抛 UNCONVERTIBLE_VALUE
诊断，原始值留在原位置（材料 content 不变），不删除、不替换为空值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .diagnostics import Diagnostic, ITCError, DiagnosticCode, Outcome

# 规范单位（SI 或领域惯例）：time s，power µcal/s，volume µL，
# concentration mM，temperature K，heat µcal。
CANONICAL: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0},
    "power": {"µcal/s": 1.0, "ucal/s": 1.0, "µW": 1 / 4.184, "uW": 1 / 4.184},
    "volume": {"µL": 1.0, "uL": 1.0, "mL": 1000.0, "L": 1e6},
    "concentration": {"mM": 1.0, "mmol/L": 1.0, "µM": 1e-3, "uM": 1e-3,
                      "M": 1e3, "nM": 1e-6},
    "temperature": {"K": 1.0, "°C": 1.0, "C": 1.0},
    "heat": {"µcal": 1.0, "ucal": 1.0, "cal": 1e6, "J": 1 / 4.184e-6},
}


def convert(value: Any, unit: str, quantity: str, *, position: dict[str, Any] | None = None):
    """把 (value, unit) 转换为该量纲的规范单位数值。

    温度的 °C 是仿射变换；其余是线性换算。
    无法转换（非数值/未知单位）时抛结构化诊断，原值仍保留在材料中。
    """
    pos = position or {}
    if unit not in CANONICAL.get(quantity, {}):
        raise ITCError(Diagnostic(
            DiagnosticCode.UNCONVERTIBLE_VALUE,
            f"未知单位 {unit!r}（量纲 {quantity}）",
            Outcome.SEMANTIC_UNSOLVABLE,
            [{**pos, "unit": unit, "raw_value": value, "problem": "unknown_unit"}],
            detail={"quantity": quantity},
        ))
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ITCError(Diagnostic(
            DiagnosticCode.UNCONVERTIBLE_VALUE,
            f"值 {value!r} 无法解析为数值（量纲 {quantity}）",
            Outcome.SEMANTIC_UNSOLVABLE,
            [{**pos, "unit": unit, "raw_value": value, "problem": "non_numeric"}],
            detail={"quantity": quantity},
        ))
    if isinstance(value, float) and (math.isnan(x) or math.isinf(x)):
        raise ITCError(Diagnostic(
            DiagnosticCode.UNCONVERTIBLE_VALUE,
            f"值 {value!r} 不是有限数（量纲 {quantity}）",
            Outcome.SEMANTIC_UNSOLVABLE,
            [{**pos, "unit": unit, "raw_value": value, "problem": "non_finite"}],
            detail={"quantity": quantity},
        ))
    if quantity == "temperature" and unit in ("°C", "C"):
        return x + 273.15
    return x * CANONICAL[quantity][unit]


@dataclass(frozen=True)
class Datum:
    """坐标/时间基准。"""

    time_origin: str = "t0_first_sample"   # 时间零点来源
    power_reference: str = "differential_reference_cell"
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"time_origin": self.time_origin,
                "power_reference": self.power_reference,
                "description": self.description}
