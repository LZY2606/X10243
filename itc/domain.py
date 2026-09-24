"""领域基础：单位、坐标基准、诊断与可区分失败类型。

原则：
- 原始表示在材料层不可变保留；规范化只发生在派生层。
- 无法转换的值留在原位置并附诊断，不用空值/默认值伪装成功。
- 资源预算耗尽、格式损坏、语义无解是三种不同的失败结果。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()


@dataclass
class Diagnostic:
    code: str
    location: str
    message: str
    severity: str = "error"  # error | warning

    def to_dict(self) -> dict:
        return asdict(self)


class ItcFailure(Exception):
    """计算/导入失败的基类；status 是机器可读的失败类别。"""

    status = "error"

    def __init__(self, message: str, location: str = "", diagnostics: list | None = None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.diagnostics = diagnostics or []

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message,
                "location": self.location, "diagnostics": self.diagnostics}


class FormatCorrupt(ItcFailure):
    status = "format_corrupt"


class BudgetExhausted(ItcFailure):
    status = "budget_exhausted"


class Unsolvable(ItcFailure):
    status = "unsolvable"


# --- 单位换算表（派生层规范化用；原始单位字符串保留在材料负载中） ---
POWER_TO_UJ_S = {"µJ/s": 1.0, "uJ/s": 1.0, "µcal/s": 4.184, "ucal/s": 4.184, "mcal/s": 4184.0}
TIME_TO_S = {"s": 1.0, "min": 60.0}
VOL_TO_UL = {"µL": 1.0, "uL": 1.0, "nL": 0.001, "mL": 1000.0}
CONC_TO_MM = {"mM": 1.0, "µM": 0.001, "uM": 0.001, "M": 1000.0}


def convert_scalar(value: Any, unit: str, table: dict, location: str, diagnostics: list) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        diagnostics.append(Diagnostic("value_not_numeric", location,
                                      f"值 {value!r} 无法解析为数值，原始表示保留在材料中").to_dict())
        return None
    factor = table.get(unit)
    if factor is None:
        diagnostics.append(Diagnostic("unit_unconvertible", location,
                                      f"单位 {unit!r} 无法转换，原始值保留在原位置").to_dict())
        return None
    return v * factor


def convert_array(values: Any, unit: str, table: dict, location: str, diagnostics: list) -> list | None:
    if not isinstance(values, list):
        diagnostics.append(Diagnostic("format_corrupt", location, "采样序列缺失或不是数组").to_dict())
        return None
    factor = table.get(unit)
    if factor is None:
        diagnostics.append(Diagnostic("unit_unconvertible", location,
                                      f"单位 {unit!r} 无法转换，序列保留原始表示").to_dict())
        return None
    out = []
    for i, v in enumerate(values):
        try:
            out.append(float(v) * factor)
        except (TypeError, ValueError):
            diagnostics.append(Diagnostic("value_not_numeric", f"{location}[{i}]",
                                          f"采样值 {v!r} 无法解析").to_dict())
            return None
    return out


def convert_temperature_k(value: Any, unit: str, location: str, diagnostics: list) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        diagnostics.append(Diagnostic("value_not_numeric", location, f"温度 {value!r} 无法解析").to_dict())
        return None
    if unit == "K":
        return v
    if unit in ("°C", "C", "degC"):
        return v + 273.15
    diagnostics.append(Diagnostic("unit_unconvertible", location, f"温度单位 {unit!r} 无法转换").to_dict())
    return None
