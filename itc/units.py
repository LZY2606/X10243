"""单位 / 坐标基准：原始表示保留，派生层才规范化。

所有领域量在派生层携带 ``unit``。无法转换的值保留在原位置（Token），
附诊断，绝不静默丢弃或用默认值替代。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic, SemanticInvalid

# 线性单位换算到 SI 基准的乘数
LINEAR_FACTORS: dict[str, float] = {
    # 时间 -> s
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    # 体积 -> L
    "L": 1.0,
    "l": 1.0,
    "mL": 1e-3,
    "ml": 1e-3,
    "uL": 1e-6,
    "µL": 1e-6,
    "ul": 1e-6,
    # 浓度 -> M (mol/L)
    "M": 1.0,
    "mol/L": 1.0,
    "mM": 1e-3,
    "um": 1e-6,  # 小写歧义单位按 uM 处理的情况在 parse_quantity 显式判定
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    # 功率 -> cal/s（热量计算保留卡路里链路）
    "cal/s": 1.0,
    "ucal/s": 1e-6,
    "µcal/s": 1e-6,
    "uJ/s": 1.0 / 4.184,
    # 热量 -> cal
    "cal": 1.0,
    "ucal": 1e-6,
    "µcal": 1e-6,
    "kcal": 1e3,
    "J": 1.0 / 4.184,
    "uJ": 1e-6 / 4.184,
    "µJ": 1e-6 / 4.184,
    # 温度（线性部分不使用，温度走专用转换）
    # 无量纲
    "": 1.0,
    "1": 1.0,
    "ratio": 1.0,
}

TEMPERATURE_SCALES = {"K", "C", "°C", "degC", "F", "°F"}


@dataclass(frozen=True)
class Token:
    """无法转换/无法解释的值，留在原位置并附诊断。"""

    raw: Any
    unit: str | None
    reason: str
    location: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "unconvertible_token",
            "raw": self.raw,
            "unit": self.unit,
            "reason": self.reason,
            "location": self.location,
        }


def is_token(v: Any) -> bool:
    return isinstance(v, Token)


def convert_temperature(value: float, unit: str) -> float:
    """规范化为 K。"""
    if unit in ("K", "kelvin"):
        return float(value)
    if unit in ("C", "°C", "degC", "celsius"):
        return float(value) + 273.15
    if unit in ("F", "°F"):
        return (float(value) - 32.0) * 5.0 / 9.0 + 273.15
    raise KeyError(unit)


def parse_quantity(
    raw: Any,
    unit: str | None,
    target: str,
    location: str,
) -> float | Token:
    """把 (原始值, 原始单位) 规范化为目标单位；不可转换返回 Token。

    原始表示不被修改；失败只在原位置留 Token + 诊断。
    """
    if isinstance(raw, bool):
        return Token(raw, unit, "boolean_is_not_a_quantity", location)
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() in ("na", "n/a", "null", "missing", ""):
            return Token(raw, unit, "missing_or_na_token", location)
        try:
            value = float(text)
        except ValueError:
            return Token(raw, unit, "non_numeric_value", location)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return Token(raw, unit, "non_numeric_value", location)
    if not math.isfinite(value):
        return Token(raw, unit, "non_finite_value", location)
    if unit is None:
        return Token(raw, unit, "missing_unit", location)

    if target == "K" and (unit in TEMPERATURE_SCALES or unit in ("kelvin", "celsius")):
        try:
            return convert_temperature(value, unit)
        except KeyError:
            return Token(raw, unit, f"unknown_temperature_unit:{unit}", location)
    if unit in TEMPERATURE_SCALES or target == "K":
        return Token(raw, unit, f"unit_not_convertible:{unit}->{target}", location)

    if unit not in LINEAR_FACTORS or target not in LINEAR_FACTORS:
        return Token(raw, unit, f"unit_not_convertible:{unit}->{target}", location)
    return value * LINEAR_FACTORS[unit] / LINEAR_FACTORS[target]


def require_number(
    value: float | Token,
    field_name: str,
    diagnostics: list[Diagnostic],
) -> float | None:
    """收集 Token 诊断；合法返回数值。不用空值/默认值伪装成功。"""
    if isinstance(value, Token):
        diagnostics.append(
            Diagnostic(
                code="unconvertible_value",
                message=f"{field_name} 无法规范化：{value.reason}",
                location=value.location,
                value=value.raw,
            )
        )
        return None
    return value


def assert_semantics(
    diagnostics: list[Diagnostic],
    message: str = "输入不完整或语义无解",
) -> None:
    if diagnostics:
        raise SemanticInvalid(message, diagnostics)
