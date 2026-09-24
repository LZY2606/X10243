"""量与单位层。

铁律：
- 导入时保留原始表示（raw）；派生层才做规范化（canonical）。
- 所有领域量都携带单位或坐标基准。
- 无法转换的值留在原位置并附诊断，绝不为画图而丢弃。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .errors import diagnostic

# 各维度规范化单位（SI 风格的内部约定）
CANONICAL_UNITS: dict[str, str] = {
    "time": "s",
    "power": "uJ/s",      # µJ/s（µcal/s 在派生层换算）
    "energy": "uJ",       # µJ
    "volume": "L",        # 升
    "concentration": "M", # mol/L
    "temperature": "K",
    "heat_per_mole": "J/mol",
    "enthalpy": "J/mol",
    "rate": "1/M",
}

# 单位别名 -> (维度, 相对 canonical 的乘数, 加数)
_UNIT_TABLE: dict[str, tuple[str, float, float]] = {
    # time
    "s": ("time", 1.0, 0.0),
    "sec": ("time", 1.0, 0.0),
    "second": ("time", 1.0, 0.0),
    "seconds": ("time", 1.0, 0.0),
    "min": ("time", 60.0, 0.0),
    # power: µJ/s, µcal/s (1 cal = 4.184 J)
    "uj/s": ("power", 1.0, 0.0),
    "µj/s": ("power", 1.0, 0.0),
    "uj s-1": ("power", 1.0, 0.0),
    "ucal/s": ("power", 4.184, 0.0),
    "µcal/s": ("power", 4.184, 0.0),
    "cal/s": ("power", 4.184e6, 0.0),
    "j/s": ("power", 1.0e6, 0.0),
    # energy
    "uj": ("energy", 1.0, 0.0),
    "µj": ("energy", 1.0, 0.0),
    "ucal": ("energy", 4.184, 0.0),
    "µcal": ("energy", 4.184, 0.0),
    "cal": ("energy", 4.184e6, 0.0),
    "j": ("energy", 1.0e6, 0.0),
    "kcal/mol": ("heat_per_mole", 4184.0, 0.0),
    "cal/mol": ("heat_per_mole", 4.184, 0.0),
    "j/mol": ("heat_per_mole", 1.0, 0.0),
    "kj/mol": ("heat_per_mole", 1000.0, 0.0),
    # volume -> liter
    "l": ("volume", 1.0, 0.0),
    "ml": ("volume", 1.0e-3, 0.0),
    "ul": ("volume", 1.0e-6, 0.0),
    "µl": ("volume", 1.0e-6, 0.0),
    # concentration -> mol/L
    "m": ("concentration", 1.0, 0.0),
    "mm": ("concentration", 1.0e-3, 0.0),
    "um": ("concentration", 1.0e-6, 0.0),
    "µm": ("concentration", 1.0e-6, 0.0),
    "nm": ("concentration", 1.0e-9, 0.0),
    # affinity
    "1/m": ("rate", 1.0, 0.0),
    "m-1": ("rate", 1.0, 0.0),
    "m^-1": ("rate", 1.0, 0.0),
}

_TOKEN_RE = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


@dataclass(frozen=True)
class QToken:
    """带原始表示的量。canonical 为 None 表示无法转换（raw 保留原位）。"""

    raw: str
    value: float | None
    unit: str | None
    dimension: str | None
    canonical: float | None
    canonical_unit: str | None
    diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.canonical is not None and not any(
            d.get("severity") == "error" for d in self.diagnostics
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension,
            "canonical": self.canonical,
            "canonical_unit": self.canonical_unit,
            "diagnostics": list(self.diagnostics),
        }


def _normalize_unit(unit: str) -> str:
    u = unit.strip().lower()
    u = u.replace("μ", "µ")
    u = re.sub(r"\s+", " ", u)
    return u


def parse_quantity(raw: Any, expected_dimension: str, path: str) -> QToken:
    """解析单个带单位的量；温度单独走 affine 规则。"""
    raw_str = "" if raw is None else str(raw).strip()
    if raw_str == "":
        return QToken(
            raw=raw_str, value=None, unit=None, dimension=None,
            canonical=None, canonical_unit=None,
            diagnostics=(diagnostic(path, "空值：缺少数值，不使用默认值", raw=raw),),
        )
    match = _TOKEN_RE.match(raw_str)
    if not match:
        return QToken(
            raw=raw_str, value=None, unit=None, dimension=None,
            canonical=None, canonical_unit=None,
            diagnostics=(diagnostic(path, f"无法解析数量: {raw_str!r}", raw=raw_str),),
        )
    number = float(match.group(1))
    unit_raw = match.group(2)
    unit_key = _normalize_unit(unit_raw)

    # 温度（仿射，不能只靠乘数）
    if expected_dimension == "temperature" or unit_key in {"c", "°c", "k", "f", "°f"}:
        if unit_key in {"c", "°c"}:
            if expected_dimension not in {"temperature", "any"}:
                pass
            return QToken(raw_str, number, unit_raw, "temperature",
                          number + 273.15, "K")
        if unit_key == "k":
            return QToken(raw_str, number, unit_raw, "temperature", number, "K")
        if unit_key in {"f", "°f"}:
            return QToken(raw_str, number, unit_raw, "temperature",
                          (number - 32.0) * 5.0 / 9.0 + 273.15, "K")
        if expected_dimension == "temperature":
            return QToken(
                raw_str, number, unit_raw or None, "temperature", None, "K",
                diagnostics=(diagnostic(path, f"未知温度单位: {unit_raw!r}", raw=raw_str),),
            )

    spec = _UNIT_TABLE.get(unit_key)
    if spec is None:
        return QToken(
            raw_str, number, unit_raw or None, None, None,
            CANONICAL_UNITS.get(expected_dimension),
            diagnostics=(diagnostic(path, f"未知单位: {unit_raw!r}", raw=raw_str),),
        )
    dimension, scale, offset = spec
    if expected_dimension != "any" and dimension != expected_dimension:
        return QToken(
            raw_str, number, unit_raw, dimension, None,
            CANONICAL_UNITS.get(expected_dimension),
            diagnostics=(diagnostic(
                path,
                f"量纲不匹配：得到 {dimension}，期望 {expected_dimension}",
                raw=raw_str,
            ),),
        )
    return QToken(
        raw_str, number, unit_raw, dimension,
        number * scale + offset, CANONICAL_UNITS[dimension],
    )


def parse_unit_header(header: str) -> tuple[str, str | None]:
    """从列名 'time [s]' / 'power(uJ/s)' 拆出 (名称, 单位原文)。"""
    match = re.match(r"^\s*(.*?)\s*[\[\(]\s*(.*?)\s*[\]\)]\s*$", header)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return header.strip(), None
