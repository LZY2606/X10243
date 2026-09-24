"""领域量：携带原始表示，仅在派生层规范化。

无法转换的值保留在原位置并附诊断，绝不为画图而丢弃。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import FormatCorrupt, SemanticImpossible

# 规范单位 -> 目标 SI 单位的乘子（温度另行处理）
_FACTORS: dict[str, dict[str, tuple[str, float]]] = {
    "power": {
        "uw": ("W", 1e-6), "µw": ("W", 1e-6), "μw": ("W", 1e-6),
        "mw": ("W", 1e-3), "w": ("W", 1.0),
        "cal/s": ("W", 4.184),
    },
    "volume": {
        "ul": ("L", 1e-6), "µl": ("L", 1e-6), "μl": ("L", 1e-6),
        "ml": ("L", 1e-3), "l": ("L", 1.0),
    },
    "concentration": {
        "nm": ("M", 1e-9), "um": ("M", 1e-6), "µm": ("M", 1e-6), "μm": ("M", 1e-6),
        "mm": ("M", 1e-3), "m": ("M", 1.0),
    },
    "time": {
        "s": ("s", 1.0), "sec": ("s", 1.0), "min": ("s", 60.0),
    },
    "heat": {
        "ucal": ("J", 4.184e-6), "µcal": ("J", 4.184e-6), "μcal": ("J", 4.184e-6),
        "mcal": ("J", 4.184e-3), "cal": ("J", 4.184),
        "kcal": ("J", 4184.0),
        "uj": ("J", 1e-6), "µj": ("J", 1e-6), "μj": ("J", 1e-6),
        "mj": ("J", 1e-3), "j": ("J", 1.0), "kj": ("J", 1e3),
    },
    "heat_per_mole": {
        "cal/mol": ("J/mol", 4.184), "kcal/mol": ("J/mol", 4184.0),
        "j/mol": ("J/mol", 1.0), "kj/mol": ("J/mol", 1e3),
    },
    "ka": {
        "m^-1": ("M^-1", 1.0), "m-1": ("M^-1", 1.0),
        "/m": ("M^-1", 1.0), "l/mol": ("M^-1", 1e-3),
    },
}

_CANONICAL = {family: next(iter(units)) for family, units in (
    (f, list(m.items())) for f, m in _FACTORS.items()
) for _ in [(None)]}


def canonical_unit(family: str) -> str:
    return _FACTORS[family] and _FACTORS[family][[*_FACTORS[family]][0]][0]


@dataclass
class Measurement:
    """一个领域量：raw 永远保留；canonical 只在可换算时存在。"""

    raw: str
    family: str
    value: float | None = None
    unit: str | None = None
    canonical: float | None = None
    canonical_unit: str | None = None
    diagnostic: str | None = None

    def as_dict(self) -> dict:
        return {
            "raw": self.raw, "family": self.family, "value": self.value,
            "unit": self.unit, "canonical": self.canonical,
            "canonical_unit": self.canonical_unit, "diagnostic": self.diagnostic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Measurement":
        return cls(**{k: d.get(k) for k in
                      ("raw", "family", "value", "unit", "canonical",
                       "canonical_unit", "diagnostic")})


def parse_measurement(raw: object, family: str, *, location: str = "") -> Measurement:
    """解析带单位的量。

    - 数字部分缺失/损坏 -> FormatCorrupt（带位置）
    - 单位无法识别/换算 -> 值留原位 + diagnostic（SemanticImpossible 不抛出，
      因为材料导入必须保留原始表示；需要硬性成功的派生步骤再据此报错）
    """
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    text = raw.strip()
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$", text)
    if not m:
        raise FormatCorrupt(
            f"无法解析数值与单位: {text!r}", locations=[location])
    try:
        value = float(m.group(1))
    except ValueError:
        raise FormatCorrupt(
            f"数值损坏: {text!r}", locations=[location])
    unit = m.group(2).strip()
    key = unit.lower().replace(" ", "")
    meas = Measurement(raw=text, family=family, value=value, unit=unit)
    if family == "temperature":
        low = unit.replace(" ", "").lower()
        if low in ("c", "°c", "oc", "degc", "celsius"):
            meas.canonical = value + 273.15
            meas.canonical_unit = "K"
        elif low in ("k", "kelvin"):
            meas.canonical = value
            meas.canonical_unit = "K"
        else:
            meas.diagnostic = f"未知温度单位 {unit!r}，原值保留未规范化"
        return meas
    table = _FACTORS.get(family)
    if table is None:
        meas.diagnostic = f"未知量族 {family!r}"
        return meas
    if key in table:
        cu, fac = table[key]
        meas.canonical = value * fac
        meas.canonical_unit = cu
    else:
        meas.diagnostic = (
            f"{family} 的单位 {unit!r} 不可换算，值保留在原位置；"
            "派生计算不使用该值")
    return meas


def require(meas: Measurement, *, what: str, where: str) -> float:
    """派生层需要规范值时调用：不可换算即语义无解，错误带位置。"""
    if meas.canonical is None:
        raise SemanticImpossible(
            f"{what} 无法参与计算：{meas.diagnostic or '缺少规范值'}",
            locations=[where])
    return meas.canonical
