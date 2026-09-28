"""单位 / 坐标基准处理。

领域量同时携带：
- raw:   导入时的原始表示（字符串，逐字保留）
- value: 解析后的数值（可能为 None）
- unit:  规范化后的内部单位（未知单位时为 None）
- diagnostics: 无法转换时留在原位置的诊断

绝不因为画不出图而丢弃原值。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import Diagnostic

# 内部 SI / 常用规范单位
# 时间: s, 功率: µW, 能量: µJ/cal, 体积: L(µL/mL), 浓度: M, 温度: K
_PREFIX = {
    "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "m": 1e-3, "c": 1e-2, "": 1.0,
}
_TIME = {"s": 1.0, "min": 60.0, "sec": 1.0, "minute": 60.0, "minutes": 60.0}
_POWER = {"W": 1.0}  # 配合词头
_ENERGY = {"J": 1.0, "cal": 4.184, "Cal": 4.184}
_VOLUME = {"L": 1.0}  # 配合词头
_CONC = {"M": 1.0, "mol/L": 1.0}  # 配合词头
_TEMP_CANON = {"K": "K", "C": "degC", "°C": "degC", "degC": "degC",
               "celsius": "degC"}

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_QTY_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*"
                     r"(.*?)\s*$")


@dataclass
class Quantity:
    raw: str
    value: float | None = None
    unit: str | None = None
    canonical_unit: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "canonical_unit": self.canonical_unit,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def split_unit(u: str) -> tuple[str, str]:
    u = u.strip()
    if not u:
        return "", ""
    # 多字母单位优先（min, mol/L）
    if u in _TIME:
        return "", u
    if u in _CONC:
        return "", u
    if len(u) >= 2 and u[0] in _PREFIX and u[1:] in ("W", "J", "L", "M"):
        return u[0], u[1:]
    if u in ("W", "J", "L", "M"):
        return "", u
    return "", u


def parse_quantity(raw, dimension: str, location: str = "") -> Quantity:
    """解析带单位的量并规范化到内部单位。

    dimension: time|power|energy|volume|concentration|temperature
    无法转换时 value=None 且附带诊断，raw 永远保留。
    """
    text = "" if raw is None else str(raw).strip()
    q = Quantity(raw=text)
    if text == "":
        q.diagnostics.append(Diagnostic(
            "missing_value", "缺少数值，禁止用默认值伪装成功", location))
        return q
    m = _QTY_RE.match(text)
    if not m:
        q.diagnostics.append(Diagnostic(
            "unparseable_quantity", f"无法解析数量表示 {text!r}", location,
            value=text))
        return q
    num_s, unit_s = m.group(1), m.group(2).strip()
    if not _NUM_RE.match(num_s):
        q.diagnostics.append(Diagnostic(
            "unparseable_number", f"无法解析数值 {num_s!r}", location,
            value=num_s))
        return q
    q.value = float(num_s)
    q.unit = unit_s

    if dimension == "temperature":
        canon = _TEMP_CANON.get(unit_s)
        if canon is None:
            q.diagnostics.append(Diagnostic(
                "unknown_unit", f"未知温度单位 {unit_s!r}，原值保留",
                location, value=text))
            q.value = None
            return q
        q.canonical_unit = canon
        if canon == "degC":
            q.value = float(num_s) + 273.15
        return q

    # 带词头单位
    prefix, base = split_unit(unit_s)
    if dimension == "time":
        factor = _TIME.get(unit_s)
        canon = "s"
    elif dimension == "power":
        factor = _PREFIX.get(prefix) if base == "W" else None
        canon = "W"
    elif dimension == "energy":
        if base == "J" and prefix in _PREFIX:
            factor = _PREFIX[prefix]
        elif unit_s in _ENERGY:
            factor = _ENERGY[unit_s]
            # cal 归一到 J
        else:
            factor = None
        canon = "J"
    elif dimension == "volume":
        factor = _PREFIX.get(prefix) if base == "L" else None
        canon = "L"
    elif dimension == "concentration":
        if base == "M" and prefix in _PREFIX:
            factor = _PREFIX[prefix]
        elif unit_s == "mol/L":
            factor = 1.0
        else:
            factor = None
        canon = "M"
    else:
        factor = None
        canon = None

    if factor is None:
        q.diagnostics.append(Diagnostic(
            "unknown_unit", f"未知或不匹配的单位 {unit_s!r}（维度 {dimension}），原值保留",
            location, value=text))
        q.value = None
        q.canonical_unit = None
        return q
    q.canonical_unit = canon
    q.value = float(num_s) * factor
    return q


def require(q: Quantity, location: str = "") -> float:
    """要求可转换，否则抛语义错误（带原位置诊断）。"""
    if q.value is None or q.canonical_unit is None:
        from .errors import SemanticInvalid
        raise SemanticInvalid(
            f"位置 {location or q.raw}: 缺少可规范化的领域量",
            q.diagnostics or [Diagnostic(
                "unconvertible", "领域量无法转换且无默认值", location)])
    return q.value
