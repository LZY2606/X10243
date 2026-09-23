"""单位 / 坐标基准层。

导入层保留原始表示（raw），只在派生层做规范化。无法转换的值不丢弃，
保留在原位置并附诊断。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .diag import Diagnostic, SemanticUnresolvable

# 线性单位换算，值为“1 该单位 = 多少基准单位”
_BASE = {
    "power": {
        "ucal/s": 1.0, "µcal/s": 1.0, "cal/s": 1.0e6,
        "uw": 1.0 / 4.184, "µw": 1.0 / 4.184,
        "mw": 1000.0 / 4.184, "w": 1.0e6 / 4.184, "j/s": 1.0e6 / 4.184,
    },
    "volume": {"ul": 1.0, "µl": 1.0, "ml": 1000.0, "l": 1.0e6},
    "concentration": {"um": 1.0, "µm": 1.0, "mm": 1000.0, "m": 1.0e6},
    "energy": {"ucal": 1.0, "µcal": 1.0, "cal": 1.0e6, "j": 4.184e6, "mj": 4184.0},
    "molar_energy": {
        "cal/mol": 1.0, "kcal/mol": 1000.0,
        "j/mol": 1.0 / 4.184, "kj/mol": 1000.0 / 4.184,
    },
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0, "minute": 60.0},
}
_KIND_BASE_UNIT = {
    "power": "ucal/s",
    "volume": "ul",
    "concentration": "um",
    "energy": "ucal",
    "molar_energy": "cal/mol",
    "time": "s",
}
_NUM = re.compile(r"^[ \t]*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.[0-9]+(?:[eE][+-]?\d+)?)[ \t]*(.*)$")


def _norm_unit(u: str) -> str:
    return u.strip().replace("μ", "µ").lower()


@dataclass
class Quantity:
    value: float | None
    unit: str
    raw: str
    kind: str
    base_value: float | None = None
    diagnostic: Diagnostic | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "value": self.value, "unit": self.unit, "raw": self.raw,
            "kind": self.kind, "base_value": self.base_value,
            "base_unit": _KIND_BASE_UNIT.get(self.kind),
        }
        if self.diagnostic is not None:
            d["diagnostic"] = self.diagnostic.as_dict()
        return d


def quantity_from_dict(d: dict[str, Any]) -> Quantity:
    q = Quantity(
        value=d.get("value"), unit=d.get("unit", ""), raw=d.get("raw", ""),
        kind=d.get("kind", ""), base_value=d.get("base_value"),
    )
    if d.get("diagnostic"):
        dd = d["diagnostic"]
        q.diagnostic = Diagnostic(dd["code"], dd["message"], dd.get("path", ""), dd.get("severity", "error"))
    return q


def parse_quantity(raw: Any, kind: str, path: str = "") -> Quantity:
    """解析带单位的量；单位未知时保留原值与诊断，不抛异常、不丢值。"""
    raw_s = str(raw).strip()
    m = _NUM.match(raw_s)
    if not m:
        q = Quantity(value=None, unit="", raw=raw_s, kind=kind)
        q.diagnostic = Diagnostic("quantity_not_numeric", f"无法解析为数值+单位: {raw_s!r}", path)
        return q
    try:
        value = float(m.group(1))
    except ValueError:
        value = None
    unit = m.group(2).strip()
    q = Quantity(value=value, unit=unit, raw=raw_s, kind=kind)
    if value is not None and math.isnan(value):
        q.value = None
        q.diagnostic = Diagnostic("quantity_nan", "数值为 NaN", path)
        return q
    table = _BASE.get(kind, {})
    nu = _norm_unit(unit)
    if value is not None and nu in table:
        q.base_value = value * table[nu]
    elif value is not None:
        q.diagnostic = Diagnostic(
            "unit_unconvertible",
            f"{kind} 量的单位 {unit!r} 无法换算到 {_KIND_BASE_UNIT[kind]}，原始表示保留",
            path,
        )
    return q


def convert(value: float, unit: str, kind: str) -> float:
    nu = _norm_unit(unit)
    table = _BASE.get(kind)
    if table is None or nu not in table:
        raise SemanticUnresolvable(
            f"未知的 {kind} 单位: {unit!r}",
            [Diagnostic("unit_unknown", f"单位 {unit!r} 无换算规则", "unit")],
        )
    return value * table[nu]


def to_unit(base_value: float, unit: str, kind: str) -> float:
    nu = _norm_unit(unit)
    table = _BASE.get(kind)
    if table is None or nu not in table:
        raise SemanticUnresolvable(
            f"未知的 {kind} 单位: {unit!r}",
            [Diagnostic("unit_unknown", f"单位 {unit!r} 无换算规则", "unit")],
        )
    return base_value / table[nu]


def parse_temperature(raw: Any, path: str = "") -> Quantity:
    """温度特殊（带偏移）：基准为 K。"""
    q = parse_quantity(raw, "temperature", path)
    if q.value is None or q.diagnostic is not None:
        return q
    u = _norm_unit(q.unit)
    if u in ("c", "°c", "degc", "celsius"):
        q.base_value = q.value + 273.15
        q.diagnostic = None
    elif u in ("k", "kelvin"):
        q.base_value = q.value
    else:
        q.diagnostic = Diagnostic(
            "unit_unconvertible", f"温度单位 {q.unit!r} 无法换算到 K，原始表示保留", path
        )
    return q


def display(base_value: float | None, kind: str, digits: int = 6) -> str | None:
    if base_value is None:
        return None
    unit = _KIND_BASE_UNIT[kind]
    text = f"{base_value:.{digits}g}"
    return f"{text} {unit}"
