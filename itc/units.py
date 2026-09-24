"""物理量：原始表示原样保留，派生层才做规范化。

Measure 同时保存：
* value/unit            —— 导入时声明的值与单位（原始表示）
* raw                   —— 文本原文（未解析时也保留）
* canonical_value       —— 规范化数值（能转换时才有）
* canonical_unit        —— 规范单位
* diagnostics           —— 无法转换的原因（值留在原位置）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import Diagnostic

CANONICAL = {
    "time": "s",
    "power": "µcal/s",
    "energy": "cal",
    "volume": "L",
    "concentration": "M",
    "temperature": "K",
}

# 单位 -> (物理量, 相对规范单位的乘数, 偏移)
_UNITS: dict[str, tuple[str, float, float]] = {
    # time
    "s": ("time", 1.0, 0.0), "sec": ("time", 1.0, 0.0), "second": ("time", 1.0, 0.0),
    "seconds": ("time", 1.0, 0.0),
    "ms": ("time", 1e-3, 0.0), "min": ("time", 60.0, 0.0),
    "minute": ("time", 60.0, 0.0), "minutes": ("time", 60.0, 0.0),
    # power（规范 µcal/s）
    "ucal/s": ("power", 1.0, 0.0), "µcal/s": ("power", 1.0, 0.0),
    "cal/s": ("power", 1e6, 0.0), "mcal/s": ("power", 1e3, 0.0),
    "uw": ("power", 0.239006, 0.0), "µw": ("power", 0.239006, 0.0),
    "mw": ("power", 239.006, 0.0), "w": ("power", 239006.0, 0.0),
    # energy（规范 cal）
    "cal": ("energy", 1.0, 0.0), "kcal": ("energy", 1e3, 0.0),
    "j": ("energy", 0.239006, 0.0), "kj": ("energy", 239.006, 0.0),
    "mj": ("energy", 0.239006e-3, 0.0), "uj": ("energy", 0.239006e-6, 0.0),
    "µj": ("energy", 0.239006e-6, 0.0),
    # volume（规范 L）
    "l": ("volume", 1.0, 0.0), "ml": ("volume", 1e-3, 0.0),
    "ul": ("volume", 1e-6, 0.0), "µl": ("volume", 1e-6, 0.0),
    # concentration（规范 M = mol/L）
    "m": ("concentration", 1.0, 0.0), "mm": ("concentration", 1e-3, 0.0),
    "um": ("concentration", 1e-6, 0.0), "µm": ("concentration", 1e-6, 0.0),
    "nm": ("concentration", 1e-9, 0.0),
    # temperature（规范 K，带偏移）
    "k": ("temperature", 1.0, 0.0),
    "c": ("temperature", 1.0, 273.15),
    "degc": ("temperature", 1.0, 273.15),
    "°c": ("temperature", 1.0, 273.15),
}

_NUM = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")


def normalize_unit(unit: str) -> str:
    return unit.strip().replace("μ", "µ").lower()


@dataclass
class Measure:
    quantity: str
    value: float | None
    unit: str
    raw: str
    canonical_value: float | None = None
    canonical_unit: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    loc: str = ""

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "value": self.value,
            "unit": self.unit,
            "raw": self.raw,
            "canonical_value": self.canonical_value,
            "canonical_unit": self.canonical_unit,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "loc": self.loc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Measure":
        m = cls(
            quantity=d["quantity"],
            value=d.get("value"),
            unit=d.get("unit", ""),
            raw=d.get("raw", ""),
            canonical_value=d.get("canonical_value"),
            canonical_unit=d.get("canonical_unit"),
            loc=d.get("loc", ""),
        )
        m.diagnostics = [
            Diagnostic(x["code"], x["message"], x.get("loc", ""), x.get("value"))
            for x in d.get("diagnostics", [])
        ]
        return m

    @property
    def ok(self) -> bool:
        return self.canonical_value is not None and not self.diagnostics


def parse_measure(raw: str, quantity: str, loc: str = "") -> Measure:
    """解析带单位的量；无法转换时原值保留并附诊断。"""
    text = (raw if isinstance(raw, str) else str(raw)).strip()
    m = Measure(quantity=quantity, value=None, unit="", raw=text, loc=loc)
    match = _NUM.match(text)
    if not match:
        m.diagnostics.append(Diagnostic("not_a_number", f"无法解析数值: {text!r}", loc, text))
        return m
    try:
        value = float(match.group(1))
    except ValueError:
        m.diagnostics.append(Diagnostic("not_a_number", f"无法解析数值: {text!r}", loc, text))
        return m
    unit_raw = match.group(2)
    m.value = value
    m.unit = unit_raw
    key = normalize_unit(unit_raw) if unit_raw else CANONICAL[quantity]
    info = _UNITS.get(key)
    if info is None:
        m.diagnostics.append(
            Diagnostic("unknown_unit", f"未知单位 {unit_raw!r}（物理量 {quantity}）", f"{loc}.unit", unit_raw)
        )
        return m
    qkind, mult, offset = info
    if qkind != quantity:
        m.diagnostics.append(
            Diagnostic("unit_quantity_mismatch",
                       f"单位 {unit_raw!r} 属于 {qkind}，期望 {quantity}", f"{loc}.unit", unit_raw)
        )
        return m
    m.canonical_value = value * mult + offset
    m.canonical_unit = CANONICAL[quantity]
    return m


def required(raw: str, quantity: str, loc: str) -> Measure:
    m = parse_measure(raw, quantity, loc)
    if m.value is None:
        from .errors import FormatCorrupt

        raise FormatCorrupt(f"{loc} 缺少可解析数值", m.diagnostics)
    return m
