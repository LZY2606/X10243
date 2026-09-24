"""领域量：导入保留原始表示，派生层才规范化；无法转换时留在原位置并附诊断。"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 线性单位到 SI 的换算
_SI_FACTOR = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "h": 3600.0},
    "power": {"uw": 1.0, "µw": 1.0, "mw": 1000.0, "w": 1e6},
    "volume": {"ul": 1.0, "µl": 1.0, "ml": 1000.0, "l": 1e6},
    "concentration": {"um": 1.0, "µm": 1.0, "mm": 1000.0, "m": 1e6, "nm": 1e-3},
    "energy": {"uj": 1.0, "µj": 1.0, "mj": 1000.0, "j": 1e6, "cal": 4.184e6, "kcal": 4.184e9},
}
_SI_UNIT = {"time": "s", "power": "µW", "volume": "µL", "concentration": "µM", "energy": "µJ",
            "temperature": "K"}
_TEMP_UNITS = {"c": "C", "°c": "C", "celsius": "C", "k": "K", "kelvin": "K",
               "f": "F", "°f": "F", "fahrenheit": "F"}


@dataclass
class Quantity:
    raw: str
    value: float | None
    si_value: float | None
    si_unit: str | None
    kind: str | None
    unit: str | None
    diagnostic: str | None = None

    def as_dict(self) -> dict:
        return {
            "raw": self.raw, "value": self.value, "si_value": self.si_value,
            "si_unit": self.si_unit, "kind": self.kind, "unit": self.unit,
            "diagnostic": self.diagnostic,
        }


_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _split(raw: str) -> tuple[str | None, str | None]:
    m = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s\[\]]+)?\s*$", raw)
    if not m:
        return None, None
    return m.group(1), (m.group(2) or "").strip()


def parse_quantity(raw: str, kind: str) -> Quantity:
    """kind in time/power/volume/concentration/energy/temperature。"""
    num, unit = _split(str(raw))
    if num is None:
        return Quantity(raw=str(raw), value=None, si_value=None, si_unit=None,
                        kind=kind, unit=None,
                        diagnostic=f"unconvertible_value:{kind}:无法解析数值与单位 '{raw}'")
    value = float(num)
    if kind == "temperature":
        u = _TEMP_UNITS.get(unit.lower()) if unit else None
        if u == "C":
            si = value + 273.15
        elif u == "K":
            si = value
        elif u == "F":
            si = (value - 32.0) * 5.0 / 9.0 + 273.15
        else:
            return Quantity(raw, value, None, None, kind, unit,
                            diagnostic=f"unconvertible_value:temperature:未知温度单位 '{unit}'")
        return Quantity(raw, value, si, "K", kind, u)
    table = _SI_FACTOR.get(kind, {})
    factor = table.get(unit.lower()) if unit else None
    if factor is None:
        return Quantity(raw, value, None, None, kind, unit,
                        diagnostic=f"unconvertible_value:{kind}:未知单位 '{unit}'")
    return Quantity(raw, value, value * factor, _SI_UNIT[kind], kind, unit)
