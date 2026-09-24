"""量与单位。

所有领域量以 :class:`Quantity` 携带原始表示、解析值与单位族；导入时保留原始表示，
派生层才规范化（规范化 SI 值集中在 :mod:`itc.models` / integration 使用）。
无法转换的值留在原位置并附诊断（见 :class:`itc.ingest.MaterialRecord`），
绝不为画图而丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass

# 单位族 -> 规范化单位 -> 相对规范化单位的乘数
FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
             "ms": 1e-3, "min": 60.0, "h": 3600.0},
    "power": {"uw": 1.0, "µw": 1.0, "uW": 1.0, "µW": 1.0, "mcal/s": 4.184,
              "cal/s": 4.184e6, "w": 1e6, "W": 1e6},
    "energy": {"cal": 1.0, "kcal": 1e3, "mcal": 1e-3,
               "j": 1 / 4.184, "J": 1 / 4.184, "mj": 1e-3 / 4.184,
               "kJ": 1000 / 4.184},
    "volume": {"ul": 1e-6, "µl": 1e-6, "uL": 1e-6, "µL": 1e-6,
               "ml": 1e-3, "mL": 1e-3, "l": 1.0, "L": 1.0},
    "concentration": {"um": 1e-6, "µm": 1e-6, "uM": 1e-6, "µM": 1e-6,
                      "mm": 1e-3, "mM": 1e-3, "moll": 1.0, "mol/L": 1.0,
                      "nm": 1e-9, "nM": 1e-9},
    "temperature": {"c": 273.15, "C": 273.15, "°c": 273.15, "°C": 273.15,
                    "k": 0.0, "K": 0.0},
    "inverse_concentration": {"m-1": 1.0, "M^-1": 1.0, "/m": 1.0, "/M": 1.0,
                              "mm-1": 1e-3, "mM^-1": 1e-3, "/mm": 1e-3,
                              "um-1": 1e-6, "uM^-1": 1e-6},
    "dimensionless": {"": 1.0, "-": 1.0},
}

# 规范化单位（派生层统一使用）
CANONICAL = {
    "time": "s",
    "power": "µW",
    "energy": "cal",
    "volume": "L",
    "concentration": "mol/L",
    "temperature": "K",
    "inverse_concentration": "M^-1",
    "dimensionless": "",
}

TEMP_FAMILIES = {"temperature"}


class UnitError(ValueError):
    """单位族未知或与已有量纲冲突。"""


@dataclass(frozen=True)
class Quantity:
    """带单位的领域量；``raw`` 是导入时的原始表示（字符串/数字原样保留）。"""

    value: float
    unit: str
    family: str
    raw: str | int | float | None = None

    @property
    def canonical_value(self) -> float:
        f = FACTORS[self.family]
        if self.family in TEMP_FAMILIES:
            return self.value + f[self.unit]
        return self.value * f[self.unit]

    @property
    def canonical_unit(self) -> str:
        return CANONICAL[self.family]

    def to(self, unit: str) -> float:
        f = FACTORS[self.family]
        if unit not in f:
            raise UnitError(f"unknown unit {unit!r} for family {self.family}")
        if self.family in TEMP_FAMILIES:
            return self.canonical_value - f[unit]
        return self.canonical_value / f[unit]

    def render(self) -> str:
        if self.raw is not None and str(self.raw) != "":
            return f"{self.raw} {self.unit}".strip()
        return f"{self.value:g} {self.unit}".strip()


def parse_quantity(text: object, family: str, *, default_unit: str | None = None) -> Quantity:
    """解析 ``"0.3 mM"`` / ``25`` 这类表示；空串与不可解析值抛 :class:`ValueError`。

    解析失败由调用方转成带位置的诊断，绝不用默认值伪装成功。
    """
    if text is None:
        raise ValueError("empty value")
    raw = text
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        value, unit = float(text), default_unit or list(FACTORS[family])[0]
    else:
        s = str(text).strip()
        if not s:
            raise ValueError("empty value")
        parts = s.split()
        if len(parts) == 1:
            token = parts[0]
            # 允许 "0.3mM" 这种粘连写法
            cut = _split_num_unit(token)
            if cut is None:
                if default_unit is None:
                    raise ValueError(f"missing unit in {s!r}")
                value, unit = float(token), default_unit
            else:
                num, unit = cut
                value = float(num)
        else:
            value = float(parts[0])
            unit = parts[1]
    if family not in FACTORS:
        raise UnitError(f"unknown family {family!r}")
    if unit not in FACTORS[family]:
        raise UnitError(f"unit {unit!r} not compatible with {family}")
    return Quantity(value=value, unit=unit, family=family, raw=raw if isinstance(raw, str) else None)


def _split_num_unit(token: str) -> tuple[str, str] | None:
    i = 0
    if token[:1] in "+-":
        i = 1
    seen_dot = seen_e = False
    while i < len(token):
        ch = token[i]
        if ch.isdigit():
            pass
        elif ch == "." and not seen_dot:
            seen_dot = True
        elif ch in "eE" and not seen_e and i > 0:
            seen_e = True
            if i + 1 < len(token) and token[i + 1] in "+-":
                i += 1
        else:
            break
        i += 1
    if i == 0 or i >= len(token):
        return None
    num, unit = token[:i], token[i:]
    try:
        float(num)
    except ValueError:
        return None
    return num, unit
