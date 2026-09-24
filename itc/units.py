"""量纲、单位换算与带单位原始值。

原则：所有领域量携带单位或坐标基准；导入时保留原始表示，
派生层才做规范化。无法转换的值留在原位置并附诊断。
"""

import math
import re
from dataclasses import dataclass, field

from .errors import Diagnostic

# 规范单位（SI 派生，热化学常用）
CANONICAL = {
    "time": "s",
    "volume": "L",
    "concentration": "mol/L",
    "temperature": "K",
    "power": "µcal/s",   # ITC 惯例：µcal/s
    "energy": "µcal",
    "heat_per_mole": "kcal/mol",
}

_TIME = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
         "min": 60.0, "minute": 60.0, "minutes": 60.0,
         "ms": 1e-3, "millisecond": 1e-3}
_VOLUME = {"l": 1.0, "liter": 1.0, "liters": 1.0,
           "ml": 1e-3, "milliliter": 1e-3,
           "µl": 1e-6, "ul": 1e-6, "uL": 1e-6, "µL": 1e-6,
           "microliter": 1e-6, "microliters": 1e-6}
_CONC = {"mol/l": 1.0, "m": 1.0,  # M 与 mol/L 等价（体积摩尔浓度）
         "mmol/l": 1e-3, "mm": 1e-3,
         "umol/l": 1e-6, "µmol/l": 1e-6, "um": 1e-6, "µm": 1e-6,
         "nmol/l": 1e-9, "nm": 1e-9}
_POWER = {"µcal/s": 1.0, "ucal/s": 1.0, "microcal/s": 1.0,
          "cal/s": 1e6, "mcal/s": 1e3,
          "µw": 1.0 / 4.184, "uw": 1.0 / 4.184,
          "mw": 1000.0 / 4.184, "w": 1e6 / 4.184}
_ENERGY = {"µcal": 1.0, "ucal": 1.0, "microcal": 1.0,
           "cal": 1e6, "kcal": 1e9, "mcal": 1e3,
           "µj": 1.0 / 4.184, "uj": 1.0 / 4.184, "microjoule": 1.0 / 4.184,
           "mj": 1000.0 / 4.184, "j": 1e6 / 4.184}
_HEAT_MOL = {"kcal/mol": 1.0, "cal/mol": 1e-3,
             "kj/mol": 4.184, "j/mol": 4.184e-3,
             "kjoule/mol": 4.184}

# 温度只接受 K / °C / C，且为偏移换算
_TEMP = {"k": (1.0, 0.0), "kelvin": (1.0, 0.0),
         "c": (1.0, 273.15), "°c": (1.0, 273.15), "degc": (1.0, 273.15),
         "celsius": (1.0, 273.15)}

_TABLE = {
    "time": _TIME,
    "volume": _VOLUME,
    "concentration": _CONC,
    "power": _POWER,
    "energy": _ENERGY,
    "heat_per_mole": _HEAT_MOL,
}

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Quantity:
    """带单位的原始值：raw 永不丢失，value 仅在可换算时存在。"""

    raw: str
    dimension: str
    value: float | None = None          # 规范单位下的值
    unit: str | None = None             # 原始单位（规整后的写法）
    canonical_unit: str | None = None
    converted: bool = False
    diagnostic: dict | None = None

    def to_dict(self):
        return {
            "raw": self.raw,
            "dimension": self.dimension,
            "value": self.value,
            "unit": self.unit,
            "canonicalUnit": self.canonical_unit,
            "converted": self.converted,
            "diagnostic": self.diagnostic,
        }


def _norm_unit(u: str) -> str:
    return u.strip().replace("μ", "µ")


def parse_quantity(raw, dimension, *, path=None):
    """解析 '0.2 mM' 这类表示；无法转换时 Quantity 留在原位置并附诊断。"""
    q = Quantity(raw=raw, dimension=dimension, canonical_unit=CANONICAL.get(dimension))
    if raw is None:
        q.diagnostic = Diagnostic(
            "MISSING_VALUE", "值缺失：不得用空值或默认值伪装成功",
            severity="error", path=path).to_dict()
        return q
    text = str(raw).strip()
    if not text:
        q.diagnostic = Diagnostic(
            "MISSING_VALUE", "空字符串：不得用空值或默认值伪装成功",
            severity="error", path=path).to_dict()
        return q
    parts = text.split(None, 1)
    num_text = parts[0]
    unit_text = parts[1] if len(parts) == 2 else ""
    if not _NUM_RE.match(num_text):
        q.diagnostic = Diagnostic(
            "UNPARSEABLE_NUMBER", f"无法解析数字 {text!r}",
            severity="error", path=path).to_dict()
        return q
    try:
        num = float(num_text)
    except ValueError:
        q.diagnostic = Diagnostic(
            "UNPARSEABLE_NUMBER", f"无法解析数字 {text!r}",
            severity="error", path=path).to_dict()
        return q
    if not math.isfinite(num):
        q.diagnostic = Diagnostic(
            "NON_FINITE", f"非有限数值 {text!r}", severity="error", path=path).to_dict()
        return q

    if dimension == "temperature":
        key = _norm_unit(unit_text).lower() if unit_text else "k"
        if key not in _TEMP:
            q.unit = unit_text or None
            q.diagnostic = Diagnostic(
                "UNKNOWN_UNIT",
                f"未知温度单位 {unit_text!r}，原始表示保留在原位置",
                severity="error", path=path).to_dict()
            return q
        factor, offset = _TEMP[key]
        q.unit = unit_text or "K"
        q.value = num * factor + offset
        q.converted = True
        return q

    table = _TABLE[dimension]
    if not unit_text:
        q.diagnostic = Diagnostic(
            "MISSING_UNIT",
            f"{text!r} 缺少单位；领域量必须携带单位，值保留在原位置",
            severity="error", path=path).to_dict()
        return q
    key = _norm_unit(unit_text)
    if key not in table:
        q.unit = unit_text
        q.diagnostic = Diagnostic(
            "UNKNOWN_UNIT",
            f"未知{dimension}单位 {unit_text!r}，原始表示保留在原位置",
            severity="error", path=path).to_dict()
        return q
    q.unit = key
    q.value = num * table[key]
    q.converted = True
    return q


def require(q: Quantity, *, what):
    """取规范值；不可转换时抛语义无解（带位置）。"""
    if not q.converted or q.value is None:
        d = q.diagnostic or {}
        from .errors import SemanticNoSolution
        raise SemanticNoSolution(
            f"{what} 无法规范化：{d.get('message', q.raw)!r}",
            code="UNCONVERTIBLE_VALUE", path=d.get("path"))
    return q.value
