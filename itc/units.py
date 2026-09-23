"""单位、坐标与时间基准。

原则：材料层保留导入时的原始表示（字符串 + 声明单位）；只有派生层才做规范化。
无法转换的值留在原位置并附诊断，绝不静默丢弃。
"""
from __future__ import annotations

import math
from .models import Diagnostic, FormatCorrupted

# 时间：统一换算到秒
TIME_TO_S = {
    "s": 1.0, "sec": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
}
# 功率：统一换算到 µW。1 cal = 4.184 J
POWER_TO_UW = {
    "uW": 1.0, "µW": 1.0,
    "mW": 1e3,
    "W": 1e6,
    "ucal/s": 4.184,          # µcal/s → µW（×4.184）
    "µcal/s": 4.184,
    "mcal/s": 4184.0,
}
# 能量：统一换算到 µJ
ENERGY_TO_UJ = {
    "uJ": 1.0, "µJ": 1.0,
    "mJ": 1e3, "J": 1e6,
    "ucal": 4.184, "µcal": 4.184,
    "mcal": 4184.0, "cal": 4.184e6,
}
# 浓度：统一换算到 mM
CONC_TO_MM = {
    "mm": 1.0, "mmol/l": 1.0, "mm": 1.0,
    "um": 1e-3, "µm": 1e-3, "umol/l": 1e-3,
    "m": 1e3, "mol/l": 1e3,
    "nm": 1e-6, "nmol/l": 1e-6,
}
# 体积：统一换算到 µL
VOL_TO_UL = {
    "ul": 1.0, "µl": 1.0,
    "ml": 1e3, "l": 1e6,
}
TEMP_SCALES = ("degC", "degF", "K")

TABLES = {
    "time": (TIME_TO_S, "s"),
    "power": (POWER_TO_UW, "uW"),
    "energy": (ENERGY_TO_UJ, "uJ"),
    "concentration": (CONC_TO_MM, "mM"),
    "volume": (VOL_TO_UL, "uL"),
}


def canonical(domain: str, raw_value: str | float | int, raw_unit: str,
              pos: str = "") -> tuple[float, Diagnostic | None]:
    """把 (原始值, 原始单位) 换算到该量纲的规范单位。

    返回 (规范值, 诊断)。无法转换时抛 FormatCorrupted（未知单位属格式/语义问题，
    由材料解析器捕获并留在原位）。
    """
    table, canon_unit = TABLES[domain]
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise FormatCorrupted(
            f"数值无法解析: {raw_value!r}",
            [Diagnostic("non_numeric", f"数值无法解析: {raw_value!r}", pos,
                        value=None if raw_value is None else str(raw_value))],
        )
    key = raw_unit.strip().lower()
    if key not in table:
        raise FormatCorrupted(
            f"未知单位 {raw_unit!r}（量纲 {domain}）",
            [Diagnostic("unknown_unit", f"未知单位 {raw_unit!r}", pos,
                        value=str(raw_unit))],
        )
    if not math.isfinite(value):
        raise FormatCorrupted(
            f"数值非有限: {raw_value!r}",
            [Diagnostic("non_finite", "数值非有限", pos, value=str(raw_value))],
        )
    return value * table[key], None


def convert_temperature(raw_value: str | float, raw_unit: str,
                        pos: str = "") -> tuple[float, Diagnostic | None]:
    value = float(raw_value)
    if not math.isfinite(value):
        raise FormatCorrupted("温度非有限",
                              [Diagnostic("non_finite", "温度非有限", pos)])
    u = raw_unit.strip()
    if u in ("degC", "C", "°C"):
        return value, None
    if u in ("K",):
        return value - 273.15, None
    if u in ("degF", "F", "°F"):
        return (value - 32.0) * 5.0 / 9.0, None
    raise FormatCorrupted(
        f"未知温标 {raw_unit!r}",
        [Diagnostic("unknown_unit", f"未知温标 {raw_unit!r}", pos,
                    value=str(raw_unit))],
    )


def is_missing_token(token: str | None) -> bool:
    """材料里的缺口/未测标记（区别于格式损坏）。"""
    if token is None:
        return True
    return token.strip() in {"", "NA", "N/A", "null", "NaN", "nan", "--", "缺失"}
