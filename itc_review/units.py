"""单位 / 坐标基准与原始表示保留。

导入层只解析、不归一化；Measure.raw 保留源文件中的原始字符串。
派生层调用 to_canonical 得到规范值；无法转换的值留在原位置并附诊断。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import Diagnostic

# 到规范单位的线性换算 factor + offset：canonical = value * factor + offset
CONVERSIONS: dict[tuple[str, str], tuple[float, float]] = {
    # 时间 -> s
    ("time", "s"): (1.0, 0.0),
    ("time", "sec"): (1.0, 0.0),
    ("time", "min"): (60.0, 0.0),
    ("time", "h"): (3600.0, 0.0),
    # 功率 -> µW
    ("power", "uW"): (1.0, 0.0),
    ("power", "µW"): (1.0, 0.0),
    ("power", "cal/s"): (4.184 * 1e6, 0.0),
    ("power", "J/s"): (1e6, 0.0),
    # 体积 -> µL
    ("volume", "uL"): (1.0, 0.0),
    ("volume", "µL"): (1.0, 0.0),
    ("volume", "mL"): (1000.0, 0.0),
    ("volume", "L"): (1e6, 0.0),
    # 浓度 -> mM
    ("concentration", "mM"): (1.0, 0.0),
    ("concentration", "umol/L"): (0.001, 0.0),
    ("concentration", "µM"): (0.001, 0.0),
    ("concentration", "uM"): (0.001, 0.0),
    ("concentration", "mol/L"): (1000.0, 0.0),
    ("concentration", "M"): (1000.0, 0),
    # 温度 -> K
    ("temperature", "K"): (1.0, 0.0),
    ("temperature", "C"): (1.0, 273.15),
    ("temperature", "°C"): (1.0, 273.15),
    # 热 -> µcal
    ("heat", "ucal"): (1.0, 0.0),
    ("heat", "µcal"): (1.0, 0.0),
    ("heat", "cal"): (1e6, 0.0),
    ("heat", "J"): (0.2390057361376673 * 1e6, 0.0),
}

CANONICAL_UNIT = {
    "time": "s",
    "power": "µW",
    "volume": "µL",
    "concentration": "mM",
    "temperature": "K",
    "heat": "µcal",
}

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Measure:
    """带单位/基准的量。raw 永远保留导入时的原始表示。"""

    raw: Any
    unit: str | None = None
    kind: str | None = None
    value: float | None = None
    canonical: float | None = None
    canonical_unit: str | None = None
    valid: bool = True
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["diagnostics"] = [x.to_dict() for x in self.diagnostics]
        return d


def parse_measure(raw: Any, kind: str, unit: str | None = None, path: str = "") -> Measure:
    m = Measure(raw=raw, unit=unit, kind=kind)
    if isinstance(raw, (int, float)):
        text = repr(float(raw))
    else:
        text = str(raw).strip()
    m.raw = raw if isinstance(raw, str) else text
    # 形如 "298.15 K" / "5 µL" 时抽取数值与单位
    embedded_unit = unit
    parts = text.split()
    if _NUMBER_RE.match(parts[0]):
        num_text = parts[0]
        if embedded_unit is None and len(parts) > 1:
            embedded_unit = " ".join(parts[1:])
    else:
        num_text = None
        mt = re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text)
        if mt and mt.group(0) != "":
            num_text = mt.group(0)
            rest = text[mt.end():].strip()
            if embedded_unit is None and rest:
                embedded_unit = rest
    m.unit = embedded_unit
    if num_text is None:
        m.valid = False
        m.diagnostics.append(
            Diagnostic("UNPARSEABLE_VALUE", path, f"无法解析数值: {text!r}", text)
        )
        return m
    try:
        m.value = float(num_text)
    except ValueError:
        m.valid = False
        m.diagnostics.append(Diagnostic("UNPARSEABLE_VALUE", path, f"无法解析数值: {text!r}", text))
        return m
    if embedded_unit is None:
        m.valid = False
        m.diagnostics.append(
            Diagnostic("MISSING_UNIT", path, f"缺少单位基准（期望 {kind}）", text)
        )
        return m
    key = (kind, embedded_unit)
    if key not in CONVERSIONS:
        m.valid = False
        m.diagnostics.append(
            Diagnostic(
                "UNKNOWN_UNIT",
                path,
                f"未知/不可转换单位 {embedded_unit!r}（{kind} 规范单位 {CANONICAL_UNIT[kind]}）",
                text,
            )
        )
        return m
    factor, offset = CONVERSIONS[key]
    m.canonical = m.value * factor + offset
    m.canonical_unit = CANONICAL_UNIT[kind]
    return m


def require(measure: Measure, path: str) -> float:
    if not measure.valid or measure.canonical is None:
        codes = ", ".join(d.code for d in measure.diagnostics) or "INVALID"
        from .errors import SemanticUnresolvable

        raise SemanticUnresolvable(
            f"{path} 无有效规范值", measure.diagnostics
        )
    return measure.canonical
