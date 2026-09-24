"""量值与单位系统。

所有领域量以 {"raw": <原始表示>, "unit": <原始单位>, "value": <规范化数值或 None>,
"canonical_unit": ..., "diagnostics": [...]} 形式携带。
导入层保留 raw；无法转换时 value 为 None，原值留在原位置并附诊断，
绝不为画图而丢弃。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic, Location

# 以 SI 为规范单位的线性换算表：单位 -> (规范单位, 乘子, 偏移)
_TABLE: dict[str, dict[str, tuple[str, float, float]]] = {
    "power": {  # -> µcal/s
        "ucal/s": ("ucal/s", 1.0, 0.0),
        "µcal/s": ("ucal/s", 1.0, 0.0),
        "cal/s": ("ucal/s", 1e6, 0.0),
        "j/s": ("ucal/s", 1e6 / 4.184, 0.0),
        "w": ("ucal/s", 1e6 / 4.184, 0.0),
        "mw": ("ucal/s", 1e3 / 4.184, 0.0),
    },
    "energy": {  # -> cal
        "cal": ("cal", 1.0, 0.0),
        "kcal": ("cal", 1e3, 0.0),
        "j": ("cal", 1.0 / 4.184, 0.0),
        "mj": ("cal", 1e-3 / 4.184, 0.0),
        "uj": ("cal", 1e-6 / 4.184, 0.0),
        "µj": ("cal", 1e-6 / 4.184, 0.0),
    },
    "volume": {  # -> L
        "l": ("L", 1.0, 0.0),
        "ml": ("L", 1e-3, 0.0),
        "ul": ("L", 1e-6, 0.0),
        "µl": ("L", 1e-6, 0.0),
    },
    "concentration": {  # -> M
        "m": ("M", 1.0, 0.0),
        "mm": ("M", 1e-3, 0.0),
        "um": ("M", 1e-6, 0.0),
        "µm": ("M", 1e-6, 0.0),
        "nm": ("M", 1e-9, 0.0),
        "pm": ("M", 1e-12, 0.0),
    },
    "temperature": {  # -> K（带偏移）
        "k": ("K", 1.0, 0.0),
        "c": ("K", 1.0, 273.15),
        "°c": ("K", 1.0, 273.15),
    },
    "time": {  # -> s
        "s": ("s", 1.0, 0.0),
        "sec": ("s", 1.0, 0.0),
        "min": ("s", 60.0, 0.0),
        "h": ("s", 3600.0, 0.0),
    },
}

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _norm_unit(u: str) -> str:
    s = (u or "").strip().lower()
    s = s.replace("μ", "µ")
    s = s.replace("μcal", "µcal")
    return s


@dataclass
class Quantity:
    raw: Any
    unit: str
    value: float | None
    canonical_unit: str | None
    diagnostics: list[Diagnostic]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "raw": self.raw,
            "unit": self.unit,
            "value": self.value,
            "canonical_unit": self.canonical_unit,
        }
        if self.diagnostics:
            out["diagnostics"] = [d.to_dict() for d in self.diagnostics]
        return out


def parse_quantity(raw: Any, dimension: str, unit: str | None = None,
                   location: Location | None = None) -> Quantity:
    """保留原始表示；仅在派生需要处规范化。

    raw 可能是数字或字符串（如 "1.23"、"12,3"、"N/A"）。
    """
    diags: list[Diagnostic] = []
    declared = unit
    text = raw
    if isinstance(raw, str):
        m = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*"
                     r"([^\s\d]+)?\s*$", raw.strip())
        if m and declared is None:
            text = m.group(1)
            declared = declared or m.group(2)
    if declared is None:
        diags.append(Diagnostic(
            "unit_missing", "缺少单位，无法规范化；原值保留在原位置",
            location=location))
        return Quantity(raw=raw, unit=unit or "", value=None,
                        canonical_unit=None, diagnostics=diags)
    table = _TABLE.get(dimension)
    if table is None:
        diags.append(Diagnostic(
            "dimension_unknown", f"未知量纲 {dimension}", location=location))
        return Quantity(raw=raw, unit=declared, value=None,
                        canonical_unit=None, diagnostics=diags)
    spec = table.get(_norm_unit(declared))
    if spec is None:
        diags.append(Diagnostic(
            "unit_unconvertible",
            f"单位 {declared!r} 无法转换到 {dimension} 的规范单位；原值保留",
            location=location))
        return Quantity(raw=raw, unit=declared, value=None,
                        canonical_unit=None, diagnostics=diags)
    canonical_unit, factor, offset = spec
    num_text = str(text).strip() if text is not None else ""
    if isinstance(raw, (int, float)):
        num = float(raw)
    elif _NUM_RE.match(num_text.replace(",", ".")):
        num = float(num_text.replace(",", "."))
    else:
        diags.append(Diagnostic(
            "value_unparseable",
            f"值 {raw!r} 不是数值；保留原始表示且不参与计算",
            location=location))
        return Quantity(raw=raw, unit=declared, value=None,
                        canonical_unit=canonical_unit, diagnostics=diags)
    return Quantity(raw=raw, unit=declared, value=num * factor + offset,
                    canonical_unit=canonical_unit, diagnostics=diags)


def require_value(q: Quantity, what: str) -> float:
    """派生层取值：无规范值即语义失败，禁止默认值顶替。"""
    if q.value is None:
        msg = f"{what} 缺少可规范化的值"
        d = q.diagnostics[0] if q.diagnostics else Diagnostic(
            "missing_value", msg)
        from .errors import SemanticUnsolvableError
        raise SemanticUnsolvableError(msg, d.location,
                                      {"raw": q.raw, "unit": q.unit})
    return q.value


def number_column(values: list[Any], dimension: str, unit: str,
                  slot: str) -> tuple[list[float | None], list[dict[str, Any]]]:
    """批量转换一列原始值，返回并行的规范值数组与诊断（不丢原值）。"""
    out: list[float | None] = []
    diags: list[dict[str, Any]] = []
    for i, v in enumerate(values):
        q = parse_quantity(v, dimension, unit,
                           Location(slot=slot, row=i, field=dimension))
        out.append(q.value)
        for d in q.diagnostics:
            diags.append(d.to_dict())
    return out, diags
