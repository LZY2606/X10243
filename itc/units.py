"""单位与坐标基准。

导入时保留原始表示（见 ingest.raw）；规范化只发生在派生层。
无法转换的值不丢弃：返回 None + Diagnostic，留在原位置。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic

# 规范单位（派生层统一坐标基准）
CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "heat": "uJ",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
}

# 线性换算：canonical = raw * factor
_FACTORS: dict[tuple[str, str], float] = {
    ("time", "s"): 1.0, ("time", "sec"): 1.0, ("time", "second"): 1.0,
    ("time", "ms"): 1e-3, ("time", "min"): 60.0, ("time", "minute"): 60.0,

    ("power", "uj/s"): 1.0, ("power", "µj/s"): 1.0,
    ("power", "ucal/s"): 4.184, ("power", "µcal/s"): 4.184,
    ("power", "cal/s"): 4.184e6, ("power", "mw"): 1e3, ("power", "w"): 1e6,

    ("heat", "uj"): 1.0, ("heat", "µj"): 1.0,
    ("heat", "ucal"): 4.184, ("heat", "µcal"): 4.184,
    ("heat", "cal"): 4.184e6, ("heat", "mj"): 1e3, ("heat", "j"): 1e6,

    ("volume", "ul"): 1.0, ("volume", "µl"): 1.0,
    ("volume", "ml"): 1e3, ("volume", "l"): 1e6,

    ("concentration", "mm"): 1.0, ("concentration", "mmol/l"): 1.0,
    ("concentration", "um"): 1e-3, ("concentration", "µm"): 1e-3,
    ("concentration", "m"): 1e3, ("concentration", "mol/l"): 1e3,
}


@dataclass
class Quantity:
    """一个带单位与坐标基准的量；raw 为原始表示。"""
    raw: Any
    raw_unit: str | None
    canonical: float | None
    canonical_unit: str | None
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "raw_unit": self.raw_unit,
            "canonical": self.canonical,
            "canonical_unit": self.canonical_unit,
            "path": self.path,
        }


def normalize(
    raw: Any,
    raw_unit: str | None,
    dimension: str,
    path: str = "",
) -> tuple[float | None, Diagnostic | None]:
    """规范化到该维度的规范单位。失败时返回 (None, 诊断)，原值不丢。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, Diagnostic(
            "value_unparsable",
            f"无法解析数值 {raw!r}（维度 {dimension}），值保留在 {path} 未参与计算",
            path, "error")
    if dimension not in CANONICAL:
        return None, Diagnostic(
            "dimension_unknown",
            f"未知量纲 {dimension!r}，无法规范化 {path}", path, "error")
    if raw_unit is None or str(raw_unit).strip() == "":
        return None, Diagnostic(
            "unit_missing",
            f"{path} 缺少单位，拒绝猜测规范值；原始值 {raw!r} 保留",
            path, "error")
    if dimension == "temperature":
        u = str(raw_unit).strip().lower()
        if u in ("k", "kelvin"):
            return value, None
        if u in ("c", "°c", "degc", "celsius"):
            return value + 273.15, None
        return None, Diagnostic(
            "unit_unknown",
            f"温度单位 {raw_unit!r} 无法转换，原值 {value} 留在 {path}，未做规范化",
            path, "error")
    key = (dimension, str(raw_unit).strip().lower())
    if key not in _FACTORS:
        return None, Diagnostic(
            "unit_unknown",
            f"单位 {raw_unit!r} 不属于 {dimension} 可转换表，原值 {value} 留在 {path}",
            path, "error")
    return value * _FACTORS[key], None
