"""单位与坐标基准：导入层保留原始表示，派生层才做规范化。

规则：
- 每个领域量都携带单位或坐标基准（raw 原样保留）。
- 无法转换的值留在原位置并附诊断，不得丢弃或用默认值伪装成功。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Optional

CANONICAL_UNITS = {
    "power": "µJ/s",
    "time": "s",
    "volume": "µL",
    "concentration": "µM",
    "temperature": "K",
    "heat": "µJ",
}

# 线性换算表：value_canonical = value_raw * factor
_LINEAR = {
    "power": {"µJ/s": 1.0, "uJ/s": 1, "µcal/s": 4.184, "ucal/s": 4.184},
    "time": {"s": 1.0, "min": 60.0},
    "volume": {"µL": 1.0, "uL": 1.0, "µl": 1.0, "ul": 1.0, "mL": 1000.0, "ml": 1000.0},
    "concentration": {
        "µM": 1.0, "uM": 1.0, "umol/L": 1.0, "µmol/L": 1.0,
        "mM": 1000.0, "mmol/L": 1000.0,
    },
}

# 温度是仿射换算：K = value + offset
_TEMP_OFFSET = {"K": 0.0, "°C": 273.15, "C": 273.15, "degC": 273.15}

_NUM_UNIT_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(.*)$")


@dataclass
class Diagnostic:
    path: str
    code: str
    message: str
    raw: Any = None

    def to_json(self) -> dict:
        return asdict(self)


def _split_raw(raw: Any):
    """把原始表示拆成 (数值, 单位或None)。无法解析返回 (None, None)。"""
    if isinstance(raw, dict):
        return raw.get("value"), raw.get("unit")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), None
    if isinstance(raw, str):
        m = _NUM_UNIT_RE.match(raw)
        if m:
            unit = m.group(2).strip() or None
            return float(m.group(1)), unit
    return None, None


def normalize_scalar(raw: Any, kind: str, path: str,
                     default_unit: Optional[str] = None):
    """规范化单个量。返回 (value_or_None, diagnostic_or_None)。

    无法转换时 value 为 None，调用方必须把 raw 留在原位置并携带诊断。
    """
    value, unit = _split_raw(raw)
    if value is None:
        return None, Diagnostic(path, "value_unparseable",
                                f"无法解析数值: {raw!r}", raw)
    unit = unit or default_unit
    if unit is None:
        return None, Diagnostic(path, "unit_missing",
                                f"缺少单位（期望 {CANONICAL_UNITS[kind]}）", raw)
    if kind == "temperature":
        if unit in _TEMP_OFFSET:
            return value + _TEMP_OFFSET[unit], None
        return None, Diagnostic(path, "unit_unconvertible",
                                f"未知温度单位 {unit!r}", raw)
    table = _LINEAR.get(kind, {})
    if unit in table:
        return value * table[unit], None
    return None, Diagnostic(path, "unit_unconvertible",
                            f"无法把单位 {unit!r} 换算为 {CANONICAL_UNITS[kind]}", raw)


def normalize_trace(raw: Any, path: str):
    """规范化功率/基线/空白曲线。

    原始 points 原样保留在 raw 中；此处只产出派生数组。
    坏点留在原位置（raw 未动），派生数组跳过该点并附诊断（含点位）。
    返回 dict(t=[...], y=[...], basis_t=..., diagnostics=[...])，
    结构损坏时返回 dict(format_error=...)。
    """
    if not isinstance(raw, dict) or "points" not in raw or "meta" not in raw:
        return {"format_error": f"{path}: 曲线需要 meta 与 points 字段",
                "diagnostics": [Diagnostic(path, "format_error",
                                           "曲线结构损坏").to_json()]}
    meta = raw.get("meta") or {}
    unit_t = meta.get("unit_t")
    unit_y = meta.get("unit_y")
    ts, ys, diags = [], [], []
    points = raw.get("points")
    if not isinstance(points, list):
        return {"format_error": f"{path}: points 必须是数组",
                "diagnostics": [Diagnostic(path, "format_error",
                                           "points 不是数组").to_json()]}
    for i, pt in enumerate(points):
        ppath = f"{path}.points[{i}]"
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            diags.append(Diagnostic(ppath, "format_error",
                                    "采样点需要 [t, y]", pt).to_json())
            continue
        tv, td = normalize_scalar(pt[0], "time", ppath + ".t", unit_t)
        yv, yd = normalize_scalar(pt[1], "power", ppath + ".y", unit_y)
        if td:
            diags.append(td.to_json())
        if yd:
            diags.append(yd.to_json())
        if tv is None or yv is None:
            continue  # 坏点不进派生数组，但 raw 中保留
        ts.append(tv)
        ys.append(yv)
    return {"t": ts, "y": ys, "basis_t": meta.get("basis_t", "epoch:unknown"),
            "unit_t": "s", "unit_y": "µJ/s", "diagnostics": diags}


def normalize_schedule(raw: Any, path: str):
    """规范化注射时序：每次注射的时间与体积。"""
    if not isinstance(raw, dict) or "injections" not in raw:
        return {"format_error": f"{path}: 时序需要 injections 字段",
                "diagnostics": [Diagnostic(path, "format_error",
                                           "时序结构损坏").to_json()]}
    meta = raw.get("meta") or {}
    out, diags = [], []
    for i, inj in enumerate(raw["injections"]):
        ppath = f"{path}.injections[{i}]"
        tv, td = normalize_scalar(inj.get("t"), "time", ppath + ".t",
                                  meta.get("unit_t"))
        vv, vd = normalize_scalar(inj.get("volume"), "volume", ppath + ".volume",
                                  meta.get("unit_volume"))
        for d in (td, vd):
            if d:
                diags.append(d.to_json())
        out.append({"index": i, "t": tv, "volume_uL": vv,
                    "ok": tv is not None and vv is not None})
    return {"injections": out, "basis_t": meta.get("basis_t", "epoch:unknown"),
            "diagnostics": diags}
