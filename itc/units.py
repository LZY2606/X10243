"""单位 / 坐标基准。

领域量在导入层保留 *原始表示*（raw 字符串 + 声明单位 + 基准），
派生层才规范化到 SI 内部单位。无法转换的值留在原位置并附诊断，
绝不丢弃、绝不用默认值替换。

内部规范单位：power cal/s, time s, volume L, concentration M,
heat cal, heat_per_mole cal/mol, temperature K（偏移单位保留 offset）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .errors import Diagnostic

# 单位名 -> 相对内部单位的乘数；温度偏移单独处理
_TO_SI: Dict[str, Tuple[str, float]] = {
    # 功率
    "cal/s": ("cal/s", 1.0),
    "ucal/s": ("cal/s", 1e-6),
    "µcal/s": ("cal/s", 1e-6),
    "w": ("cal/s", 1.0 / 4.184),
    # 能量 / 热量
    "cal": ("cal", 1.0),
    "ucal": ("cal", 1e-6),
    "µcal": ("cal", 1e-6),
    "j": ("cal", 1.0 / 4.184),
    "mj": ("cal", 1e-3 / 4.184),
    # 时间
    "s": ("s", 1.0),
    "sec": ("s", 1.0),
    "min": ("s", 60.0),
    "minute": ("s", 60.0),
    # 体积
    "l": ("L", 1.0),
    "ml": ("L", 1e-3),
    "ul": ("L", 1e-6),
    "µl": ("L", 1e-6),
    # 浓度
    "m": ("M", 1.0),
    "mm": ("M", 1e-3),
    "um": ("M", 1e-6),
    "µm": ("M", 1e-6),
    # 每摩尔热
    "cal/mol": ("cal/mol", 1.0),
    "kcal/mol": ("cal/mol", 1e3),
    "j/mol": ("cal/mol", 1.0 / 4.184),
    "kj/mol": ("cal/mol", 1e3 / 4.184),
    # 亲和力
    "m^-1": ("M^-1", 1.0),
    "1/m": ("M^-1", 1.0),
}

_TEMPERATURE: Dict[str, Tuple[str, float, float]] = {
    "k": ("K", 1.0, 0.0),
    "c": ("K", 1.0, 273.15),
    "degc": ("K", 1.0, 273.15),
    "°c": ("K", 1.0, 273.15),
}

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Quantity:
    """一个携带单位与坐标/时间基准的领域量。

    value=None 且 unconvertible=True 表示无法转换：raw 与 declared_unit
    仍保留在原位置，diagnostics 说明原因。
    """

    raw: str
    declared_unit: Optional[str] = None
    value: Optional[float] = None          # 规范化后的内部值
    si_unit: Optional[str] = None
    basis: Optional[str] = None            # 坐标/时间基准，如 "experiment_t0"
    unconvertible: bool = False
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return not self.unconvertible and self.value is not None


def _norm_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    return u.strip().replace("μ", "µ").lower()


def parse_quantity(raw: Any, declared_unit: Optional[str] = None,
                   basis: Optional[str] = None, location: str = "") -> Quantity:
    """解析单个量；无法转换时返回 unconvertible Quantity（不抛异常、不丢值）。"""
    raw_s = "" if raw is None else str(raw).strip()
    diags: List[Diagnostic] = []
    unit_s, embedded = _split_unit(raw_s, declared_unit)
    token = embedded if embedded is not None else raw_s
    if token == "" or not _NUM_RE.match(token):
        diags.append(Diagnostic("unparseable_number", f"无法解析数值: {raw_s!r}", location))
        return Quantity(raw_s, unit_s, None, None, basis, True, [d.as_dict() for d in diags])
    num = float(token)
    nu = _norm_unit(unit_s)
    if nu is None:
        diags.append(Diagnostic("missing_unit", "数值缺少单位声明", location))
        return Quantity(raw_s, unit_s, None, None, basis, True, [d.as_dict() for d in diags])
    if nu in _TEMPERATURE:
        siu, mult, off = _TEMPERATURE[nu]
        return Quantity(raw_s, unit_s, num * mult + off, siu, basis, False)
    if nu in _TO_SI:
        siu, mult = _TO_SI[nu]
        return Quantity(raw_s, unit_s, num * mult, siu, basis, False)
    diags.append(Diagnostic("unknown_unit", f"未知或无法转换的单位: {unit_s!r}", location))
    return Quantity(raw_s, unit_s, None, None, basis, True, [d.as_dict() for d in diags])


def _split_unit(raw_s: str, declared: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """从 '12.3 ucal/s' 中拆出数字与单位；无内嵌单位时使用声明单位。"""
    m = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*(\S.*)$", raw_s)
    if m and m.group(1).strip():
        return m.group(1).strip(), raw_s[: m.start(1)].strip()
    return declared, None


def parse_column(values: List[Any], declared_unit: Optional[str],
                 basis: Optional[str], location: str) -> Tuple[List[Quantity], List[Quantity]]:
    """解析整列，返回 (可转换量列表（保持行位置）, 不可转换量列表)。

    不可转换条目仍以 None 占位形式保留在列表原位置之外另存，绘图层可用
    掩码跳过它们，但原始行永远存在。
    """
    parsed: List[Quantity] = []
    bad: List[Quantity] = []
    for i, v in enumerate(values):
        q = parse_quantity(v, declared_unit, basis, f"{location}#row{i + 1}")
        parsed.append(q)
        if not q.ok:
            bad.append(q)
    return parsed, bad


def numeric_array(qs: List[Quantity]) -> Tuple["np.ndarray", "np.ndarray"]:
    """把量列转成 float 数组与有效掩码；无效位置为 NaN（不删除行）。"""
    import numpy as np

    arr = np.array([q.value if (q is not None and q.value is not None) else np.nan for q in qs], dtype=float)
    mask = ~np.isnan(arr)
    return arr, mask
