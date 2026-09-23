"""单位与坐标基准。

铁律：导入层只解析、不规范化。每个领域量都同时保留

* ``raw``      —— 原始表示（字符串，逐字保留）
* ``unit``     —— 原始单位
* ``canonical`` —— 派生层规范化后的数值（转换失败时为 None）
* ``canonical_unit`` —— 规范化目标单位（转换失败时仍标注，便于诊断）
* ``diagnostics`` —— 无法转换时的带位置诊断；值留在原位置，绝不为画图丢弃

转换规则集中在本模块，禁止在别处硬编码单位换算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 规范化目标单位（全应用唯一坐标基准）
CANONICAL = {
    "power": "uJ/s",
    "heat": "uJ",
    "time": "s",
    "volume": "uL",
    "concentration": "mM",
    "temperature": "K",
    "dimensionless": "",
}

# 线性量纲换算到规范单位
_PREFIX_SCALE = {
    "power": {"uJ/s": 1.0, "µJ/s": 1.0, "cal/s": 4.184e6, "mcal/s": 4184.0},
    "heat": {"uJ": 1.0, "µJ": 1.0, "mJ": 1000.0, "J": 1e6, "cal": 4.184e6, "mcal": 4184.0},
    "time": {"s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0},
    "volume": {"uL": 1.0, "µL": 1.0, "mL": 1000.0, "L": 1e6, "nL": 1e-3},
    "concentration": {"mM": 1.0, "uM": 1e-3, "µM": 1e-3, "M": 1000.0, "nM": 1e-6},
    "dimensionless": {"": 1.0, "-": 1.0, "1": 1.0},
}


@dataclass
class Diagnostic:
    """带位置的诊断。``path`` 形如 ``rows[7].cell(T)`` / ``materials.concentration``。"""

    code: str
    path: str
    message: str
    raw: Any = None

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": self.message, "raw": self.raw}


@dataclass
class Quantity:
    """一个带单位与坐标基准的领域量。"""

    raw: str
    unit: str | None
    kind: str
    canonical: float | None = None
    canonical_unit: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.canonical is not None and not self.diagnostics

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "unit": self.unit,
            "kind": self.kind,
            "canonical": self.canonical,
            "canonical_unit": self.canonical_unit or CANONICAL.get(self.kind),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _as_float(raw: str) -> float:
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("finite number required")
    return value


def parse_quantity(raw: Any, kind: str, unit: str | None, path: str) -> Quantity:
    """解析单个领域量；转换失败时保留原值并附诊断，不抛异常。"""
    text = "" if raw is None else str(raw).strip()
    q = Quantity(raw=text, unit=(unit.strip() if isinstance(unit, str) else unit), kind=kind,
                 canonical_unit=CANONICAL.get(kind))
    if text == "":
        q.diagnostics.append(Diagnostic("SEMANTIC_EMPTY", path, "缺少数值；不得用空值或默认值伪装成功", raw))
        return q
    try:
        value = _as_float(text)
    except (TypeError, ValueError):
        q.diagnostics.append(Diagnostic("FORMAT_NOT_A_NUMBER", path, f"无法解析为数值: {text!r}", raw))
        return q
    if kind == "dimensionless" and q.unit is None:
        q.unit = ""
    if kind == "temperature":
        return _convert_temperature(q, value, path)
    table = _PREFIX_SCALE.get(kind)
    if not table:
        q.diagnostics.append(Diagnostic("SEMANTIC_UNKNOWN_KIND", path, f"未知量纲 {kind!r}", raw))
        return q
    if q.unit is None:
        q.diagnostics.append(Diagnostic("SEMANTIC_UNIT_MISSING", path, f"{kind} 缺少单位", raw))
        return q
    scale = table.get(q.unit)
    if scale is None:
        q.diagnostics.append(
            Diagnostic("SEMANTIC_UNIT_UNKNOWN", path, f"单位 {q.unit!r} 不属于 {kind} 可识别单位", raw))
        return q
    q.canonical = value * scale
    return q


def _convert_temperature(q: Quantity, value: float, path: str) -> Quantity:
    if q.unit in ("K", "Kelvin"):
        q.canonical = value
    elif q.unit in ("C", "°C", "degC", "Celsius"):
        q.canonical = value + 273.15
    elif q.unit in ("F", "°F", "Fahrenheit"):
        q.canonical = (value - 32.0) * 5.0 / 9.0 + 273.15
    else:
        q.diagnostics.append(
            Diagnostic("SEMANTIC_UNIT_UNKNOWN", path, f"温度单位 {q.unit!r} 无法转换（K/C/F）", q.raw))
    return q


def require(q: Quantity) -> float:
    """派生层调用：要求可转换，否则抛语义错误（带原位置）。"""
    if not q.ok:
        bad = q.diagnostics[0]
        raise SemanticError(bad.code, bad.path, bad.message, q.raw)
    assert q.canonical is not None
    return q.canonical


class IngestError(Exception):
    """输入问题的三类不同结果之一的公共基类（绝不混用）。"""

    kind = "INGEST_ERROR"


class FormatCorruptError(IngestError):
    """格式损坏：结构无法解析，附行列位置。"""

    kind = "FORMAT_CORRUPT"

    def __init__(self, message: str, path: str = "input"):
        super().__init__(f"[FORMAT_CORRUPT] {path}: {message}")
        self.message = message
        self.path = path


class BudgetExhaustedError(IngestError):
    """资源预算耗尽：输入过大 / 拟合迭代耗尽，与损坏、语义无解区分。"""

    kind = "BUDGET_EXHAUSTED"

    def __init__(self, message: str, path: str = "input", limit: int | None = None):
        super().__init__(f"[BUDGET_EXHAUSTED] {path}: {message}")
        self.message = message
        self.path = path
        self.limit = limit


class SemanticError(IngestError):
    """语义无解：结构能读，但领域含义不成立。"""

    kind = "SEMANTIC_NO_SOLUTION"

    def __init__(self, code: str, path: str, message: str, raw: Any = None):
        super().__init__(f"[{code}] {path}: {message}")
        self.code = code
        self.path = path
        self.message = message
        self.raw = raw


def error_dict(exc: Exception) -> dict:
    if isinstance(exc, FormatCorruptError):
        return {"result": "FORMAT_CORRUPT", "path": exc.path, "message": exc.message}
    if isinstance(exc, BudgetExhaustedError):
        return {"result": "BUDGET_EXHAUSTED", "path": exc.path, "limit": exc.limit, "message": exc.message}
    if isinstance(exc, SemanticError):
        return {"result": "SEMANTIC_NO_SOLUTION", "code": exc.code, "path": exc.path,
                "message": exc.message, "raw": exc.raw}
    return {"result": "ERROR", "message": str(exc)}
