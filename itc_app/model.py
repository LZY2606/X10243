"""材料层（不可变、版本化）与派生层（绑定父依赖）的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


MATERIAL_KINDS = (
    "power",      # 功率采样
    "protocol",   # 注射体积与时间
    "ligand",     # 配体（注射器）浓度
    "analyte",    # 池中被滴定物浓度
    "temperature",
    "cell",       # 池体积
    "baseline",   # 基线曲线（单独版本化）
    "blank",      # 空白对照曲线（单独版本化）
)


@dataclass
class Sample:
    t: float
    p: float | None  # None 表示数据缺口
    gap: bool = False

    def to_dict(self) -> dict:
        return {"t": self.t, "p": self.p, "gap": self.gap}


@dataclass
class Injection:
    index: int          # 1-based 协议序号（稳定身份的一部分）
    time: float         # s
    duration: float     # s
    volume: float       # L
    raw_time: str
    raw_duration: str
    raw_volume: str


@dataclass
class Material:
    kind: str
    version: int
    source: str                 # 直接来源（文件名等）
    raw: str                    # 导入原始文本（逐字）
    data: dict[str, Any]        # 解析后结构（Quantity 保留 raw）
    diagnostics: list[dict] = field(default_factory=list)
    supersedes: int | None = None
    # 局部依赖改动时标记受影响的协议注射序号（None=全局）
    affects: list[int] | None = None

    def mid(self) -> str:
        return f"{self.kind}@v{self.version}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntegralPoint:
    injection_index: int
    id: str
    win_start: float
    win_end: float
    energy_j: float | None
    baseline_used: dict
    overlap_with: list[int]
    crosses_gap: bool
    gap_segments: list[list[float]]
    endpoints: list[float]
    endpoint_ok: bool
    discarded: bool
    status: str          # ok | discarded | crosses_gap | no_data
    reason: str
    inputs: dict         # 输入摘要 + 父依赖版本
    samples_used: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlankCorrection:
    injection_index: int
    id: str
    matched_blank_index: int | None
    blank_energy_j: float | None
    corrected_energy_j: float | None
    rule: str
    rule_version: str
    status: str          # corrected | no_match | not_applied
    reason: str
    inputs: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelCandidate:
    id: str
    model: str           # one_site | two_site | dilution
    params: dict[str, Any]
    param_units: dict[str, str]
    aic: float | None
    sse: float | None
    n_points: int
    rank: int
    equivalent: bool
    converged: bool
    budget_exhausted: bool
    reason: str
    inputs: dict
    valid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
