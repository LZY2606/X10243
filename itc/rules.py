"""可修改决定（规则层）与不可变材料分层。

规则版本是命名、版本化、内容寻址的；计算结果记录使用的规则版本，
旧结果永不被新计算覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .identity import rule_id

RULE_SET_INTEGRATION = "injection_integration"
RULE_SET_BLANK = "blank_correspondence"
RULE_SET_MODEL = "binding_model_selection"
RULE_SET_FIRST_INJECTION = "first_injection"


@dataclass(frozen=True)
class IntegrationRules:
    version: str
    # 注射后峰积分窗：[t_inj, t_inj + window_seconds]
    window_seconds: float = 120.0
    # 判定相邻采样构成“数据缺口”的最小时间间隔（秒）
    gap_threshold_seconds: float = 10.0
    # 重叠峰处理：independent=各峰用完整名义窗并标记 overlap；partition=窗在下一注射处截断
    overlap_policy: str = "independent"
    # 首注舍弃
    discard_first: bool = True
    # 积分端点检验：窗末 tail_fraction 段的残差不得超过 tail_sigma * 噪声
    tail_fraction: float = 0.2
    tail_sigma: float = 4.0
    baseline_fit_seconds: float = 60.0
    baseline_degree: int = 0  # 0=常数, 1=线性

    def content(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "version"}

    @property
    def id(self) -> str:
        return rule_id(RULE_SET_INTEGRATION, self.version, self.content())


@dataclass(frozen=True)
class BlankRules:
    version: str
    # 空白对应规则：必须显式说明，缺失时不得静默
    # same_index=按注射序号对应；match_volume=按最接近注射体积对应；manual=人工映射；skip=不扣除
    correspondence: str = "same_index"
    volume_tolerance_l: float = 0.05e-6
    manual_map: dict[str, int] | None = None  # titration injection index -> blank injection index
    require_correspondence: bool = True

    def content(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "version"}

    @property
    def id(self) -> str:
        return rule_id(RULE_SET_BLANK, self.version, self.content())


@dataclass(frozen=True)
class ModelRules:
    version: str
    n_sites: tuple[int, ...] = (1, 2)
    include_dilution: bool = True
    # 多个等优模型判定：delta BIC <= tie_delta 视为等优，全部保留
    tie_delta: float = 2.0
    # 资源预算：模型评估次数上限
    eval_budget: int = 20000
    dilution_degree: int = 0  # 0=常数, 1=常数+斜率

    def content(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "version"}
        d["n_sites"] = list(self.n_sites)
        return d

    @property
    def id(self) -> str:
        return rule_id(RULE_SET_MODEL, self.version, self.content())
