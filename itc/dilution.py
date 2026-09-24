"""稀释热（空白）扣除。

硬规则：**稀释热扣除必须说明对应规则**。没有显式对应规则时返回 UNRESOLVABLE，
不做隐式猜测。对应规则是可版本化的决定（rule 版本），支持：

- ``same_index``   样本注射序号 == 空白注射序号；
- ``explicit``     人工显式映射 ``{sample_index: blank_index}``；
- ``mean``         以全部空白注射热的均值作为每注稀释热（需显式声明）。

空白对应缺失（规则存在但找不到对应空白）时，受影响注射保留原热并打
``BLANK_CORRESPONDENCE_MISSING`` 状态，其余照常扣除。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from .integration import InjectionResult, InjectionStatus
from .results import Diagnostic


class RuleKind(str, Enum):
    SAME_INDEX = "same_index"
    EXPLICIT = "explicit"
    MEAN = "mean"


class CorrectionStatus(str, Enum):
    CORRECTED = "corrected"
    MISSING = "blank_correspondence_missing"
    NO_RULE = "no_blank_rule"          # 未声明对应规则：不扣除
    SKIPPED = "skipped"               # 首注舍弃等非积分点


@dataclass(frozen=True)
class BlankRule:
    rule: RuleKind
    mapping: dict[int, int] = field(default_factory=dict)
    note: str = ""

    @property
    def rule_id(self) -> str:
        body = repr((self.rule.value,
                     tuple(sorted(self.mapping.items())), self.note))
        return "blankrule:" + hashlib.sha256(body.encode()).hexdigest()[:12]


@dataclass
class CorrectedHeat:
    injection_index: int
    status: CorrectionStatus
    sample_heat_cal: float | None
    blank_heat_cal: float | None = None
    net_heat_cal: float | None = None
    blank_injection_index: int | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    rule_id: str = ""

    def as_dict(self) -> dict:
        return {
            "injection_index": self.injection_index,
            "status": self.status.value,
            "sample_heat_cal": self.sample_heat_cal,
            "blank_heat_cal": self.blank_heat_cal,
            "net_heat_cal": self.net_heat_cal,
            "blank_injection_index": self.blank_injection_index,
            "rule_id": self.rule_id,
            "diagnostics": [vars(d) for d in self.diagnostics],
        }


def subtraction_id(sample_integration_id: str, blank_integration_id: str,
                   rule: BlankRule | None) -> str:
    body = repr((sample_integration_id, blank_integration_id,
                 rule.rule_id if rule else None))
    return "sub_" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _blank_lookup(blank_results: list[InjectionResult]) -> dict[int, InjectionResult]:
    return {r.injection_index: r for r in blank_results
            if r.status is InjectionStatus.INTEGRATED and r.heat_cal is not None}


def subtract_blank(
    *,
    sample: list[InjectionResult],
    blank: list[InjectionResult],
    rule: BlankRule | None,
) -> list[CorrectedHeat]:
    """按显式规则扣除；缺规则不猜测，缺对应不伪造。"""
    lookup = _blank_lookup(blank)
    mean_heat = (sum(r.heat_cal for r in lookup.values()) / len(lookup)
                 if rule and rule.rule is RuleKind.MEAN and lookup else None)
    out: list[CorrectedHeat] = []
    for sr in sample:
        ch = CorrectedHeat(sr.injection_index, CorrectionStatus.SKIPPED,
                           sr.heat_cal, rule_id=rule.rule_id if rule else "")
        if sr.status is InjectionStatus.DISCARDED:
            ch.status = CorrectionStatus.SKIPPED
            ch.diagnostics.append(Diagnostic(
                "SKIP_DISCARDED", "首注舍弃，不参与稀释扣除",
                f"injection#{sr.injection_index}", severity="warning"))
            out.append(ch)
            continue
        if sr.status is not InjectionStatus.INTEGRATED or sr.heat_cal is None:
            ch.status = CorrectionStatus.SKIPPED
            ch.diagnostics.append(Diagnostic(
                "SKIP_NON_INTEGRATED", f"样本注射状态 {sr.status.value}，不可扣除",
                f"injection#{sr.injection_index}", severity="warning"))
            out.append(ch)
            continue
        if rule is None:
            # 未声明对应规则：语义无解，不做扣除
            ch.status = CorrectionStatus.NO_RULE
            ch.sample_heat_cal = sr.heat_cal
            ch.net_heat_cal = sr.heat_cal
            ch.diagnostics.append(Diagnostic(
                "BLANK_RULE_MISSING",
                "稀释热扣除缺少对应规则；保留未扣除热，等待显式规则",
                f"injection#{sr.injection_index}"))
            out.append(ch)
            continue
        blank_idx: int | None = None
        if rule.rule is RuleKind.SAME_INDEX:
            blank_idx = sr.injection_index
        elif rule.rule is RuleKind.EXPLICIT:
            blank_idx = rule.mapping.get(sr.injection_index)
        elif rule.rule is RuleKind.MEAN:
            blank_idx = -1
        br = lookup.get(blank_idx) if blank_idx is not None and blank_idx >= 0 else None
        if rule.rule is RuleKind.MEAN:
            if mean_heat is None:
                ch.status = CorrectionStatus.MISSING
                ch.sample_heat_cal = sr.heat_cal
                ch.net_heat_cal = sr.heat_cal
                ch.diagnostics.append(Diagnostic(
                    "BLANK_CORRESPONDENCE_MISSING",
                    "mean 规则下没有任何可用空白注射积分",
                    f"injection#{sr.injection_index}"))
            else:
                ch.status = CorrectionStatus.CORRECTED
                ch.sample_heat_cal = sr.heat_cal
                ch.blank_heat_cal = mean_heat
                ch.net_heat_cal = sr.heat_cal - mean_heat
                ch.blank_injection_index = -1
            out.append(ch)
            continue
        if br is None:
            ch.status = CorrectionStatus.MISSING
            ch.sample_heat_cal = sr.heat_cal
            ch.net_heat_cal = sr.heat_cal
            ch.blank_injection_index = blank_idx
            ch.diagnostics.append(Diagnostic(
                "BLANK_CORRESPONDENCE_MISSING",
                f"对应规则 {rule.rule.value} 指向空白注射 #{blank_idx}，"
                "但该空白缺失/未积分；该注保留未扣除热",
                f"injection#{sr.injection_index}"))
            out.append(ch)
            continue
        ch.status = CorrectionStatus.CORRECTED
        ch.sample_heat_cal = sr.heat_cal
        ch.blank_heat_cal = br.heat_cal
        ch.net_heat_cal = sr.heat_cal - br.heat_cal
        ch.blank_injection_index = blank_idx
        out.append(ch)
    return out
