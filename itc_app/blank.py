"""空白对照 / 稀释热扣除。

对应规则必须显式说明：
- index_match (v1)：主实验第 i 注 ↔ 空白第 i 注（同注射序号、同体积）。
空白对应缺失时不猜测、不回退为 0，状态置 no_match，corrected=None。
"""
from __future__ import annotations

from .model import BlankCorrection

RULE_VERSION = "blank-index-2026.1"


def apply_blank(integral_points, blank_points, rule="index_match",
                rule_version=RULE_VERSION, power_id="", blank_power_id=""):
    blank_by_index = {b.injection_index: b for b in (blank_points or [])}
    out = []
    for p in integral_points:
        idx = p.injection_index
        base = {
            "rule": rule, "rule_version": rule_version,
            "sample_integral": p.id, "blank_power": blank_power_id,
        }
        if p.discarded or p.energy_j is None:
            out.append(BlankCorrection(
                injection_index=idx,
                id=f"blank:{power_id}:inj{idx}",
                matched_blank_index=None, blank_energy_j=None,
                corrected_energy_j=None, rule=rule, rule_version=rule_version,
                status="not_applied",
                reason=f"原积分状态为 {p.status}，不进行空白扣除",
                inputs=base))
            continue
        b = blank_by_index.get(idx)
        if b is None or b.discarded or b.energy_j is None:
            out.append(BlankCorrection(
                injection_index=idx,
                id=f"blank:{power_id}:inj{idx}",
                matched_blank_index=None, blank_energy_j=None,
                corrected_energy_j=None, rule=rule, rule_version=rule_version,
                status="no_match",
                reason=("空白对应缺失：index_match 规则要求同序号空白注，"
                        "缺失时禁止回退为 0"),
                inputs=base))
            continue
        # 体积一致性属于对应规则的一部分（在调用层准备 volume 字段）
        out.append(BlankCorrection(
            injection_index=idx,
            id=f"blank:{power_id}:inj{idx}",
            matched_blank_index=idx, blank_energy_j=b.energy_j,
            corrected_energy_j=p.energy_j - b.energy_j,
            rule=rule, rule_version=rule_version, status="corrected",
            reason="按注射序号一一对应扣除稀释热",
            inputs={**base, "blank_integral": b.id}))
    return out
