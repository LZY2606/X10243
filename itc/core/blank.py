"""空白对照对应与稀释热扣除。

每条扣除必须说明对应规则（blank-match-v1）：
- match_by_index：按注射序号对应；
- match_by_volume：按体积最近邻对应（稳定并列：取序号较小者）；
- no_blank：不扣除。
空白对应缺失 -> blank_missing，保留原始热量，绝不以零稀释热冒充扣除。
空白注射也走同一窗口/积分规则，体积不同的稀释热按体积比缩放。
"""
from __future__ import annotations

from typing import Any

from .constants import (BLANK_RULE_INDEX, BLANK_RULE_NONE, BLANK_RULE_VERSION,
                        BLANK_RULE_VOLUME, STATUS_BLANK_MISSING,
                        STATUS_DISCARDED, STATUS_OK)
from .integration import integrate_all
from .materials import blank_view
from .windows import compute_windows


def build_blank_heats(blank_material: dict[str, Any],
                      baseline_material: dict[str, Any] | None
                      ) -> list[dict[str, Any]]:
    view = blank_view(blank_material)
    windows = compute_windows(view["injections"], view)
    return integrate_all(blank_material, baseline_material, windows, view,
                         set())


def match_and_subtract(raw_heats: list[dict[str, Any]],
                       injections: list[dict[str, Any]],
                       rule: str,
                       blank_material: dict[str, Any] | None,
                       baseline_material: dict[str, Any] | None
                       ) -> dict[str, Any]:
    """返回 blank_map 节点：逐针对应、缩放、缺失标记与规则证据。"""
    entries: list[dict[str, Any]] = []
    blank_heats: dict[int, dict[str, Any]] = {}
    if rule != BLANK_RULE_NONE and blank_material is not None:
        for h in build_blank_heats(blank_material, baseline_material):
            blank_heats[h["ordinal"]] = h

    for rh in raw_heats:
        ordinal = rh["ordinal"]
        entry: dict[str, Any] = {
            "ordinal": ordinal,
            "label": rh["label"],
            "raw_heat": rh.get("value"),
            "raw_heat_unit": rh.get("unit", "ucal"),
            "rule": rule,
            "rule_version": BLANK_RULE_VERSION,
        }
        if rh.get("status") != STATUS_OK:
            entry.update(status=rh["status"], matched=None,
                         dilution_heat=None, net_heat=None,
                         reason=rh.get("reason"))
            entries.append(entry)
            continue
        if rule == BLANK_RULE_NONE or blank_material is None:
            entry.update(status=STATUS_OK, matched=None,
                         dilution_heat=0.0,
                         net_heat=rh["value"], scale=1.0,
                         correspondence="explicit_no_blank")
            entries.append(entry)
            continue
        target = _match(ordinal, injections, rule,
                        list(blank_heats.values()), blank_material)
        if target is None:
            entry.update(status=STATUS_BLANK_MISSING, matched=None,
                         dilution_heat=None, net_heat=rh["value"],
                         net_heat_unsubtracted=True,
                         reason="no_blank_correspondence")
            entries.append(entry)
            continue
        bh = blank_heats[target]
        if bh.get("status") not in (STATUS_OK, STATUS_DISCARDED) or \
                bh.get("value") is None:
            entry.update(status=STATUS_BLANK_MISSING, matched=target,
                         dilution_heat=None, net_heat=rh["value"],
                         net_heat_unsubtracted=True,
                         reason=f"blank_status:{bh.get('status')}")
            entries.append(entry)
            continue
        v_sample = _volume(injections, ordinal)
        b_injs = blank_view(blank_material)["injections"]
        v_blank = _volume(b_injs, target)
        scale = (v_sample / v_blank) if v_sample and v_blank else 1.0
        dilution = bh["value"] * scale
        entry.update(status=STATUS_OK, matched=target, scale=scale,
                     blank_heat=bh["value"],
                     dilution_heat=dilution,
                     net_heat=rh["value"] - dilution,
                     correspondence=rule)
        entries.append(entry)
    entries.sort(key=lambda e: e["ordinal"])
    return {"node_type": "blank_map", "rule": rule,
            "rule_version": BLANK_RULE_VERSION, "entries": entries}


def _match(ordinal: int, injections: list[dict[str, Any]], rule: str,
           blank_heats: list[dict[str, Any]],
           blank_material: dict[str, Any]) -> int | None:
    if not blank_heats:
        return None
    ordinals = sorted(h["ordinal"] for h in blank_heats)
    if rule == BLANK_RULE_INDEX:
        return ordinal if ordinal in ordinals else None
    if rule == BLANK_RULE_VOLUME:
        target_v = _volume(injections, ordinal)
        if not target_v:
            return None
        b_injs = blank_view(blank_material)["injections"]
        best: tuple[float, int] | None = None
        for b_ord in ordinals:
            bv = _volume(b_injs, b_ord)
            if not bv:
                continue
            cand = (abs(bv - target_v), b_ord)
            if best is None or cand < best:
                best = cand
        return best[1] if best else None
    return None


def _volume(injections: list[dict[str, Any]], ordinal: int) -> float | None:
    for inj in injections:
        if inj["ordinal"] == ordinal:
            return inj.get("volume")
    return None
