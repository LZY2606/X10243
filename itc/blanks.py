"""稀释热扣除：空白对应规则必须显式说明。

规则来自版本化的 BlankRules；对应缺失是带位置的显式诊断，绝不静默。
"""
from __future__ import annotations

from .errors import Diagnostic
from .identity import result_id
from .integration import IntegrationOutcome
from .materials import BLANK
from .rules import BlankRules


def _diag(code: str, message: str, location: str) -> dict:
    return Diagnostic(code, message, location).to_dict()


def apply_blank(
    titration: IntegrationOutcome,
    blank: IntegrationOutcome,
    materials: dict,
    rules: BlankRules,
) -> IntegrationOutcome:
    """返回新的积分结果（同一父材料 + 空白 + 规则），不覆盖旧结果。"""
    blank_by_index = {b["index"]: b for b in blank.injections}
    out_injections = []
    diagnostics = list(titration.diagnostics)

    for inj in titration.injections:
        r = dict(inj)
        r["diagnostics"] = list(inj["diagnostics"])
        q = inj["q"]
        q_blank = None
        matched_index = None
        rule_note = rules.correspondence
        loc = f"injections[{inj['index']}].blank"

        if q is None:
            r["q_net"] = None
            r["blank_rule"] = {"correspondence": rules.correspondence, "matched_blank": None,
                               "scale": None, "note": "原始积分缺失，无法扣除"}
            out_injections.append(r)
            continue

        matched: dict | None = None
        if rules.correspondence == "same_index":
            matched = blank_by_index.get(inj["index"])
            matched_index = inj["index"] if matched else None
        elif rules.correspondence == "match_volume":
            v = inj["volume_l"]
            if v is not None:
                cands = [b for b in blank.injections
                         if b["q"] is not None and b["volume_l"] is not None
                         and abs(b["volume_l"] - v) <= rules.volume_tolerance_l]
                cands.sort(key=lambda b: (abs(b["volume_l"] - v), b["index"]))
                matched = cands[0] if cands else None
                matched_index = matched["index"] if matched else None
        elif rules.correspondence == "manual":
            mapping = rules.manual_map or {}
            if str(inj["index"]) in mapping:
                mi = mapping[str(inj["index"])]
                matched = blank_by_index.get(mi)
                matched_index = mi if matched else None
        elif rules.correspondence == "skip":
            r["q_net"] = q
            r["blank_rule"] = {"correspondence": "skip", "matched_blank": None, "scale": None,
                               "note": "显式声明不扣除稀释热"}
            out_injections.append(r)
            continue
        else:
            r["q_net"] = None
            r["diagnostics"].append(_diag("unknown_blank_rule",
                                          f"未知空白对应规则 {rules.correspondence}", loc))
            out_injections.append(r)
            continue

        if matched is None or matched.get("q") is None:
            if rules.require_correspondence:
                r["q_net"] = None
                r["diagnostics"].append(_diag(
                    "blank_correspondence_missing",
                    f"按规则 {rules.correspondence} 未找到第{inj['index']}注的空白对应；不得静默扣除",
                    loc))
            else:
                r["q_net"] = q
                r["diagnostics"].append(_diag(
                    "blank_correspondence_missing",
                    "空白对应缺失；按规则允许回退为不扣除", loc))
            r["blank_rule"] = {"correspondence": rules.correspondence, "matched_blank": matched_index,
                               "scale": None, "note": "对应缺失"}
            out_injections.append(r)
            continue

        scale = inj["volume_l"] / matched["volume_l"] if (inj["volume_l"] and matched["volume_l"]) else 1.0
        q_blank = matched["q"] * scale
        r["q_net"] = q - q_blank
        r["blank_rule"] = {"correspondence": rules.correspondence,
                           "matched_blank": matched_index,
                           "scale": scale,
                           "q_blank_scaled": q_blank,
                           "note": f"{rule_note}：滴定第{inj['index']}注 ↔ 空白第{matched_index}注，按体积比 {scale:.4g} 缩放"}
        out_injections.append(r)

    parent_ids = sorted(set(titration.parent_ids) | {blank.id, materials[BLANK].id, rules.id})
    input_summary = {
        "titration_result": titration.id,
        "blank_result": blank.id,
        "blank_injections": len(blank.injections),
        "heat_basis": "cal",
    }
    rid = result_id("integration_corrected", input_summary, rules.content(), parent_ids)
    for r in out_injections:
        from .identity import point_id
        r["point_ids"] = [point_id(rid, f"injection-{r['index']}")]

    return IntegrationOutcome(
        id=rid,
        injections=out_injections,
        diagnostics=diagnostics,
        input_summary=input_summary,
        parent_ids=parent_ids,
        rule_version=rules.version,
        rule_id=rules.id,
    )
