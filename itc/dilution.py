"""空白（稀释热）对应与扣除。

规则：
- 对应规则必须显式声明（DILUTION_RULE_VERSION 记录在产物里），没有“默认对上”。
  支持 by_index：主滴定第 i 注 ↔ 空白第 i 注（序号相同）。
- 空白自身也是一次滴定，用相同积分窗与基线规则积分得到空白热。
- 空白对应缺失（缺注或空白注不可积分）时，该注 net=None 并附带位置诊断，
  该注不进入绑定模型；其余注正常。
- 体积修正：空白热按注射体积比例缩放后再扣除。
"""
from __future__ import annotations

from .integration import integrate_injections
from .models import Diagnostic
from .storage import digest

DILUTION_RULE_VERSION = "dilution-1.0.0"
SUPPORTED_RULES = ("by_index",)


def apply_dilution(integration_doc: dict, blank: dict | None,
                   blank_baseline: dict | None, params: dict) -> dict:
    rule = params.get("rule", "by_index")
    if rule not in SUPPORTED_RULES:
        from .models import ITCError
        raise ITCError(
            f"不支持的稀释对应规则 {rule!r}",
            [Diagnostic("unknown_dilution_rule",
                        f"支持的规则: {SUPPORTED_RULES}", "params.rule")],
        )
    window_s = integration_doc["params"]["window_s"]
    entries = integration_doc["entries"]

    if blank is None:
        # 没有空白材料：不做扣除，但必须留下明确诊断，且不伪装成功
        out_entries = []
        for e in entries:
            ee = dict(e)
            ee.update({"q_blank_uJ": None, "q_net_uJ": None,
                       "blank_ordinal": None,
                       "blank_status": "blank_material_missing"})
            ee["diagnostics"] = ee["diagnostics"] + [Diagnostic(
                "blank_missing", "空白曲线未提供，无法说明稀释热对应规则",
                f"injection #{e['ordinal']}", severity="warning").to_dict()]
            out_entries.append(ee)
        return _doc(integration_doc, None, rule, None, out_entries,
                    [{"rule": rule, "matched": 0, "total": len(entries)}])

    if "blank_schedule" not in blank["parsed"]:
        # 空白材料没有自身注射时序 → 所有对应缺失（这是“空白对应缺失”情形之一）
        out_entries = []
        for e in entries:
            ee = dict(e)
            ee.update({"q_blank_uJ": None, "q_net_uJ": None,
                       "blank_ordinal": None,
                       "blank_status": "blank_schedule_missing"})
            ee["diagnostics"] = ee["diagnostics"] + [Diagnostic(
                "blank_schedule_missing",
                "空白材料缺少注射时序，无法建立对应",
                f"injection #{e['ordinal']}", severity="warning").to_dict()]
            out_entries.append(ee)
        return _doc(integration_doc, blank["id"], rule, None, out_entries,
                    [{"rule": rule, "matched": 0, "total": len(entries)}])

    blank_int = integrate_injections(
        blank["parsed"],
        blank_baseline["parsed"] if blank_baseline else None,
        {"id": "blank_schedule_ref",
         "schedule": blank["parsed"]["blank_schedule"]},
        {"window_s": window_s, "discard_ordinals": []},
    )
    by_index = {b["ordinal"]: b for b in blank_int["entries"]}
    matched = 0
    out_entries = []
    for e in entries:
        ee = dict(e)
        ordinal = e["ordinal"]
        b = by_index.get(ordinal)
        if b is None:
            ee.update({"q_blank_uJ": None, "q_net_uJ": None,
                       "blank_ordinal": None, "blank_status": "blank_missing"})
            ee["diagnostics"] = ee["diagnostics"] + [Diagnostic(
                "blank_missing",
                f"空白中找不到与第 {ordinal} 注对应的注射（规则 {rule}）",
                f"injection #{ordinal}", severity="warning").to_dict()]
        elif b.get("q_gross_uJ") is None or b.get("fatal"):
            ee.update({"q_blank_uJ": None, "q_net_uJ": None,
                       "blank_ordinal": ordinal,
                       "blank_status": "blank_unintegrable"})
            ee["diagnostics"] = ee["diagnostics"] + [Diagnostic(
                "blank_unintegrable", "对应空白注无法积分（缺口覆盖整个窗等）",
                f"blank injection #{ordinal}", severity="warning").to_dict()]
        elif e.get("q_gross_uJ") is None or e.get("fatal"):
            ee.update({"q_blank_uJ": b["q_gross_uJ"], "q_net_uJ": None,
                       "blank_ordinal": ordinal,
                       "blank_status": "sample_unintegrable"})
        else:
            scale = e["injection_volume_uL"] / b["injection_volume_uL"]
            q_blank = b["q_gross_uJ"] * scale
            q_net = e["q_gross_uJ"] - q_blank
            ee.update({"q_blank_uJ": round(q_blank, 9),
                       "q_net_uJ": round(q_net, 9),
                       "blank_ordinal": ordinal,
                       "blank_scale_volume": round(scale, 9),
                       "blank_status": "matched"})
            matched += 1
        out_entries.append(ee)
    summary = {
        "rule": rule, "rule_version": DILUTION_RULE_VERSION,
        "matched": matched, "total": len(entries),
        "blank_material": blank["id"],
        "blank_integration": blank_int,
        "input_digest": digest({
            "integration": integration_doc.get("artifact_id"),
            "blank": blank["id"], "rule": rule,
            "window_s": window_s,
        }),
    }
    return _doc(integration_doc, blank["id"], rule, blank_int,
                out_entries, summary)


def _doc(integration_doc, blank_id, rule, blank_int, entries, summary):
    return {
        "kind": "dilution",
        "rule": rule,
        "rule_version": DILUTION_RULE_VERSION,
        "time_basis": integration_doc["time_basis"],
        "energy_unit": "uJ",
        "inputs": {
            "integration_artifact": integration_doc.get("artifact_id"),
            "blank_material": blank_id,
        },
        "entries": entries,
        "blank_integration": blank_int,
        "summary": summary,
    }
