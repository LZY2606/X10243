"""稀释热（空白）扣除层。

扣除 *必须说明对应规则*，否则不产生被扣除的值：
* rule=ordinal       样品第 i 注 ↔ 空白第 i 注（按出现顺序）
* rule=volume        按注射体积最接近匹配（确定性并列处理：体积相同取序号较小者）
* rule=none          不扣除

对应缺失时：corrected_heat 保留 null、原始积分保留、状态 blank_unmatched，
绝不用 0 或猜测值伪装成“已扣除”。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SCALE_TOL = 1e-9


def subtract_blank(sample_integrals: List[Dict[str, Any]],
                   blank_integrals: List[Dict[str, Any]],
                   rule: str,
                   volume_scale: bool = True) -> Dict[str, Any]:
    if rule not in {"ordinal", "volume", "none"}:
        raise ValueError(f"未知空白对应规则: {rule}")
    out_rows: List[Dict[str, Any]] = []
    used_blank: set[int] = set()

    blank_valid = [b for b in blank_integrals if b.get("valid")]

    for row in sample_integrals:
        out = dict(row)
        matched: Optional[Dict[str, Any]] = None
        match_kind = None
        if rule == "none":
            match_kind = "no_rule"
        elif not row.get("valid"):
            match_kind = "sample_invalid"
        else:
            if rule == "ordinal":
                # 按 ordinal 字段对齐
                for b in blank_valid:
                    if b["ordinal"] == row["ordinal"]:
                        matched, match_kind = b, "ordinal"
                        break
                if matched is None:
                    match_kind = "ordinal_missing"
            elif rule == "volume":
                best = None
                best_diff = None
                for b in blank_valid:
                    diff = abs(b["volume_l"] - row["volume_l"])
                    if best is None or diff < best_diff - SCALE_TOL or \
                            (abs(diff - (best_diff or 0.0)) <= SCALE_TOL and b["ordinal"] < best["ordinal"]):
                        best, best_diff = b, diff
                if best is not None:
                    matched, match_kind = best, "volume_nearest"
            if matched is not None:
                used_blank.add(matched["ordinal"])

        out["blank_match"] = None if matched is None else {
            "blank_ordinal": matched["ordinal"],
            "rule": rule,
            "kind": match_kind,
        }
        q_corr = row.get("integral_corrected_cal")
        if matched is not None:
            scale = row["volume_l"] / matched["volume_l"] if volume_scale and matched["volume_l"] else 1.0
            out["blank_heat_cal"] = matched["integral_corrected_cal"] * scale
            out["dilution_corrected_cal"] = q_corr - out["blank_heat_cal"]
            out["status"] = list(row.get("status", [])) + [f"blank_{match_kind}"]
        else:
            out["blank_heat_cal"] = None
            out["dilution_corrected_cal"] = None
            if rule != "none":
                out["status"] = list(row.get("status", [])) + ["blank_unmatched"]
                out.setdefault("diagnostics", []).append({
                    "code": "blank_unmatched",
                    "message": f"规则 {rule} 下第 {row['ordinal']} 注无空白对应，不做稀释扣除",
                    "location": f"injection#{row['ordinal']}",
                })
            else:
                out["status"] = list(row.get("status", [])) + ["blank_rule_none"]
        out_rows.append(out)

    unmatched = [b["ordinal"] for b in blank_valid if b["ordinal"] not in used_blank]
    return {"rule": rule, "volume_scale": volume_scale,
            "rows": out_rows, "unused_blank_ordinals": unmatched}


def normalized_heats(subtracted_rows: List[Dict[str, Any]],
                     syringe_conc_m: float,
                     cell_volume_l: float) -> List[Dict[str, Any]]:
    """q_i = ΔQ_i / (V_inj * C_syringe)。扣除缺失或被舍弃注射 → null 但保留行。"""
    out = []
    for r in subtracted_rows:
        dq = r.get("dilution_corrected_cal")
        moles = r["volume_l"] * syringe_conc_m
        row = {
            "ordinal": r["ordinal"],
            "x": None,
            "q_cal_mol": None,
            "dilution_corrected_cal": dq,
            "moles_injected_mol": moles,
            "used_for_fit": True,
            "status": list(r.get("status", [])),
            "blank_match": r.get("blank_match"),
        }
        if "first_injection_discarded" in r.get("status", []):
            row["used_for_fit"] = False
            row["status"].append("excluded_first_injection")
        if dq is None:
            row["used_for_fit"] = False
            row["status"].append("excluded_no_blank_match")
        elif moles > 0 and cell_volume_l > 0:
            row["q_cal_mol"] = dq / moles
        else:
            row["used_for_fit"] = False
            row["status"].append("excluded_bad_moles")
        out.append(row)
    return out
