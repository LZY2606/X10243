"""稀释热扣除：空白对应规则必须显式说明。

- correspondence = index：按注射序号严格对应；缺号 => BLANK_CORRESPONDENCE_MISSING
- correspondence = nearest_time：按时间就近对应，必须报告配对的空白序号
规则缺失 => SEMANTIC_UNRESOLVABLE，绝不用“零稀释热”静默伪装。
浓度修订不改变本层结果（只依赖功率/空白/时序/规则）。
"""
from __future__ import annotations

import numpy as np

from .errors import Diagnostic, SemanticUnresolvable
from .ids import digest, stable_deps
from .integration import integrate_all
from .sampling import build_segments


def blank_integrals(blank: dict, rules: dict) -> dict:
    """用与样品完全一致的积分规则对空白曲线逐注射积分。"""
    seg = build_segments(blank, rules["values"]["gap_factor"])
    groups: dict[int, list[tuple[float, float]]] = {}
    for s in blank["samples"]:
        if s["time"]["valid"] and s["power"]["valid"] and s["injection_index"] is not None:
            groups.setdefault(s["injection_index"], []).append(
                (s["time"]["canonical"], s["power"]["canonical"])
            )
    per: dict[int, dict] = {}
    post = rules["values"]["post_window_s"]
    pre = rules["values"]["pre_window_s"]
    for idx, pts in groups.items():
        pts.sort()
        tt = np.array([p[0] for p in pts])
        pp = np.array([p[1] for p in pts])
        t0 = float(tt[0])
        pre_vals = pp[tt <= t0 + pre]
        base = float(np.mean(pre_vals)) if len(pre_vals) else float(np.mean(pp))
        sel = (tt >= t0) & (tt <= t0 + post)
        if sel.sum() < 2:
            per[idx] = {"injection_index": idx, "valid": False, "heat_ucal": None,
                        "diagnostics": [Diagnostic(
                            "BLANK_WINDOW_TOO_SHORT", f"{blank['source']}#blank{idx}",
                            f"空白 {idx} 积分窗内有效点 {int(sel.sum())} < 2").to_dict()]}
            continue
        q_uj = float(np.trapezoid(pp[sel] - base, tt[sel]))
        per[idx] = {"injection_index": idx, "valid": True, "heat_ucal": q_uj / 4.184,
                    "time_s": t0, "diagnostics": []}
    return {"kind": "blank_integral_set", "blank_id": blank["id"], "rules_id": rules["id"], "per": per}


def apply_dilution(integral_set: dict, schedule: dict, blank: dict | None,
                   rules: dict) -> dict:
    correspondence = rules["values"].get("blank_correspondence")
    if correspondence not in ("index", "nearest_time"):
        raise SemanticUnresolvable(
            "稀释热缺少对应规则，无法扣除",
            [Diagnostic("BLANK_RULE_MISSING", "rules.blank_correspondence",
                        f"未知对应规则 {correspondence!r}")],
        )

    blank_set = blank_integrals(blank, rules) if blank is not None else None
    net_results: list[dict] = []
    judgements: list[dict] = []

    sample_times = {
        inj["index"]: inj["time"]["canonical"]
        for inj in schedule["injections"] if inj["time"]["valid"]
    }

    for r in integral_set["results"]:
        idx = r["injection_index"]
        matched: int | None = None
        diag: list[dict] = []
        status = "matched"
        if blank_set is None:
            status = "no_blank_curve"
            diag.append(Diagnostic(
                "BLANK_CORRESPONDENCE_MISSING",
                f"blank#inj{idx}",
                "未导入空白曲线；必须说明稀释热处理（保留 raw heat 或显式忽略）",
            ).to_dict())
        elif correspondence == "index":
            cand = blank_set["per"].get(idx)
            if cand is None or not cand.get("valid"):
                status = "missing_index"
                diag.append(Diagnostic(
                    "BLANK_CORRESPONDENCE_MISSING",
                    f"blank#inj{idx}",
                    f"空白曲线缺少与样品第 {idx} 注同序号的可积分峰（index 对应规则）",
                ).to_dict())
            else:
                matched = idx
        else:  # nearest_time
            valid_blank = [(b["injection_index"], b["time_s"])
                           for b in blank_set["per"].values() if b.get("valid")]
            if not valid_blank:
                status = "missing_index"
                diag.append(Diagnostic(
                    "BLANK_CORRESPONDENCE_MISSING", f"blank#inj{idx}",
                    "空白曲线没有任何可积分峰（nearest_time 对应规则）").to_dict())
            else:
                t = sample_times.get(idx)
                matched = min(valid_blank, key=lambda kv: abs(kv[1] - t))[0]

        q_sample = r["heat_ucal"]
        q_blank_scaled = None
        q_net = None
        if status == "matched":
            qb = blank_set["per"][matched]["heat_ucal"]
            if r.get("volume_ul"):
                # 空白注射体积默认与样品相等；若注入体积不同，按体积线性缩放稀释热
                q_blank_scaled = qb
            q_net = q_sample - q_blank_scaled if q_sample is not None else None

        deps = {
            "integral": r["id"],
            "rules": rules["id"],
            "blank": blank["id"] if blank else None,
            "correspondence": correspondence,
            "matched_blank_index": matched,
        }
        nid = digest("net", stable_deps(deps))
        net_results.append({
            "id": nid,
            "kind": "net_heat",
            "injection_index": idx,
            "valid": r["valid"] and status == "matched",
            "sample_heat_ucal": q_sample,
            "blank_heat_ucal": blank_set["per"][matched]["heat_ucal"] if status == "matched" else None,
            "blank_scaled_ucal": q_blank_scaled,
            "net_heat_ucal": q_net,
            "correspondence_rule": correspondence,
            "matched_blank_index": matched,
            "status": status,
            "volume_ul": r.get("volume_ul"),
            "diagnostics": diag,
            "deps": deps,
        })
        if status != "matched":
            judgements.append({
                "type": "DILUTION_JUDGEMENT",
                "injection_index": idx,
                "rule": "BLANK_CORRESPONDENCE_MISSING",
                "correspondence": correspondence,
                "detail": {"status": status, "matched_blank_index": None},
                "rule_version": rules["id"],
            })
        elif correspondence == "nearest_time":
            judgements.append({
                "type": "DILUTION_JUDGEMENT",
                "injection_index": idx,
                "rule": "BLANK_MATCH_NEAREST_TIME",
                "correspondence": correspondence,
                "detail": {"status": status, "matched_blank_index": matched},
                "rule_version": rules["id"],
            })

    return {
        "kind": "net_heat_set",
        "results": net_results,
        "events": judgements,
        "correspondence_rule": correspondence,
        "deps": {
            "integral_set_input_hash": digest("integset", [[x["id"]] for x in integral_set["results"]]),
            "blank": blank["id"] if blank else None,
            "rules": rules["id"],
        },
    }
