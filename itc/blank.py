"""空白/稀释热扣除。

规则: 稀释热扣除必须说明对应规则 (rule 记录在事件与结果里);
空白对应缺失时返回带注射位置的诊断, 保留未扣除序列, 不得用空值伪装成功。
被采纳(扣除后)与未采纳(原始)序列同时保留。
"""
from __future__ import annotations

from .units import normalize, diag

RULE_VERSION = "blank/1.0"
RULES = ("by_index", "by_volume")


def _blank_entries(blank_payload):
    entries, diags = [], []
    for i, e in enumerate(blank_payload.get("entries", [])):
        heat, d1 = normalize(e["heat"], "heat", position={"blank_entry": i, "field": "heat"})
        vol, d2 = (None, None)
        if "volume" in e:
            vol, d2 = normalize(e["volume"], "volume",
                                position={"blank_entry": i, "field": "volume"})
        if d1 or d2:
            diags.extend(d for d in (d1, d2) if d)
            entries.append({"index": e.get("index", i), "heat_uJ": heat,
                            "vol_uL": vol, "ok": False})
            continue
        entries.append({"index": e.get("index", i), "heat_uJ": heat,
                        "vol_uL": vol, "ok": True})
    return entries, diags


def subtract(integrals_payload, blank_payload, rule):
    """按对应规则扣除空白。integrals_payload 是积分结果 payload。"""
    if rule.get("kind") not in RULES:
        raise ValueError(f"未知空白对应规则 {rule!r}; 必须为 {RULES} 之一")
    entries, diags = _blank_entries(blank_payload)
    by_index = {e["index"]: e for e in entries if e["ok"]}

    raw_series, corrected_series, missing = [], [], []
    for inj in integrals_payload.get("injections", []):
        pos = {"injection": inj["index"]}
        raw_series.append({
            "injection_id": inj["injection_id"], "index": inj["index"],
            "heat_uJ": inj["heat_uJ"], "status": inj["status"],
        })
        if inj["status"] != "ok":
            corrected_series.append({
                "injection_id": inj["injection_id"], "index": inj["index"],
                "status": "unavailable",
                "diagnostics": [diag("integral_unavailable",
                                     "积分不可用, 无法扣除空白", position=pos)],
            })
            continue
        match = None
        if rule["kind"] == "by_index":
            match = by_index.get(inj["index"])
        else:  # by_volume: 最近体积且不超过容差
            tol = rule.get("tolerance_uL", 0.5)
            cands = [e for e in entries
                     if e["ok"] and e["vol_uL"] is not None
                     and inj["vol_ok"] and abs(e["vol_uL"] - inj["vol_uL"]) <= tol]
            if cands:
                match = min(cands, key=lambda e: abs(e["vol_uL"] - inj["vol_uL"]))
        if match is None:
            missing.append(inj["index"])
            corrected_series.append({
                "injection_id": inj["injection_id"], "index": inj["index"],
                "status": "blank_missing",
                "diagnostics": [diag(
                    "blank_missing",
                    f"注射 {inj['injection_id']} 缺少空白对应 "
                    f"(规则 {rule['kind']}), 未扣除",
                    position=pos)],
            })
            continue
        corrected_series.append({
            "injection_id": inj["injection_id"], "index": inj["index"],
            "status": "ok",
            "heat_uJ": inj["heat_uJ"] - match["heat_uJ"],
            "blank_heat_uJ": match["heat_uJ"],
            "blank_index": match["index"],
        })

    status = "ok" if not missing else "partial"
    if missing:
        diags = diags + [diag(
            "blank_correspondence_missing",
            f"注射序号 {missing} 缺少空白对应 (规则 {rule['kind']})",
            position={"injections": missing}, severity="warning")]
    return {
        "kind": "blank_subtraction",
        "param_version": RULE_VERSION,
        "rule": rule,  # 稀释热扣除必须说明对应规则
        "units": {"heat": "µJ"},
        "status": status,
        "diagnostics": diags,
        # 被采纳与未采纳的结果同时保留
        "series": {"raw": raw_series, "blank_corrected": corrected_series},
    }
