"""计算编排：给定某一事件点的材料版本与决定集合，重放派生 DAG。

节点为内容寻址：输入摘要（原始载荷哈希）+ 规则版本 + 父节点 id 决定节点 id。
因此等价输入总是复用同一节点、顺序稳定；浓度修订只改变归一化/拟合链，
原始积分节点身份与值不变。
"""
from __future__ import annotations

from typing import Any

from . import materials as M
from .blank import match_and_subtract
from .constants import (DEC_BLANK_RULE, DEC_DISCARD_FIRST,
                        DEC_DISCARD_INJECTIONS, DEC_ENDPOINT_OVERRIDE,
                        DEC_POLICY, BLANK_RULE_INDEX, NODE_FIT,
                        NODE_ISOTHERM, NODE_NORMALIZED, NODE_RAW_HEAT,
                        NODE_WINDOW, STATUS_DISCARDED)
from .fitter import Budget, fit_dataset
from .hashing import digest
from .integration import integrate_all
from .normalize import normalize
from .windows import compute_windows


def active_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """决定只追加；同键后到覆盖先到，并记录冲突。"""
    order: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for d in decisions:
        key = d["key"]
        if key in order:
            conflicts.append({"key": key,
                              "superseded_event": order[key]["event_id"],
                              "by_event": d["event_id"]})
        order[key] = d
    return {"by_key": order, "conflicts": conflicts}


def recompute(materials: dict[str, dict[str, Any]],
              decisions: list[dict[str, Any]],
              budget: Budget | None = None) -> dict[str, Any]:
    """materials: slot -> {material_id, version, payload, raw_digest}。"""
    act = active_decisions(decisions)
    by_key = act["by_key"]

    def pol(name: str, default: Any) -> Any:
        d = by_key.get(f"policy:{name}")
        return d["value"] if d else default

    blank_rule = by_key.get(DEC_BLANK_RULE, {}).get(
        "value", BLANK_RULE_INDEX)
    discarded = set(by_key.get(DEC_DISCARD_INJECTIONS, {}).get(
        "value", []))
    if by_key.get(DEC_DISCARD_FIRST, {}).get("value", True):
        discarded.add(1)
    overrides = {int(k.split(":")[1]): v for k, v in by_key.items()
                 if k.startswith("endpoint:")}

    summaries = {slot: {
        "material_id": m.get("material_id"),
        "version": m.get("version"),
        "raw_digest": m.get("raw_digest"),
    } for slot, m in materials.items()}

    nodes: dict[str, dict[str, Any]] = {}

    power_mat = materials.get("power")
    base_mat = materials.get("baseline")
    inj_mat = materials.get("injections")
    blank_mat = materials.get("blank")
    temp_mat = materials.get("temperature")
    vol_mat = materials.get("cell_volume")
    conc_mat = materials.get("concentrations")

    power_view = M.power_view(power_mat) if power_mat else None
    injections = M.injections_view(inj_mat) if inj_mat else []

    # 1) 积分窗
    windows = (compute_windows(injections, power_view, overrides)
               if power_view else [])
    win_parents = ["power", "injections"] + ["endpoint_decisions"]
    win_input = {"summaries": {k: summaries.get(k) for k in
                               ("power", "injections")},
                 "overrides": overrides,
                 "discarded": sorted(discarded)}
    win_id = "win_" + digest(["windows", win_input])
    nodes[win_id] = {"id": win_id, "node_type": NODE_WINDOW,
                     "parents": win_parents, "input_summary": win_input,
                     "value": windows}

    # 2) 原始热量（对空白缺失、浓度修订免疫）
    raw_heats = integrate_all(power_mat, base_mat, windows, power_view,
                              discarded) if power_view else []
    raw_input = {"window_node": win_id,
                 "baseline": summaries.get("baseline")}
    raw_id = "raw_" + digest(["raw_heat", raw_input,
                              [round(h["value"], 10) if h.get("value")
                               is not None else None for h in raw_heats]])
    nodes[raw_id] = {"id": raw_id, "node_type": NODE_RAW_HEAT,
                     "parents": [win_id, "baseline"],
                     "input_summary": raw_input, "value": raw_heats}

    # 3) 空白对应 / 稀释热
    blank_map = match_and_subtract(raw_heats, injections, blank_rule,
                                   blank_mat, base_mat)
    bm_id = "bm_" + digest(["blank_map", blank_rule,
                            summaries.get("blank"), raw_id])
    nodes[bm_id] = {"id": bm_id, "node_type": "blank_map",
                    "parents": [raw_id, "blank", "baseline"],
                    "input_summary": {"rule": blank_rule,
                                      "blank": summaries.get("blank")},
                    "value": blank_map}

    # 4) 归一化热（依赖浓度、温度、池体积）
    c_syringe = (M.concentration_view(conc_mat, "syringe")["value"]
                 if conc_mat else None)
    c_cell = (M.concentration_view(conc_mat, "cell")["value"]
              if conc_mat else None)
    v0 = M.scalar_view(vol_mat, "volume", "cell_volume")["value"] \
        if vol_mat else None
    temp = M.scalar_view(temp_mat, "temperature", "temperature")["value"] \
        if temp_mat else None
    norm = normalize(blank_map, injections, c_syringe, c_cell, v0,
                     discarded, temp)
    norm_input = {"blank_map": bm_id,
                  "concentrations": summaries.get("concentrations"),
                  "temperature": summaries.get("temperature"),
                  "cell_volume": summaries.get("cell_volume"),
                  "discarded": sorted(discarded)}
    norm_id = "norm_" + digest(["normalized", norm_input])
    nodes[norm_id] = {"id": norm_id, "node_type": NODE_NORMALIZED,
                      "parents": [bm_id, "concentrations", "temperature",
                                  "cell_volume"],
                      "input_summary": norm_input, "value": norm}

    # 5) 等温线坐标（薄节点，身份随归一化）
    iso_id = "iso_" + digest(["isotherm", norm_id])
    nodes[iso_id] = {"id": iso_id, "node_type": NODE_ISOTHERM,
                     "parents": [norm_id],
                     "input_summary": {"normalized": norm_id},
                     "value": {"points": norm["points"]}}

    # 6) 拟合候选
    fit = fit_dataset(norm["points"], v0, c_cell, c_syringe, temp,
                      tie_tol=float(pol("tie_tol", 2.0)),
                      budget=budget)
    fit_input = {"isotherm": iso_id,
                 "tie_tol": pol("tie_tol", 2.0),
                 "rule_versions": {"fit": "fit-lm-bic-v1"}}
    fit_id = "fit_" + digest(["fit", fit_input,
                              [(c["model"], round(c["bic"], 6))
                               for c in fit.get("candidates", [])]])
    nodes[fit_id] = {"id": fit_id, "node_type": NODE_FIT,
                     "parents": [iso_id],
                     "input_summary": fit_input, "value": fit}

    return {
        "nodes": nodes,
        "order": [win_id, raw_id, bm_id, norm_id, iso_id, fit_id],
        "summaries": summaries,
        "discarded": sorted(discarded),
        "blank_rule": blank_rule,
        "conflicts": act["conflicts"],
        "scalars": {"c_syringe": c_syringe, "c_cell": c_cell,
                    "v0": v0, "temperature_k": temp},
    }
