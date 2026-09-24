"""归一化热：扣除空白与池置换修正（normalization-wiseman-v1）。

每针注入摩尔 n_i = C_syringe * dV_i；
池内大分子浓度随置换衰减 C0_i = C_cell * V0 / (V0 + V_{i-1})；
置换修正：Q_i,corr = Q_net,i + dV_i/V0 * (Q_net,1+...+Q_net,i-1)/2。
归一化 q_i = Q_i,corr / n_i（cal/mol，报告 kcal/mol）。
浓度修订只作用于本层及模型；原始积分节点不被改写。
"""
from __future__ import annotations

from typing import Any

from .constants import (STATUS_BLANK_MISSING, STATUS_BLOCKED_GAP,
                        STATUS_DISCARDED, STATUS_OK,
                        NORMALIZATION_RULE_VERSION)


def normalize(blank_map: dict[str, Any], injections: list[dict[str, Any]],
              c_syringe: float | None, c_cell: float | None,
              v0: float | None, discarded: set[int],
              temperature_k: float | None) -> dict[str, Any]:
    missing_diag = []
    if c_syringe is None:
        missing_diag.append("syringe_concentration")
    if c_cell is None:
        missing_diag.append("cell_concentration")
    if v0 is None:
        missing_diag.append("cell_volume")
    entries = blank_map["entries"]
    points: list[dict[str, Any]] = []
    cumulative_volume = 0.0
    cumulative_heat = 0.0
    by_ord = {inj["ordinal"]: inj for inj in injections}

    for e in entries:
        ordinal = e["ordinal"]
        inj = by_ord.get(ordinal, {})
        dv = inj.get("volume")
        base = {
            "ordinal": ordinal,
            "label": e["label"],
            "dv": dv,
            "dv_unit": "L",
            "pre_injection_volume": cumulative_volume,
            "cumulative_volume": cumulative_volume + (dv or 0.0),
            "rule_version": NORMALIZATION_RULE_VERSION,
        }
        if missing_diag or dv is None or c_syringe is None or v0 is None:
            base.update(status="blocked:missing_input",
                        missing=missing_diag or ["injection_volume"],
                        q_cal_mol=None)
            points.append(base)
            continue
        net = e.get("net_heat")
        if e.get("status") in (STATUS_BLOCKED_GAP,):
            base.update(status=STATUS_BLOCKED_GAP, q_cal_mol=None,
                        reason=e.get("reason"))
            points.append(base)
            continue
        if net is None:
            base.update(status=e.get("status", "blocked"),
                        q_cal_mol=None, reason=e.get("reason"))
            points.append(base)
            continue
        displacement_correction = (dv / v0) * cumulative_heat
        q_corr = net + displacement_correction
        moles = c_syringe * dv
        q_cal_mol = (q_corr * 1e-6) / moles  # µcal -> cal
        # 与模型同构的两个坐标：
        # ratio_x_i = 累计注入体积比（= M_i/(C_cell V0) 尺度）
        # phi_i     = 池内累计热量密度（cal/L 池），供 Wiseman 差分
        phi = ((cumulative_heat + net) * 1e-6 / v0
               if e.get("status") == STATUS_OK else None)
        phi_prev = (cumulative_heat * 1e-6 / v0)
        ratio_x = (c_syringe * cumulative_volume
                   / (c_cell * v0)) if c_cell else None
        ratio_x_after = (c_syringe * (cumulative_volume + dv)
                         / (c_cell * v0)) if c_cell else None
        c_cell_now = (c_cell * v0 / (v0 + cumulative_volume)
                      if c_cell is not None else None)
        base.update(
            status=STATUS_DISCARDED if ordinal in discarded else STATUS_OK,
            net_heat=net, dilution_heat=e.get("dilution_heat"),
            displacement_correction=displacement_correction,
            q_corr=q_corr, injected_moles=moles,
            cell_concentration_now=c_cell_now,
            q_cal_mol=q_cal_mol,
            q_kcal_mol=q_cal_mol / 1000.0,
            ratio_x=ratio_x,
            ratio_x_after=ratio_x_after,
            phi=phi,
            phi_prev=phi_prev,
            blank_status=e.get("status"),
            participates=ordinal not in discarded,
            blank_missing=e.get("status") == STATUS_BLANK_MISSING,
        )
        if ordinal not in discarded:
            cumulative_heat += net if e.get("status") == STATUS_OK else 0.0
        cumulative_volume += dv
        points.append(base)

    ratio = []
    usable = [p for p in points if p.get("participates")
              and p.get("status") == STATUS_OK
              and not p.get("blank_missing")]
    total_v = sum(p["dv"] or 0.0 for p in points)
    for p in usable:
        if c_cell is None or not v0:
            continue
        denom = c_cell * v0
        ratio.append((p["ordinal"], c_syringe * (p["cumulative_volume"])
                      / denom if denom else None))
    return {
        "node_type": "normalized_heat",
        "rule_version": NORMALIZATION_RULE_VERSION,
        "units": {"heat": "kcal/mol", "volume": "L",
                  "concentration": "M", "temperature": "K"},
        "temperature_k": temperature_k,
        "points": points,
        "ratio": ratio,
    }
