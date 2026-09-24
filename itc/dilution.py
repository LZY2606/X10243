"""空白对照 / 稀释热扣除。

对应规则必须显式声明（correspondence）：
* by_index  —— 按注射序号对应（默认）：样本第 i 注 ↔ 空白第 i 注。
* by_time   —— 按注射时刻最近邻对应（容差 time_tolerance_s）。
缺对应的注射不强行扣除，标记 blank_missing 并给出 finding（空白对应缺失）。
空白热按体积比缩放：Q_dil = Q_blank * V_sample / V_blank。
"""
from __future__ import annotations

import numpy as np

from .errors import Diagnostic, SemanticUnresolved
from .integration import canonical_samples, baseline_at, DEFAULT_POLICY, detect_gaps, _endpoint

DEFAULT_CORRESPONDENCE = {
    "rule": "by_index",
    "time_tolerance_s": 30.0,
    "volume_scale": True,
}


def integrate_blank_series(blank_material: dict, policy: dict | None = None) -> dict:
    pol = {**DEFAULT_POLICY, "discard_first": False, **(policy or {})}
    if not blank_material.get("injections"):
        raise SemanticUnresolved(
            "空白曲线缺少自己的注射时序，无法建立对应",
            [Diagnostic("blank_injections_missing",
                        "空白材料没有 @@ injections 段", "/blank/injections", None)],
        )
    pseudo_power = {
        "samples": blank_material["samples"],
        "meta": blank_material.get("meta", {}),
    }
    t, p, _ = canonical_samples(pseudo_power)
    gaps, _ = detect_gaps(t, pol["gap_factor"])
    injs = []
    for r in blank_material["injections"]["records"]:
        tv, vv = r["time"]["canonical_value"], r["volume"]["canonical_value"]
        if tv is None or vv is None:
            continue
        injs.append({"index": r["index"], "t": float(tv), "volume": float(vv)})
    injs.sort(key=lambda z: z["t"])
    # 空白曲线没有独立基线材料：用第一注之前的中位数作为仪器基线
    first_t = injs[0]["t"] if injs else t[-1]
    pre = p[t < first_t]
    base_level = float(np.median(pre)) if len(pre) >= 5 else 0.0
    y = p - base_level
    noise = float(np.std(pre)) if len(pre) >= 5 else 0.0
    band = max(pol["baseline_band_sigma"] * noise, 1e-12)
    out = []
    for pos, inj in enumerate(injs):
        next_t = injs[pos + 1]["t"] if pos + 1 < len(injs) else None
        hard_end = t[-1]
        gap_block = None
        for g in gaps:
            if g["t_start"] > inj["t"]:
                if next_t is None or g["t_start"] < next_t:
                    hard_end, gap_block = g["t_start"], g
                break
        seg_end = min(t[-1] if next_t is None else min(next_t, t[-1]), hard_end)
        seg = (t >= inj["t"]) & (t <= seg_end + 1e-12)
        t_seg, y_seg = t[seg], y[seg]
        if len(t_seg) < 2:
            out.append({"index": inj["index"], "t_inject": inj["t"], "volume_l": inj["volume"],
                        "heat_cal": None, "clipped_by_gap": bool(gap_block)})
            continue
        ei, t_end, kind = _endpoint(t_seg, y_seg, band, int(pol["keep_samples"]), next_t)
        tw, yw = t_seg[: ei + 1], y_seg[: ei + 1]
        heat_ucal = float(np.trapezoid(yw, tw)) if hasattr(np, "trapezoid") else float(np.trapz(yw, tw))
        heat = heat_ucal * 1e-6
        out.append({"index": inj["index"], "t_inject": inj["t"], "volume_l": inj["volume"],
                    "heat_cal": heat, "endpoint_kind": kind,
                    "clipped_by_gap": bool(gap_block)})
    return {"kind": "blank_integration", "injections": out, "gaps": gaps, "noise_sigma": noise}


def apply_dilution(sample_run: dict, blank_run: dict,
                   correspondence: dict | None = None) -> dict:
    corr = {**DEFAULT_CORRESPONDENCE, **(correspondence or {})}
    blank_by_index = {b["index"]: b for b in blank_run["injections"]}
    blank_sorted = sorted(blank_run["injections"], key=lambda b: b["t_inject"])
    rows: list[dict] = []
    findings: list[dict] = []

    for e in sample_run["injections"]:
        row = {
            "injection_id": e["injection_id"],
            "index": e["index"],
            "discarded": e["discarded"],
            "sample_heat_cal": e["heat_cal"],
            "sample_volume_l": e["volume_l"],
            "blank_heat_cal": None,
            "blank_index": None,
            "net_heat_cal": None,
            "subtracted": False,
            "correspondence_rule": corr["rule"],
            "findings": [],
        }
        if e["discarded"]:
            row["findings"].append({
                "finding_id": f"{e['injection_id']}::discarded", "severity": "info",
                "code": "first_injection_discarded",
                "message": "首注舍弃，不参与稀释扣除",
                "loc": f"/injections/row/0", "injection_index": e["index"]})
            rows.append(row)
            continue
        if not e["integrable"]:
            row["findings"].append({
                "finding_id": f"{e['injection_id']}::not_integrable", "severity": "error",
                "code": "not_integrable", "message": "样本注射本身不可积分，跳过扣除",
                "loc": f"/injections/{e['index']}", "injection_index": e["index"]})
            rows.append(row)
            continue

        match = None
        if corr["rule"] == "by_index":
            match = blank_by_index.get(e["index"])
        elif corr["rule"] == "by_time":
            tol = corr["time_tolerance_s"]
            cands = [b for b in blank_sorted if abs(b["t_inject"] - e["t_inject"]) <= tol]
            if cands:
                match = min(cands, key=lambda b: abs(b["t_inject"] - e["t_inject"]))
        else:
            raise SemanticUnresolved(
                f"未知对应规则 {corr['rule']!r}",
                [Diagnostic("unknown_correspondence", "允许 by_index / by_time",
                            "/correspondence/rule", corr["rule"])],
            )

        if match is None or match.get("heat_cal") is None:
            row["findings"].append({
                "finding_id": f"{e['injection_id']}::blank_missing", "severity": "warning",
                "code": "blank_missing",
                "message": f"注射 {e['index']} 找不到符合 {corr['rule']} 规则的空白积分；不扣除",
                "loc": f"/blank/injections/{e['index']}",
                "injection_index": e["index"], "rule": corr["rule"]})
            findings.append(row["findings"][-1])
            rows.append(row)
            continue

        blank_heat = match["heat_cal"]
        if corr.get("volume_scale", True) and match["volume_l"]:
            blank_heat = blank_heat * e["volume_l"] / match["volume_l"]
        row["blank_heat_cal"] = blank_heat
        row["blank_index"] = match["index"]
        row["net_heat_cal"] = e["heat_cal"] - blank_heat
        row["subtracted"] = True
        if match.get("clipped_by_gap"):
            row["findings"].append({
                "finding_id": f"{e['injection_id']}::blank_gap", "severity": "warning",
                "code": "blank_window_clipped", "message": "对应空白积分窗曾被缺口夹断",
                "loc": f"/blank/injections/{match['index']}",
                "injection_index": e["index"], "blank_index": match["index"]})
        rows.append(row)

    return {
        "kind": "dilution_run",
        "correspondence": corr,
        "rows": rows,
        "findings": findings,
        "energy_unit": "cal",
    }
