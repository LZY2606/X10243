"""逐注积分：注射后峰可以重叠；积分窗不跨越数据缺口。

规则（算法版本 alg:integration:1，参数版本随事件记录）：
  * 基线是独立版本化材料，按时间插值后扣除；
  * 积分窗 [t_inj - pre, t_inj + post] 允许互相重叠；
  * 窗口与任何数据缺口相交 => 拒绝积分（NO_SIGNAL/error），绝不跨缺口插值或拼接；
  * 端点可以人工调整，调整后落在缺口/范围外同样拒绝；
  * 首注（first injection）按 ITC 惯例标记 discarded，不进入拟合，但积分仍照算留档；
  * 积分能量单位规范为 µcal（原始 µcal/s × s）。
"""

import numpy as np

from .canonical import canonical_json, short_id
from .materials import trace_arrays, injections as inj_rows

ALG_VERSION = "integration:1"


def baseline_on_grid(power_mat, baseline_mat):
    """基线扣减：基线曲线独立版本化，逐点对齐 / 线性插值（不跨缺口）。"""
    t, p, valid, _ = trace_arrays(power_mat)
    if baseline_mat is None:
        corrected = np.where(valid, p, np.nan)
        return t, corrected, valid, []
    bt, bp, bvalid, _ = trace_arrays(baseline_mat)
    corrected = np.full_like(p, np.nan, dtype=float)
    notes = []
    bgood = bvalid & np.isfinite(bt) & np.isfinite(bp)
    if not bgood.any():
        from .errors import SemanticNoSolution
        raise SemanticNoSolution(
            "基线曲线没有任何有效样本，无法扣减", code="EMPTY_BASELINE")
    order = np.argsort(bt[bgood])
    bxs, bys = bt[bgood][order], bp[bgood][order]
    if len(bxs) == 1:
        interp = np.full(t.shape, float(bys[0]))
        notes.append({"code": "CONSTANT_BASELINE",
                      "message": "基线只有一个有效点，按常量扣除"})
    else:
        interp = np.interp(t, bxs, bys,
                           left=float(bys[0]), right=float(bys[-1]))
    for i in range(len(t)):
        if valid[i]:
            corrected[i] = p[i] - interp[i]
    return t, corrected, valid, notes


def _windows_overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def integrate(power_mat, injection_mat, baseline_mat=None,
              *, endpoint_overrides=None, discard_first=True,
              first_injection_no=1):
    """返回 {inputs, windows[], diagnostics, algorithm}。

    每个 window 都带稳定 uid；失败窗口 status=rejected，保留请求与诊断，
    绝不用零/空值伪装成功。
    """
    endpoint_overrides = endpoint_overrides or {}
    t, corrected, valid, gaps = baseline_on_grid(power_mat, baseline_mat)
    n = len(t)
    times_valid = t[valid]
    t_min, t_max = (float(times_valid.min()), float(times_valid.max())) \
        if len(times_valid) else (None, None)

    rows = inj_rows(injection_mat)
    windows = []
    diagnostics = []

    for row in rows:
        no = row["injectionNo"]
        req_pre = row["winPre"]
        req_post = row["winPost"]
        t_inj = row["tInj"]
        raw = {"injectionNo": no, "tInj": t_inj, "volume": row["volume"],
               "sourceRow": row["sourceRow"], "requested":
                   {"pre": req_pre, "post": req_post},
               "problems": row["problems"]}
        if t_inj is None or req_pre is None or req_post is None:
            raw["status"] = "rejected"
            raw["energy"] = None
            raw["diagnostics"] = [{
                "code": "MISSING_WINDOW_INPUT",
                "severity": "error",
                "message": "注射时间或窗口端点无法规范化，拒绝积分",
                "path": f"row{row['sourceRow']}",
                "related": [f"injection:{no}"]}]
            diagnostics.extend(raw["diagnostics"])
            windows.append(_finalize_window(raw, power_mat, injection_mat,
                                            baseline_mat, endpoint_overrides))
            continue

        ovr = endpoint_overrides.get(no, {})
        pre = ovr.get("pre", req_pre)
        post = ovr.get("post", req_post)
        start, end = t_inj - pre, t_inj + post
        win = {**raw, "start": start, "end": end,
               "endpointsAdjusted": bool(ovr),
               "adjustment": ovr or None,
               "energy": None, "samples": 0}
        probs = []

        if pre < 0 or post < 0 or end <= start:
            probs.append({"code": "BAD_ENDPOINT", "severity": "error",
                          "message": f"积分窗非法：[{start}, {end}]",
                          "related": [f"injection:{no}"]})
        if t_min is not None and (start < t_min or end > t_max):
            probs.append({"code": "ENDPOINT_OUT_OF_RANGE", "severity": "error",
                          "message": (f"窗口 [{start:.3f},{end:.3f}] 超出有效采样范围 "
                                      f"[{t_min:.3f},{t_max:.3f}]，拒绝（不裁剪成伪结果）"),
                          "related": [f"injection:{no}"]})
        # 缺口检测：窗口区间内是否含缺口样本，且窗口并非整体落在缺口中。
        gap_hits = []
        for a, b in gaps:
            g0 = t[a - 1] if a > 0 and valid[a - 1] else (
                t[b + 1] if b + 1 < n and valid[b + 1] else -np.inf)
            g1 = t[b + 1] if b + 1 < n and valid[b + 1] else (
                t[a - 1] if a > 0 and valid[a - 1] else np.inf)
            if start < g1 and g0 < end:
                gap_hits.append((a, b, float(g0), float(g1)))
        if gap_hits:
            probs.append({
                "code": "WINDOW_CROSSES_GAP", "severity": "error",
                "message": (f"注射 {no} 积分窗与 {len(gap_hits)} 处数据缺口相交，"
                            "积分窗不得跨越数据缺口；未做插值/拼接"),
                "related": [f"injection:{no}"],
                "gaps": [{"sampleStart": a, "sampleEnd": b,
                          "tStart": None if not np.isfinite(g0) else g0,
                          "tEnd": None if not np.isfinite(g1) else g1}
                         for a, b, g0, g1 in gap_hits]})

        if not any(p["severity"] == "error" for p in probs):
            mask = valid & (t >= start) & (t <= end)
            idx = np.where(mask)[0]
            seg_t, seg_p = t[idx], corrected[idx]
            # 窗口内不得再藏缺口（被前面的范围检测兜底，双保险）
            if len(seg_t) < 2:
                probs.append({"code": "NO_SIGNAL", "severity": "error",
                              "message": f"窗口内有效样本不足：{len(seg_t)}",
                              "related": [f"injection:{no}"]})
            else:
                _trap = getattr(np, "trapezoid", np.trapz)
                energy = float(_trap(seg_p, seg_t))  # µcal
                win["energy"] = energy
                win["samples"] = int(len(seg_t))
                win["segmentIndexStart"] = int(idx[0])
                win["segmentIndexEnd"] = int(idx[-1])
                win["status"] = "integrated"

        if any(p["severity"] == "error" for p in probs):
            win["status"] = "rejected"
        win["diagnostics"] = probs
        diagnostics.extend(probs)
        win["discarded"] = discard_first and no == first_injection_no
        if win["discarded"] and win["status"] == "integrated":
            win["discardReason"] = "FIRST_INJECTION_DISCARDED"
        windows.append(_finalize_window(win, power_mat, injection_mat,
                                        baseline_mat, endpoint_overrides))

    # 重叠：峰允许重叠 => 只记录结构关系，不作为错误
    overlap_notes = []
    valids_w = [w for w in windows if "start" in w]
    for i in range(len(valids_w)):
        for j in range(i + 1, len(valids_w)):
            a, b = valids_w[i], valids_w[j]
            if _windows_overlap(a, b):
                note = {
                    "code": "OVERLAPPING_PEAKS", "severity": "info",
                    "message": (f"注射 {a['injectionNo']} 与 {b['injectionNo']} "
                                "的积分窗重叠：允许重叠，两窗各自积分"),
                    "related": [f"injection:{a['injectionNo']}",
                                f"injection:{b['injectionNo']}"]}
                overlap_notes.append(note)
                a.setdefault("overlapsWith", []).append(b["injectionNo"])
                b.setdefault("overlapsWith", []).append(a["injectionNo"])

    inputs = _input_summary(power_mat, injection_mat, baseline_mat,
                            endpoint_overrides)
    return {"algorithm": ALG_VERSION, "inputs": inputs,
            "windows": windows,
            "diagnostics": diagnostics + overlap_notes,
            "overlaps": overlap_notes}


def _finalize_window(win, power_mat, injection_mat, baseline_mat, overrides):
    basis = {"injectionNo": win["injectionNo"],
             "tInj": win.get("tInj"), "start": win.get("start"),
             "end": win.get("end"),
             "status": win.get("status"),
             "power": power_mat["contentHash"],
             "schedule": injection_mat["contentHash"],
             "baseline": baseline_mat["contentHash"] if baseline_mat else None,
             "adjustment": win.get("adjustment"),
             "alg": ALG_VERSION}
    win["uid"] = short_id(basis, "win")
    win["identityBasis"] = basis
    return win


def _input_summary(power_mat, injection_mat, baseline_mat, overrides):
    return {
        "powerTrace": {"id": power_mat["id"],
                       "hash": power_mat["contentHash"]},
        "injectionSchedule": {"id": injection_mat["id"],
                              "hash": injection_mat["contentHash"]},
        "baselineTrace": ({"id": baseline_mat["id"],
                           "hash": baseline_mat["contentHash"]}
                          if baseline_mat else None),
        "endpointOverrides": overrides or {},
        "algorithm": ALG_VERSION,
    }
