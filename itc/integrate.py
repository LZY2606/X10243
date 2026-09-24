"""注射积分引擎。

规则（机器可读事件在管线层生成）：
- 注射后峰可重叠：窗口在下次注射处相接是合法的（abuts_next）。
- 积分窗不跨越数据缺口：窗口在缺口起点截断并附诊断；落在缺口内的注射判 invalid。
- 基线按段提供；每次注射记录实际依赖的基线段 id，供局部失效判定。
"""
from __future__ import annotations

import numpy as np

from .domain import Diagnostic

PARAM_VERSION = "integ-v1"

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def baseline_at(segments: list[dict], t: float):
    for seg in segments:
        if seg["t_start"] <= t <= seg["t_end"]:
            return seg["offset_uJ_s"] + seg["drift_uJ_s_per_s"] * (t - seg["t_ref"]), seg["id"]
    return None, None


def integrate(trace: dict, injections: list[dict], segments: list[dict], window_s: float) -> dict:
    """trace: {"times": [...], "power": [...], "gaps": [[s, e], ...]}（已规范化为 s 与 µJ/s）
    injections: [{"time_s": ...}, ...]；返回每次注射的积分与诊断。"""
    times = np.asarray(trace["times"], dtype=float)
    power = np.asarray(trace["power"], dtype=float)
    gaps = trace.get("gaps", [])
    results: list[dict] = []
    diagnostics: list[dict] = []
    for idx, inj in enumerate(injections):
        t0 = float(inj["time_s"])
        loc = f"injections[{idx}]"
        diags: list[dict] = []
        in_gap = next((g for g in gaps if g[0] <= t0 < g[1]), None)
        if in_gap:
            d = Diagnostic("injection_in_gap", loc,
                           f"注射 {idx} 落在数据缺口 [{in_gap[0]},{in_gap[1]}] s 内，无法积分").to_dict()
            diagnostics.append(d)
            results.append({"index": idx, "time_s": t0, "status": "invalid", "diagnostics": [d]})
            continue
        t1 = t0 + window_s
        abuts = False
        if idx + 1 < len(injections):
            nxt = float(injections[idx + 1]["time_s"])
            if nxt < t1:
                t1 = nxt
                abuts = True  # 峰可重叠：窗口相接合法
        truncated_by = None
        for g in gaps:
            if t0 < g[0] < t1:
                t1 = float(g[0])
                truncated_by = g  # 积分窗不跨越数据缺口
        if truncated_by is not None:
            diags.append(Diagnostic("window_truncated_by_gap", loc,
                                    f"积分窗在数据缺口前截断为 [{t0},{t1}] s", "warning").to_dict())
            diagnostics.extend(diags)
        mask = (times >= t0) & (times <= t1)
        ts, ps = times[mask], power[mask]
        if len(ts) < 3:
            d = Diagnostic("window_too_short", loc, "有效采样点不足，无法积分").to_dict()
            diagnostics.append(d)
            results.append({"index": idx, "time_s": t0, "status": "invalid", "diagnostics": diags + [d]})
            continue
        base = np.empty_like(ts)
        depends: set = set()
        missing = False
        for k, tk in enumerate(ts):
            b, sid = baseline_at(segments, float(tk))
            if sid is None:
                missing = True
                break
            base[k] = b
            depends.add(sid)
        if missing:
            d = Diagnostic("baseline_missing", loc, f"窗口 [{t0},{t1}] s 内缺少基线段").to_dict()
            diagnostics.append(d)
            results.append({"index": idx, "time_s": t0, "status": "invalid", "diagnostics": diags + [d]})
            continue
        heat = float(_trapezoid(ps - base, ts))
        results.append({"index": idx, "time_s": t0, "window": [t0, t1],
                        "heat_uJ": heat, "n_points": int(len(ts)),
                        "abuts_next": abuts, "status": "ok",
                        "depends": sorted(depends), "diagnostics": diags})
    return {"param_version": PARAM_VERSION, "window_s": window_s,
            "injections": results, "diagnostics": diagnostics}
