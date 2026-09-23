"""注射积分引擎。

领域规则:
- 注射后峰可重叠: 相邻注射的峰允许重叠, 积分窗在下一针开始处结束
  (overlap_policy = window_ends_at_next_injection), 两针都正常积分。
- 积分窗不跨越数据缺口: 窗口在缺口起点截断并打标记; 注射落在缺口内则该针不可积,
  返回带位置的诊断而不是空值。
- 窗口超出数据末端: 截断到最后采样点并打标记。
- 资源预算: 采样数超过 max_samples 时整体返回 budget_exhausted,
  与 format_corrupt / semantically_unsolvable 是三种不同结果。
"""
from __future__ import annotations

from .units import normalize, diag

PARAM_VERSION = "integrator/1.0"
DEFAULT_PARAMS = {
    "max_window_s": 150.0,
    "overlap_policy": "window_ends_at_next_injection",
    "max_samples": 200000,
}


def normalize_power(payload):
    """原始功率曲线 -> [(t_s, p_uJ_s)], 诊断带样本位置。原始表示不变。"""
    samples, diags = [], []
    for i, s in enumerate(payload.get("samples", [])):
        t, d1 = normalize(s["t"], "time", position={"sample": i, "field": "t"})
        p, d2 = normalize(s["power"], "power", position={"sample": i, "field": "power"})
        if d1:
            diags.append(d1)
        if d2:
            diags.append(d2)
        if d1 or d2:
            continue  # 无法转换的值留在原材料里, 派生层跳过并记录
        samples.append((t, p))
    samples.sort(key=lambda sp: (sp[0], sp[1]))
    gaps, gap_diags = normalize_gaps(payload)
    return samples, gaps, diags + gap_diags


def normalize_gaps(payload):
    gaps, diags = [], []
    for i, g in enumerate(payload.get("gaps", [])):
        gs, d1 = normalize(g["start"], "time", position={"gap": i, "field": "start"})
        ge, d2 = normalize(g["end"], "time", position={"gap": i, "field": "end"})
        if d1 or d2:
            diags.extend(d for d in (d1, d2) if d)
            continue
        gaps.append((gs, ge, g.get("reason", "")))
    gaps.sort(key=lambda g: g[0])
    return gaps, diags


def normalize_schedule(payload):
    """注射时序 -> [{index,t_s,vol_uL,vol_ok,role}], 体积无法转换时保留诊断。"""
    injections, diags = [], []
    for i, inj in enumerate(payload.get("injections", [])):
        t, d1 = normalize(inj["t"], "time", position={"injection": i, "field": "t"})
        vol, d2 = normalize(inj["volume"], "volume", position={"injection": i, "field": "volume"})
        if d1:
            diags.append(d1)
            continue
        if d2:
            diags.append(d2)  # 体积不可转换: 积分仍可做(与体积无关), 摩尔热在派生层诊断
        injections.append({
            "index": i,
            "injection_id": inj.get("id", f"inj-{i:02d}"),
            "t_s": t,
            "vol_uL": vol,
            "vol_ok": d2 is None,
            "role": inj.get("role", "titration"),
        })
    injections.sort(key=lambda x: (x["t_s"], x["index"]))
    return injections, diags


def baseline_fn(payload):
    """基线材料 -> 函数 b(t_s) (µJ/s)。常量或分段线性曲线。"""
    if payload.get("kind") == "constant":
        val, d = normalize(payload["power"], "power")
        if d:
            raise ValueError(d["message"])
        return lambda t: val
    pts = []
    for i, p in enumerate(payload.get("points", [])):
        t, d1 = normalize(p["t"], "time", position={"baseline_point": i})
        v, d2 = normalize(p["power"], "power", position={"baseline_point": i})
        if d1 or d2:
            raise ValueError((d1 or d2)["message"])
        pts.append((t, v))
    pts.sort()
    ts = [p[0] for p in pts]
    vs = [p[1] for p in pts]

    def b(t):
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        for k in range(len(ts) - 1):
            if ts[k] <= t <= ts[k + 1]:
                w = (t - ts[k]) / (ts[k + 1] - ts[k])
                return vs[k] * (1 - w) + vs[k + 1] * w
        return vs[-1]
    return b


def integrate(power_payload, schedule_payload, baseline_payload, params=None):
    """对每次注射积分。返回结果 payload(不含存储身份)。"""
    params = {**DEFAULT_PARAMS, **(params or {})}
    samples, gaps, diags = normalize_power(power_payload)
    injections, sched_diags = normalize_schedule(schedule_payload)
    diags = diags + sched_diags

    base = {
        "kind": "injection_integrals",
        "param_version": PARAM_VERSION,
        "params": params,
        "units": {"heat": "µJ", "time": "s", "power": "µJ/s"},
        "time_basis": power_payload.get("basis", {}).get("time", "elapsed_s_from_run_start"),
        "diagnostics": diags,
    }

    if len(samples) > params["max_samples"]:
        return {**base, "status": "budget_exhausted",
                "diagnostics": diags + [diag(
                    "budget_exhausted",
                    f"采样数 {len(samples)} 超过预算 {params['max_samples']}")],
                "injections": []}
    if len(samples) < 2:
        return {**base, "status": "semantically_unsolvable",
                "diagnostics": diags + [diag("insufficient_samples", "功率采样不足, 无法积分")],
                "injections": []}
    if not injections:
        return {**base, "status": "semantically_unsolvable",
                "diagnostics": diags + [diag("no_injections", "注射时序为空")],
                "injections": []}

    b = baseline_fn(baseline_payload)
    last_t = samples[-1][0]
    out = []
    for k, inj in enumerate(injections):
        t0 = inj["t_s"]
        next_t = injections[k + 1]["t_s"] if k + 1 < len(injections) else None
        desired_end = t0 + params["max_window_s"]
        if next_t is not None:
            desired_end = min(desired_end, next_t)
        flags, inj_diags = [], []
        # 规则: 积分窗不跨越数据缺口
        gap_hit = next((g for g in gaps if g[0] < desired_end and g[1] > t0), None)
        if gap_hit and gap_hit[0] <= t0:
            out.append({
                "injection_id": inj["injection_id"], "index": inj["index"],
                "t_s": t0, "vol_uL": inj["vol_uL"], "vol_ok": inj["vol_ok"],
                "role": inj["role"], "status": "error",
                "flags": ["injection_in_gap"],
                "diagnostics": [diag(
                    "injection_in_gap",
                    f"注射 {inj['injection_id']} 落在数据缺口 "
                    f"[{gap_hit[0]}, {gap_hit[1]}] 内, 不可积分",
                    position={"injection": inj["index"]})],
                "window_s": None, "heat_uJ": None, "n_samples": 0,
            })
            continue
        window_end = desired_end
        if gap_hit:
            window_end = min(window_end, gap_hit[0])
            flags.append("truncated_by_gap")
        if window_end > last_t:
            window_end = last_t
            flags.append("truncated_at_end")
        # 窗口 [t0, window_end]: 当右端点等于下一针起始时刻时排除该样本
        # (该样本已含下一针峰上升沿, 属于下一针窗口)
        win = [(t, p) for (t, p) in samples
               if t0 <= t <= window_end
               and not (next_t is not None and window_end == next_t
                        and t == window_end)]
        if len(win) < 2:
            out.append({
                "injection_id": inj["injection_id"], "index": inj["index"],
                "t_s": t0, "vol_uL": inj["vol_uL"], "vol_ok": inj["vol_ok"],
                "role": inj["role"], "status": "error",
                "flags": flags + ["insufficient_samples"],
                "diagnostics": [diag(
                    "insufficient_samples",
                    f"注射 {inj['injection_id']} 窗口 [{t0}, {window_end}] 内采样不足",
                    position={"injection": inj["index"]})],
                "window_s": [t0, window_end], "heat_uJ": None, "n_samples": len(win),
            })
            continue
        heat = 0.0
        for (ta, pa), (tb, pb) in zip(win, win[1:]):
            heat += 0.5 * ((pa - b(ta)) + (pb - b(tb))) * (tb - ta)
        out.append({
            "injection_id": inj["injection_id"], "index": inj["index"],
            "t_s": t0, "vol_uL": inj["vol_uL"], "vol_ok": inj["vol_ok"],
            "role": inj["role"], "status": "ok",
            "flags": flags, "diagnostics": [],
            "window_s": [t0, window_end], "heat_uJ": heat, "n_samples": len(win),
        })

    n_err = sum(1 for o in out if o["status"] == "error")
    status = "ok" if n_err == 0 else "partial"
    return {**base, "status": status, "injections": out}
