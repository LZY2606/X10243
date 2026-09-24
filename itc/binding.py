"""绑定模型候选：一位点 / 多位点 / 稀释假设。

规则：
- 无法唯一决定时保留全部合法候选（等优容差内不裁剪）。
- 资源预算耗尽返回 budget_exhausted，与格式损坏、语义无解是三种不同结果。
- 被排除/未决的针列入 excluded，不静默丢弃。
"""
from __future__ import annotations

import numpy as np

MODEL_NAMES = ["one_site", "two_site", "dilution_only"]
EQUAL_TOL = 0.10  # SSE 相对容差：等优模型全部保留


class BudgetExhausted(Exception):
    pass


def _q_content(mt_uM, xt_uM, vol_L, n, ka, dh):
    """Wiseman 单组分热含量（µJ）。mt/xt 单位 µM=µmol/L，dh 单位 µJ/µmol。"""
    n = max(n, 1e-6)
    r = xt_uM / (n * mt_uM)
    c = 1.0 / (n * ka * mt_uM)
    disc = np.sqrt(np.maximum((1.0 + r + c) ** 2 - 4.0 * xt_uM / (n * mt_uM), 0.0))
    frac = 0.5 * (1.0 + r + c - disc)
    return frac * n * mt_uM * vol_L * dh


def _per_injection(q_cum, inj_vols_uL, v0_L):
    """累计热含量 → 每针热量，含排出体积校正。"""
    dq = np.diff(np.concatenate([[0.0], q_cum]))
    dv_L = inj_vols_uL * 1e-6
    prev = np.concatenate([[0.0], q_cum[:-1]])
    return dq + (dv_L / v0_L) * (q_cum + prev) / 2.0


def predict(model, params, ctx):
    """模型预测的每针热量（µJ）。ctx 携带浓度、池体积、注射体积。"""
    n_inj = ctx["n_inj"]
    v0_L = ctx["v0_L"]
    inj_vols = np.asarray(ctx["inj_volumes_uL"], dtype=float)
    cum_v_uL = np.cumsum(inj_vols)
    vol_L = v0_L + cum_v_uL * 1e-6 / 2.0
    mt = ctx["mt0_uM"] * v0_L / (v0_L + cum_v_uL * 1e-6)
    xt = ctx["xt_uM"] * (cum_v_uL - inj_vols / 2.0) * 1e-6 / vol_L
    if model == "dilution_only":
        return np.full(n_inj, params[0])
    if model == "one_site":
        n, log_ka, dh = params
        q = _q_content(mt, xt, vol_L, n, 10.0 ** log_ka, dh)
        return _per_injection(q, inj_vols, v0_L)
    if model == "two_site":
        n1, lk1, dh1, n2, lk2, dh2 = params
        q = (_q_content(mt, xt, vol_L, n1, 10.0 ** lk1, dh1)
             + _q_content(mt, xt, vol_L, n2, 10.0 ** lk2, dh2))
        return _per_injection(q, inj_vols, v0_L)
    raise ValueError(f"unknown model {model}")


_STARTS = {
    "one_site": (np.array([1.0, 0.3, -25.0]), np.array([0.4, 1.0, 8.0])),
    "two_site": (np.array([0.6, 0.8, -18.0, 0.6, -0.8, -12.0]),
                 np.array([0.3, 0.8, 6.0, 0.3, 0.8, 6.0])),
}


def _nelder_mead(fn, x0, steps, budget):
    """小型 Nelder-Mead；评估计数超预算抛 BudgetExhausted。"""
    n = len(x0)
    simplex = [x0.astype(float).copy()]
    for i in range(n):
        pt = x0.astype(float).copy()
        pt[i] += steps[i]
        simplex.append(pt)
    vals = []
    used = 0
    for pt in simplex:
        vals.append(fn(pt))
        used += 1
        if used >= budget:
            raise BudgetExhausted()
    for _ in range(400):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if np.max(np.abs(np.subtract(simplex[-1], simplex[0]))) < 1e-7:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = fn(xr)
        used += 1
        if used >= budget:
            raise BudgetExhausted()
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = fn(xe)
            used += 1
            if used >= budget:
                raise BudgetExhausted()
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        xc = centroid + 0.5 * (worst - centroid)
        fc = fn(xc)
        used += 1
        if used >= budget:
            raise BudgetExhausted()
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue
        best = simplex[0]
        simplex = [best] + [best + 0.5 * (p - best) for p in simplex[1:]]
        vals = [vals[0]]
        for pt in simplex[1:]:
            vals.append(fn(pt))
            used += 1
            if used >= budget:
                raise BudgetExhausted()
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]], used


def fit_models(ctx, heats, model_names=None, budget=20000, equal_tol=EQUAL_TOL):
    """拟合候选模型。heats 为 subtract_blank 的输出。"""
    basis = {"rule": "binding_model_candidates",
             "equal_tolerance": equal_tol,
             "meaning": "等优容差内的模型全部保留；浓度修订使本结果过期但不改原积分"}
    model_names = model_names or list(MODEL_NAMES)
    usable = [h for h in heats if h["status"] == "ok" and h["net_uJ"] is not None]
    excluded = [h["index"] for h in heats
                if not (h["status"] == "ok" and h["net_uJ"] is not None)]
    if len(usable) < 4:
        return {"status": "semantic_unresolved", "candidates": [],
                "excluded": excluded,
                "diagnostics": [{"path": "heats", "code": "too_few_points",
                                 "message": "可用热量点不足，无法拟合"}],
                "rule_basis": basis}
    ys = np.array([h["net_uJ"] for h in usable], dtype=float)
    idx = [h["index"] for h in usable]
    inj_vols = np.array([ctx["inj_volumes_uL"][i] for i in idx], dtype=float)
    sub_ctx = {"n_inj": len(idx), "v0_L": ctx["v0_L"],
               "mt0_uM": ctx["mt0_uM"], "xt_uM": ctx["xt_uM"],
               "inj_volumes_uL": inj_vols}
    candidates, total_used, exhausted = [], 0, False
    for name in model_names:
        if name == "dilution_only":
            q = float(np.mean(ys[-3:])) if len(ys) >= 3 else float(np.mean(ys))
            pred = predict(name, np.array([q]), sub_ctx)
            sse = float(np.sum((ys - pred) ** 2))
            candidates.append({"model": name, "params": {"q_const_uJ": q},
                               "sse": sse, "evaluations": 1, "partial": False})
            continue
        x0, steps = _STARTS[name]

        def sse_of(p, _name=name):
            pred = predict(_name, p, sub_ctx)
            if not np.all(np.isfinite(pred)):
                return 1e18
            return float(np.sum((ys - pred) ** 2))

        try:
            best, sse, used = _nelder_mead(sse_of, x0, steps, budget - total_used)
            total_used += used
            candidates.append({"model": name, "params": _params_dict(name, best),
                               "sse": sse, "evaluations": used, "partial": False})
        except BudgetExhausted:
            exhausted = True
            candidates.append({"model": name, "params": {}, "sse": None,
                               "evaluations": total_used, "partial": True})
            break
    done = [c for c in candidates if c["sse"] is not None]
    if done:
        best_sse = min(c["sse"] for c in done)
        for c in done:
            c["kept"] = bool(c["sse"] <= best_sse * (1.0 + equal_tol))
        done.sort(key=lambda c: (c["sse"], c["model"]))
        for rank, c in enumerate(done):
            c["rank"] = rank
    ordered = done + [c for c in candidates if c["sse"] is None]
    for c in done:
        p = _params_array(c["model"], c["params"])
        pred = predict(c["model"], p, sub_ctx)
        c["curve"] = [{"index": i, "pred_uJ": float(pv),
                       "obs_uJ": float(yv), "resid_uJ": float(yv - pv)}
                      for i, pv, yv in zip(idx, pred, ys)]
    return {"status": "budget_exhausted" if exhausted else "ok",
            "candidates": ordered, "excluded": excluded,
            "used_indices": idx, "evaluations": total_used,
            "rule_basis": basis}


def _params_dict(name, arr):
    if name == "one_site":
        return {"n": float(arr[0]), "log10_Ka": float(arr[1]),
                "dH_uJ_per_umol": float(arr[2])}
    if name == "two_site":
        return {"n1": float(arr[0]), "log10_Ka1": float(arr[1]),
                "dH1_uJ_per_umol": float(arr[2]),
                "n2": float(arr[3]), "log10_Ka2": float(arr[4]),
                "dH2_uJ_per_umol": float(arr[5])}
    return {"q_const_uJ": float(arr[0])}


def _params_array(name, d):
    if name == "one_site":
        return np.array([d["n"], d["log10_Ka"], d["dH_uJ_per_umol"]])
    if name == "two_site":
        return np.array([d["n1"], d["log10_Ka1"], d["dH1_uJ_per_umol"],
                         d["n2"], d["log10_Ka2"], d["dH2_uJ_per_umol"]])
    return np.array([d["q_const_uJ"]])
