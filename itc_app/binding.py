"""绑定模型候选：一位点、两位点（多位点）与稀释假设。

仅依赖 NumPy。拟合：参数尺度归一 + Nelder-Mead 单纯形，带显式求值预算。
模型选择：AICc；等优容差 AIC_TIE_TOL 随规则版本化。
浓度修订只改变进入这里的浓度向量 -> 旧候选作废（在引擎层标记过期），
但原始积分永不被模型重算覆盖。
"""
from __future__ import annotations

import math

import numpy as np

from .errors import SemanticInvalid, BudgetExhausted

MODEL_RULE_VERSION = "binding-2026.1"
AIC_TIE_TOL = 4.0  # 等优 AIC 容差
DEFAULT_BUDGET = 4000

MODEL_PARAM_UNITS = {
    "one_site": {"K": "M^-1", "dH": "J/mol", "n": "stoichiometry"},
    "two_site": {"K1": "M^-1", "dH1": "J/mol", "n1": "stoichiometry",
                 "K2": "M^-1", "dH2": "J/mol", "n2": "stoichiometry"},
    "dilution": {"q_dil": "J"},
}


def concentrations(injections, cell_volume_l, ligand_conc_by_index):
    """返回每次注射后池内配体浓度 X(M) 与被滴定物浓度稀释因子。"""
    X, cum_v, v, xs = 0.0, 0.0, cell_volume_l, []
    for inj in injections:
        dv = inj["volume"]
        c = ligand_conc_by_index.get(inj["index"])
        if c is None:
            raise SemanticInvalid(
                f"注射 {inj['index']} 缺少配体浓度，禁止用默认浓度",
                diagnostics=[])
        # 注射后混合
        X = (X * v + c * dv) / (v + dv)
        v += dv
        cum_v += dv
        xs.append(X)
    return np.array(xs)


def analyte_concentration(injections, cell_volume_l, m0):
    v, out = cell_volume_l, []
    for inj in injections:
        m_now = m0 * cell_volume_l / v
        v += inj["volume"]
        out.append(m_now)
    return np.array(out)


def _total_heat_one(X, M, K, dH, V0):
    s = X + M + 1.0 / K
    bound = 0.5 * (s - np.sqrt(np.maximum(s * s - 4.0 * X * M, 0.0)))
    return V0 * dH * bound


def _total_heat_two(X, M, K1, dH1, n1, K2, dH2, n2, V0):
    b1 = n1 * M * K1 * X / (1.0 + K1 * X)
    b2 = n2 * M * K2 * X / (1.0 + K2 * X)
    return V0 * (dH1 * b1 + dH2 * b2)


def injection_heats(model, params, X, M, injections, V0):
    """由总热有限差分（含池体积稀释校正）得到每次注射热量 (J)。"""
    X0 = np.concatenate(([0.0], X[:-1]))
    V_before = np.concatenate(([V0], V0 + np.cumsum(
        [i["volume"] for i in injections])[:-1]))
    V_after = V0 + np.cumsum([i["volume"] for i in injections])
    if model == "one_site":
        Q = _total_heat_one(X, M, params["K"], params["dH"], V0)
        Qp = _total_heat_one(X0, np.concatenate(([M[0]], M[:-1])),
                             params["K"], params["dH"], V0)
    elif model == "two_site":
        kw = dict(K1=params["K1"], dH1=params["dH1"], n1=params["n1"],
                  K2=params["K2"], dH2=params["dH2"], n2=params["n2"],
                  V0=V0)
        Q = _total_heat_two(X, M, **kw)
        Mp = np.concatenate(([M[0]], M[:-1]))
        Qp = _total_heat_two(X0, Mp, **kw)
    elif model == "dilution":
        qd = params["q_dil"]
        return np.full(len(X), qd)
    else:
        raise SemanticInvalid(f"未知模型 {model}")
    dV = np.array([i["volume"] for i in injections])
    dQ = Q - Qp + (dV / V_after) * 0.5 * (Q + Qp)
    return dQ


# ---------------- 拟合（带预算的 Nelder-Mead，仅 NumPy） ----------------

def _nelder_mead(fun, x0, scales, budget, tol=1e-8):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    rng = np.random.RandomState(7)
    for j in range(n):
        x = np.array(x0, dtype=float)
        x[j] += scales[j] * (0.5 if x[j] != 0 else 1.0)
        simplex.append(x)
    used = 0

    def ev(x):
        nonlocal used
        used += 1
        if used > budget:
            raise BudgetExhausted(
                f"拟合求值预算耗尽（{budget} 次），结果不可信", budget, used)
        return fun(x)

    fx = [ev(x) for x in simplex]
    for _ in range(10000):
        order = np.argsort(fx)
        simplex = [simplex[i] for i in order]
        fx = [fx[i] for i in order]
        if np.max(np.abs(np.array(fx) - fx[0])) <= tol:
            return simplex[0], fx[0], used, True
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = ev(xr)
        if fx[0] <= fr < fx[-2]:
            simplex[-1], fx[-1] = xr, fr
            continue
        if fr < fx[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = ev(xe)
            if fe < fr:
                simplex[-1], fx[-1] = xe, fe
            else:
                simplex[-1], fx[-1] = xr, fr
            continue
        xc = centroid - 0.5 * (centroid - simplex[-1])
        fc = ev(xc)
        if fc < fx[-1]:
            simplex[-1], fx[-1] = xc, fc
        else:
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                fx[i] = ev(simplex[i])
    return simplex[0], fx[0], used, False


def _fit_model(model, x_k, heats, X, M, injections, V0, budget,
               force_exhaust=False):
    y = np.array([float(h) for h in heats], dtype=float)
    # 起始网格
    if model == "dilution":
        p0, scales, names = [np.mean(y)], [np.std(y) + 1e-6], ["q_dil"]
        minp = 1
    elif model == "one_site":
        names = ["K", "dH", "n"]
        minp = 3
        grid = []
        for lgK in (3.0, 5.0, 7.0, 9.0):
            for n in (0.5, 1.0, 2.0):
                grid.append([10 ** lgK, -1e4, n])
        scales = [1e7, 2e4, 1.5]
    elif model == "two_site":
        names = ["K1", "dH1", "n1", "K2", "dH2", "n2"]
        minp = 6
        grid = [[1e5, -1.5e4, 0.7, 1e3, -3e3, 0.3],
                [1e7, -8e3, 1.0, 1e4, -6e3, 0.5],
                [1e4, -2e4, 0.5, 1e6, -1e3, 0.5]]
        scales = [1e7, 2e4, 1.0, 1e7, 2e4, 1.0]
    else:
        raise SemanticInvalid(f"未知模型 {model}")

    if len(y) < minp:
        raise SemanticInvalid(
            f"{model} 至少需要 {minp} 个有效积分点，实际 {len(y)} 个；"
            "信息不足，禁止用默认参数伪装成功")

    def unpack(v):
        d = dict(zip(names, v))
        # 物理约束：K,n 为正
        for key in d:
            if key.startswith("K") or key.startswith("n"):
                d[key] = max(d[key], 1e-12)
        return d

    def sse(v):
        try:
            pred = injection_heats(model, unpack(v), X, M, injections, V0)
            r = pred - y
            return float(np.sum(r * r))
        except Exception:
            return 1e30

    exhausted = False
    best, best_sse, used = None, np.inf, 0
    try:
        if model == "dilution":
            xhat, f, used, conv = _nelder_mead(sse, p0, scales, budget)
            best, best_sse, converged = xhat, f, conv
        else:
            converged = True
            for g in grid:
                xhat, f, u, conv = _nelder_mead(sse, g, scales, budget - used)
                used += u
                if f < best_sse:
                    best_sse, best, converged = f, xhat, conv
    except BudgetExhausted:
        exhausted = True
        converged = False
        if best is None:
            raise
    pred = injection_heats(model, unpack(best), X, M, injections, V0)
    resid = (pred - y).tolist()
    params = unpack(best)
    return {
        "model": model, "params": params, "sse": float(best_sse),
        "residuals": resid, "predicted": pred.tolist(),
        "evaluations": used, "converged": bool(converged),
        "budget_exhausted": exhausted,
        "x_injection": list(x_k),
    }


def rank_candidates(fits, tie_tol=AIC_TIE_TOL):
    rows = []
    for f in fits:
        n = f["n_points"]
        p = len(f["params"])
        sse = max(f["sse"], 1e-30)
        aic = n * math.log(sse / n) + 2 * p
        if n - p - 1 > 0:
            aic += 2 * p * (p + 1) / (n - p - 1)
        rows.append({**f, "aic": aic, "n_params": p})
    valid = [r for r in rows if r["converged"] and not r["budget_exhausted"]]
    if not valid:
        for r in rows:
            r["rank"] = 0
            r["equivalent"] = False
        return rows
    best_aic = min(r["aic"] for r in valid)
    # 稳定顺序：aic，然后模型名，然后参数四舍五入
    valid.sort(key=lambda r: (round(r["aic"], 9), r["model"],
                              tuple(sorted((k, round(v, 9))
                                           for k, v in r["params"].items()))))
    invalid = [r for r in rows if not (r["converged"]
                                       and not r["budget_exhausted"])]
    ordered = valid + sorted(invalid, key=lambda r: r["model"])
    for rank, r in enumerate(ordered, start=1):
        r["rank"] = rank
        r["equivalent"] = (r in valid) and (r["aic"] - best_aic <= tie_tol)
    return ordered


def fit_all(x_k, heats, X, M, injections, V0,
            budget=DEFAULT_BUDGET, models=("one_site", "two_site", "dilution"),
            tie_tol=AIC_TIE_TOL):
    fits = []
    for model in models:
        f = _fit_model(model, x_k, heats, X, M, injections, V0, budget)
        f["n_points"] = len(heats)
        fits.append(f)
    return rank_candidates(fits, tie_tol)
