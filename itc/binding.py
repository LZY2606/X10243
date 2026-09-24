"""绑定模型候选：一位点 / 多位点 / 稀释假设。

- 拟合用 NumPy 实现的 Levenberg-Marquardt（数值雅可比）。
- 资源预算耗尽、语义无解是不同结果状态。
- 多个等优模型（RSS 在容差内）全部保留为候选，不擅自一决定。
- 浓度修订只让旧拟合过期（stale），不改原积分。
"""

import numpy as np

from .outcomes import BUDGET_EXHAUSTED, SEMANTIC_UNSOLVABLE, diag, failure, ok

EQUIPOTENT_RTOL = 0.01  # RSS 相对容差内的模型互为等优


# ---------------- 模型 ----------------
def _complex_conc(K, M, X):
    """一位点复合物浓度。K, M, X 单位 M。"""
    K = np.maximum(K, 1e-12)
    b = 1.0 + K * (M + X)
    disc = np.maximum(b * b - 4.0 * K * K * M * X, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * K)


def _diluted_concs(M0, X0, V0, cumvol):
    """累积注射体积 cumvol (uL) 后的池内大分子浓度与配体总浓度。""。"""
    V = V0 + cumvol
    M = M0 * V0 / V
    X = X0 * cumvol / V
    return M, X


def model_heats(model, x, cond):
    """每次注射热量 q_i (uJ)。x 为缩放参数向量。"""
    V0 = cond["cell_volume_uL"]
    cumvol = np.cumsum(cond["volumes_uL"])
    M, X = _diluted_concs(cond["cell_conc_M"], cond["syringe_conc_M"], V0, cumvol)
    if model == "one_site":
        logK, dH, n = x[0], x[1] * 1e4, x[2]
        c = _complex_conc(10.0 ** logK, n * M, X)
        Q = V0 * dH * c
    elif model == "two_site":
        c1 = _complex_conc(10.0 ** x[0], x[2] * M, X)
        c2 = _complex_conc(10.0 ** x[3], x[5] * M, X)
        Q = V0 * (x[1] * 1e4 * c1 + x[4] * 1e4 * c2)
    elif model == "dilution":
        return x[0] * cond["volumes_uL"] + x[1]
    else:
        raise ValueError(model)
    q = np.diff(np.concatenate(([0.0], Q)))
    return q


MODEL_PARAMS = {
    "one_site": ["log10K", "dH_x1e4_J_mol", "n"],
    "two_site": ["log10K1", "dH1_x1e4_J_mol", "n1", "log10K2", "dH2_x1e4_J_mol", "n2"],
    "dilution": ["heat_per_uL_uJ", "offset_uJ"],
}


# ---------------- LM 拟合 ----------------
def _residuals(model, x, cond, observed):
    return model_heats(model, x, cond) - observed


def _jac(model, x, cond, observed):
    r0 = _residuals(model, x, cond, observed)
    J = np.empty((len(r0), len(x)))
    for j in range(len(x)):
        h = 1e-6 * (1.0 + abs(x[j]))
        xp = x.copy(); xp[j] += h
        J[:, j] = (_residuals(model, xp, cond, observed) - r0) / h
    return J, r0


def lm_fit(model, x0, cond, observed, max_iter):
    """返回 (x, rss, converged, iters)。预算耗尽时 converged=False。"""
    x = np.asarray(x0, dtype=float)
    lam = 1e-3
    r = _residuals(model, x, cond, observed)
    rss = float(r @ r)
    for it in range(max_iter):
        J, r = _jac(model, x, cond, observed)
        A = J.T @ J
        g = J.T @ r
        try:
            dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        except np.linalg.LinAlgError:
            return x, rss, False, it
        if np.linalg.norm(dx) < 1e-10 * (np.linalg.norm(x) + 1e-10):
            return x, float(r @ r), True, it
        x2 = x + dx
        r2 = _residuals(model, x2, cond, observed)
        rss2 = float(r2 @ r2)
        if rss2 < rss:
            x, rss = x2, rss2
            lam = max(lam / 10.0, 1e-12)
            if abs(rss2 - rss) < 1e-14 * (1.0 + rss):
                return x, rss, True, it
        else:
            lam = min(lam * 10.0, 1e12)
    return x, rss, False, max_iter


def _starts(model, observed):
    sign = -1.0 if np.mean(observed) < 0 else 1.0
    if model == "one_site":
        return [[lk, sign * 4.0, 1.0] for lk in (3.0, 5.0, 7.0)]
    if model == "two_site":
        return [[lk1, sign * 2.0, 0.5, lk2, sign * 2.0, 0.5]
                for lk1, lk2 in ((4.0, 6.0), (6.0, 4.0), (5.0, 5.0))]
    return [[float(np.mean(observed)) / 10.0, 0.0]]


def fit_models(integration, cond, budget=None):
    """对积分结果拟合全部模型候选。

    integration: run_integration 的 result。
    cond: {cell_conc_M, syringe_conc_M, cell_volume_uL}
    budget: {"max_iter": int}
    """
    budget = budget or {"max_iter": 200}
    usable = [i for i in integration["injections"]
              if i["status"] == "ok" and not i.get("discarded")]
    diagnostics = []
    for i in integration["injections"]:
        if i["status"] == "incomplete":
            diagnostics.append(diag(
                "excluded_from_fit", f"integration.injections[{i['index']}]",
                f"第 {i['index']} 针因空白对应缺失被排除出拟合输入",
                injection_index=i["index"]))
    if cond.get("cell_conc_M", 0) <= 0 or cond.get("syringe_conc_M", 0) <= 0:
        return failure(SEMANTIC_UNSOLVABLE, diagnostics + [diag(
            "concentration_invalid", "concentration",
            "池内或针筒浓度缺失/非正，绑定模型无语义解")])
    volumes = np.array([i["volume_uL"] for i in usable])
    observed = np.array([i["net_heat_uJ"] for i in usable])
    cond = dict(cond, volumes_uL=volumes)
    max_params = max(len(v) for v in MODEL_PARAMS.values())
    if len(usable) < 3:
        return failure(SEMANTIC_UNSOLVABLE, diagnostics + [diag(
            "insufficient_points", "integration",
            f"可用注射点 {len(usable)} 个，不足以拟合任何绑定模型")])

    candidates = []
    for model in ("one_site", "two_site", "dilution"):
        if len(usable) < len(MODEL_PARAMS[model]):
            diagnostics.append(diag(
                "model_underdetermined", f"fit.{model}",
                f"{model} 需要 {len(MODEL_PARAMS[model])} 参数，"
                f"可用点仅 {len(usable)} 个，跳过"))
            continue
        best_local = []
        for x0 in _starts(model, observed):
            x, rss, converged, iters = lm_fit(model, x0, cond, observed,
                                              budget["max_iter"])
            if not converged:
                return failure(BUDGET_EXHAUSTED, diagnostics + [diag(
                    "fit_budget_exhausted", f"fit.{model}",
                    f"{model} 在 {budget['max_iter']} 次迭代预算内未收敛",
                    model=model, start=list(x0))])
            best_local.append((rss, x))
        best_local.sort(key=lambda t: (round(t[0], 9), list(np.round(t[1], 9))))
        best_rss = best_local[0][0]
        # 同一模型的多个等价极小也保留。
        kept = []
        for rss, x in best_local:
            if rss <= best_rss * (1.0 + EQUIPOTENT_RTOL) + 1e-12:
                if all(np.linalg.norm(x - kx) > 1e-3 for _, kx in kept):
                    kept.append((rss, x))
        for rss, x in kept:
            params = {name: float(val) for name, val in zip(MODEL_PARAMS[model], x)}
            fitted = model_heats(model, x, cond)
            candidates.append({
                "model": model,
                "params": params,
                "rss": float(rss),
                "fitted_uJ": [float(v) for v in fitted],
                "residuals_uJ": [float(a - b) for a, b in zip(fitted, observed)],
                "injection_indices": [i["index"] for i in usable],
            })
    if not candidates:
        return failure(SEMANTIC_UNSOLVABLE, diagnostics + [diag(
            "no_fittable_model", "fit", "没有任何模型可拟合当前输入")])
    best = min(c["rss"] for c in candidates)
    for c in candidates:
        c["equipotent"] = c["rss"] <= best * (1.0 + EQUIPOTENT_RTOL) + 1e-12
    result = {
        "candidates": candidates,
        "observed_uJ": [float(v) for v in observed],
        "conditions": {k: v for k, v in cond.items() if k != "volumes_uL"},
        "units": {"heat": "uJ", "conc": "M", "volume": "uL"},
    }
    return ok(result=result, diagnostics=diagnostics)
