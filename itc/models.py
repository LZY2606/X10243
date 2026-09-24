"""绑定模型候选：one_site / two_site / dilution_const / dilution_linear。

模型（ITC 标准 Wiseman 形式，带注射体积置换校正）：
    V_i = V0 + Σ_{k<=i} dv_k          池内当前体积
    M_i  = cell_conc                   池内大分子浓度（恒定，稀释已在置换项处理）
    X_i  = (Σ dv_k * syr_conc) / V_i   池内总配体浓度
    Q_i  = n * dH * V_i * M_i * f(X_i, M_i, Ka)   总结合热
    dq_i = Q_i - Q_{i-1} + (dv_i/V_i) * (Q_i+Q_{i-1})/2   逐注热（含置换项）

* one_site:           二次方程解 f（结合分数）
* two_site:           两组独立位点，二分法解自由配体 L 的质量守恒
* dilution_const:     dq_i = q_dil                       （1 参数）
* dilution_linear:    dq_i = a + b*X_i                  （2 参数）
"""
from __future__ import annotations

import math

import numpy as np

from .errors import BudgetExhausted, Diagnostic, SemanticUnresolved
from .identity import stable_id

MODEL_FAMILIES = ("one_site", "two_site", "dilution_const", "dilution_linear")
FAMILY_RANK = {"one_site": 0, "two_site": 1, "dilution_const": 2, "dilution_linear": 3}
AIC_TIE = 2.0
CAL_TO_J = 4.184
R_GAS = 1.987204258e-3  # kcal/(mol*K)

DEFAULT_BUDGET = {"max_evals": 8000, "tie_aic": AIC_TIE}


def _solve_binding_fraction(X: np.ndarray, M: np.ndarray, Ka: float, n: float) -> np.ndarray:
    """一位点：结合位点占据分数 f，解 nM f = X * Ka(1-f)/(1+Ka(1-f)...) 的标准二次式。"""
    K, nm = Ka, n * M
    a = K
    b = 1.0 + K * (nm + X)
    c = K * nm * X
    disc = b * b - 4.0 * a * c
    disc = np.maximum(disc, 0.0)
    root = (b - np.sqrt(disc)) / (2.0 * a)
    f = (X - root) / np.where(nm > 0, nm, np.nan)
    return np.clip(np.nan_to_num(f, nan=0.0), 0.0, 1.0)


def _free_ligand(X: float, M: float, sites: list[tuple[float, float]]) -> float:
    """多位点：X = L + M Σ n_j K_j L/(1+K_j L)，二分法求 L。"""
    def mass(L: float) -> float:
        bound = sum(nj * Kj * L / (1.0 + Kj * L) for nj, Kj in sites)
        return L + M * bound - X

    lo, hi = 0.0, max(X, 1e-3)
    while mass(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mass(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def predict(family: str, params: dict, X: np.ndarray, M: float,
            V: np.ndarray, dv: np.ndarray) -> np.ndarray:
    if family == "one_site":
        f = _solve_binding_fraction(X, np.full_like(X, M), params["Ka"], params["n"])
        Q = 1000.0 * params["n"] * params["dH"] * V * M * f
    elif family == "two_site":
        sites = [(params["n1"], params["Ka1"]), (params["n2"], params["Ka2"])]
        Ls = np.array([_free_ligand(float(x), M, sites) for x in X])
        bound = (params["n1"] * params["Ka1"] * Ls / (1.0 + params["Ka1"] * Ls)
                 + params["n2"] * params["Ka2"] * Ls / (1.0 + params["Ka2"] * Ls))
        Q = 1000.0 * V * M * (params["dH1"] * params["n1"] * params["Ka1"] * Ls / (1.0 + params["Ka1"] * Ls)
                     + params["dH2"] * params["n2"] * params["Ka2"] * Ls / (1.0 + params["Ka2"] * Ls))
    elif family == "dilution_const":
        Q = np.cumsum(np.full_like(X, params["q_dil"]))
    elif family == "dilution_linear":
        step = np.empty_like(X)
        step[0] = params["a"] + params["b"] * X[0]
        step[1:] = params["b"] * (X[1:] - X[:-1])
        step[0] = params["a"] + params["b"] * X[0]
        Q = np.cumsum(step)
    else:
        raise SemanticUnresolved(f"未知模型族 {family!r}",
                                 [Diagnostic("unknown_family", "未知模型族", "/models", family)])
    Qprev = np.concatenate(([0.0], Q[:-1]))
    dq = Q - Qprev + (dv / V) * 0.5 * (Q + Qprev)
    return dq


def _displacement_operator(V: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """逐注 dq 与累计 Q 之间的线性算子（含体积置换项）。"""
    n = len(V)
    D = np.eye(n)
    for i in range(n):
        if i > 0:
            D[i, i - 1] = -1.0 - 0.5 * dv[i] / V[i]
        D[i, i] += 0.5 * dv[i] / V[i]
    return D


def build_dataset(dilution_run: dict, integration_run: dict,
                  concentration: dict, cell_volume: dict) -> dict:
    heats = {r["index"]: r["net_heat_cal"] for r in dilution_run["rows"]
             if r["subtracted"] and r["net_heat_cal"] is not None}
    by_index = {e["index"]: e for e in integration_run["injections"]}
    used = sorted(heats)
    if len(used) < 3:
        raise SemanticUnresolved(
            "扣除空白后的有效注射少于 3 个，无法拟合绑定模型",
            [Diagnostic("insufficient_points",
                        f"有效点 {len(used)} 个（至少 3 个）", "/dilution", len(used))],
        )
    M = concentration["cell"]["canonical_value"]
    syr = concentration["syringe"]["canonical_value"]
    V0 = cell_volume["cell_volume"]["canonical_value"]
    if not (M and syr and V0):
        raise SemanticUnresolved(
            "浓度或池体积缺少规范化值，模型输入不完整",
            [Diagnostic("missing_canonical", "需要 cell/syringe 浓度与池体积",
                        "/materials", None)],
        )
    all_idx = sorted(by_index)
    dv_all = np.array([by_index[i]["volume_l"] for i in all_idx], dtype=float)
    V_all = V0 + np.cumsum(dv_all)
    X_all = np.cumsum(dv_all * syr) / V_all
    keep = [all_idx.index(i) for i in used]
    dv = dv_all[keep]
    V = V_all[keep]
    X = X_all[keep]
    q = np.array([heats[i] for i in used], dtype=float)
    if np.any(dv <= 0) or np.any(np.diff(V) <= 0) == False and np.any(V <= 0):
        raise SemanticUnresolved("体积序列非正",
                                 [Diagnostic("bad_volume", "池体积或注射体积必须为正", "/injections", None)])
    return {"injection_ids": [by_index[i]["injection_id"] for i in used],
            "indices": used, "q": q, "M": M, "syr": syr, "V0": V0,
            "V": V, "dv": dv, "X": X}


def _nelder_mead(fn, x0: np.ndarray, step: np.ndarray, max_evals: int):
    """返回 x, f(x), 评估次数, 是否正常收敛。"""
    n = len(x0)
    simplex = [np.asarray(x0, dtype=float)]
    for j in range(n):
        v = np.asarray(x0, dtype=float).copy()
        v[j] += step[j]
        simplex.append(v)
    vals = []
    evals = 0

    def f(v):
        nonlocal evals
        evals += 1
        val = fn(v)
        if not np.isfinite(val):
            val = 1e300
        return val

    for v in simplex:
        vals.append(f(v))
    converged = False
    while evals < max_evals:
        order = np.argsort(vals, kind="stable")
        simplex = [simplex[k] for k in order]
        vals = [vals[k] for k in order]
        if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-7 and abs(vals[-1] - vals[0]) < 1e-10:
            converged = True
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = f(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
            continue
        xc = centroid - 0.5 * (centroid - simplex[-1])
        fc = f(xc)
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue
        best = simplex[0]
        simplex = [best] + [best + 0.5 * (v - best) for v in simplex[1:]]
        vals = [f(best)] + [f(v) for v in simplex[1:]]
    order = np.argsort(vals, kind="stable")
    return simplex[order[0]], vals[order[0]], evals, converged


def _rss_shape(data: dict, shapes: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """给定形状列，线性最小二乘求焓并返回 (系数, rss, 预测)。"""
    q = data["q"]
    coeffs, *_ = np.linalg.lstsq(shapes, q, rcond=None)
    pred = shapes @ coeffs
    return coeffs, float(np.sum((q - pred) ** 2)), pred


def _fit_one_site(data: dict, budget: dict) -> dict:
    V, dv, X, M = data["V"], data["dv"], data["X"], data["M"]
    D = _displacement_operator(V, dv)
    grid_logK = np.linspace(2.5, 9.0, 14)
    grid_logn = np.linspace(-1.6, 0.7, 10)
    best = None
    evals = 0

    def objective(z):
        logK, logn = z
        Ka, n = 10.0 ** logK, 10.0 ** logn
        f = _solve_binding_fraction(X, np.full_like(X, M), Ka, n)
        Q0 = 1000.0 * n * V * M * f
        shape = D @ Q0
        denom = float(shape @ shape)
        if denom <= 0:
            return 1e300
        dH = float(data["q"] @ shape) / denom
        rss = float(np.sum((data["q"] - dH * shape) ** 2))
        return math.log(max(rss, 1e-30))

    starts = [(a, b) for a in grid_logK for b in grid_logn]
    per_start = max(400, budget["max_evals"] // len(starts))
    for st in starts:
        x, val, used, conv = _nelder_mead(objective, np.array(st), np.array([0.5, 0.3]), per_start)
        evals += used
        if best is None or val < best[0]:
            best = (val, x)
        if evals >= budget["max_evals"]:
            break
    if best is None or evals >= budget["max_evals"]:
        raise BudgetExhausted(
            f"一位点拟合在 {budget['max_evals']} 次评估内未收敛",
            [Diagnostic("fit_budget", "一位点评估预算耗尽", "/models/one_site", evals)])
    logK, logn = best[1]
    Ka, n = 10.0 ** logK, 10.0 ** logn
    f = _solve_binding_fraction(X, np.full_like(X, M), Ka, n)
    shape = D @ (n * V * M * f)
    dH = float(data["q"] @ shape) / float(shape @ shape)
    pred = dH * shape
    rss = float(np.sum((data["q"] - pred) ** 2))
    return {"params": {"Ka": Ka, "n": n, "dH": dH}, "rss": rss, "pred": pred, "evals": evals}


def _fit_two_site(data: dict, budget: dict) -> dict:
    V, dv, X, M = data["V"], data["dv"], data["X"], data["M"]
    D = _displacement_operator(V, dv)

    def shapes_from(z):
        logK1, logn1, logK2, logn2 = z
        K1, n1, K2, n2 = 10.0 ** logK1, 10.0 ** logn1, 10.0 ** logK2, 10.0 ** logn2
        Ls = np.array([_free_ligand(float(x), M, [(n1, K1), (n2, K2)]) for x in X])
        b1 = n1 * K1 * Ls / (1.0 + K1 * Ls)
        b2 = n2 * K2 * Ls / (1.0 + K2 * Ls)
        return D @ (1000.0 * V * M * b1), D @ (1000.0 * V * M * b2)

    def objective(z):
        s1, s2 = shapes_from(z)
        A = np.column_stack([s1, s2])
        try:
            c, *_ = np.linalg.lstsq(A, data["q"], rcond=None)
        except np.linalg.LinAlgError:
            return 1e300
        rss = float(np.sum((data["q"] - A @ c) ** 2))
        return math.log(max(rss, 1e-30))

    starts = [
        np.array([3.0, -1.0, 5.5, -0.5]),
        np.array([4.5, -0.8, 7.0, -0.9]),
        np.array([6.0, -0.4, 3.5, -1.2]),
        np.array([7.5, -1.0, 4.0, -0.3]),
        np.array([5.0, -1.3, 6.5, -0.7]),
        np.array([3.5, -0.4, 8.0, -1.1]),
    ]
    best, evals = None, 0
    for s in starts:
        x, val, used, conv = _nelder_mead(objective, s, np.array([0.4] * 4),
                                          max(300, budget["max_evals"] // len(starts)))
        evals += used
        if best is None or val < best[0]:
            best = (val, x)
        if evals >= budget["max_evals"]:
            break
    if best is None or evals >= budget["max_evals"]:
        limit = budget["max_evals"]
        raise BudgetExhausted(
            f"两位点拟合在 {limit} 次评估内未收敛",
            [Diagnostic("fit_budget", "两位点评估预算耗尽", "/models/two_site", evals)])
    s1, s2 = shapes_from(best[1])
    A = np.column_stack([s1, s2])
    (dH1, dH2), rss, pred = _rss_shape(data, A)
    logK1, logn1, logK2, logn2 = best[1]
    return {"params": {"Ka1": 10.0 ** logK1, "n1": 10.0 ** logn1, "dH1": float(dH1),
                       "Ka2": 10.0 ** logK2, "n2": 10.0 ** logn2, "dH2": float(dH2)},
            "rss": rss, "pred": pred, "evals": evals}


def _fit_dilution(data: dict, linear: bool) -> dict:
    V, dv, X = data["V"], data["dv"], data["X"]
    D = _displacement_operator(V, dv)
    if not linear:
        shape = D @ V
        qd = float(data["q"] @ shape) / float(shape @ shape)
        pred = qd * shape
        return {"params": {"q_dil": qd}, "rss": float(np.sum((data["q"] - pred) ** 2)),
                "pred": pred, "evals": 1}
    col_const = D @ V
    col_X = D @ np.cumsum(X * dv)
    A = np.column_stack([col_const, col_X])
    (a, b), rss, pred = _rss_shape(data, A)
    return {"params": {"a": float(a), "b": float(b)}, "rss": rss, "pred": pred, "evals": 1}


def _model_row(family: str, fit: dict, data: dict, n_points: int, temperature_k: float | None) -> dict:
    k = len(fit["params"])
    rss = max(fit["rss"], 1e-30)
    aic = n_points * math.log(rss / n_points) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(n_points - k - 1, 1)
    residuals = (data["q"] - fit["pred"]).tolist()
    params_out = dict(fit["params"])
    thermo: dict = {}
    if family == "one_site":
        Ka, dH = params_out["Ka"], params_out["dH"]
        thermo["Kd_M"] = 1.0 / Ka
        thermo["dG_kcal_per_mol"] = -R_GAS * (temperature_k or 298.15) * math.log(Ka)
        thermo["dH_kcal_per_mol"] = dH
        thermo["dS_cal_per_mol_K"] = (dH - thermo["dG_kcal_per_mol"]) * 1000.0 / (temperature_k or 298.15)
    if family == "two_site":
        for j, (Kk, Hk) in enumerate((("Ka1", "dH1"), ("Ka2", "dH2")), start=1):
            thermo[f"Kd{j}_M"] = 1.0 / params_out[Kk]
            thermo[f"dG{j}_kcal_per_mol"] = -R_GAS * (temperature_k or 298.15) * math.log(params_out[Kk])
            thermo[f"dH{j}_kcal_per_mol"] = params_out[Hk]
    return {
        "family": family,
        "params": params_out,
        "param_units": _param_units(family),
        "k": k,
        "rss": fit["rss"],
        "aic": aic,
        "aicc": aicc,
        "residuals_cal": residuals,
        "predicted_heat_cal": fit["pred"].tolist(),
        "thermo": thermo,
        "evals": fit.get("evals", 0),
        "equivalent": False,
        "candidate_id": None,
    }


def _param_units(family: str) -> dict:
    if family == "one_site":
        return {"Ka": "M^-1", "n": "sites", "dH": "kcal/mol"}
    if family == "two_site":
        return {"Ka1": "M^-1", "n1": "sites", "dH1": "kcal/mol",
                "Ka2": "M^-1", "n2": "sites", "dH2": "kcal/mol"}
    if family == "dilution_const":
        return {"q_dil": "cal"}
    return {"a": "cal", "b": "cal/mol"}


def fit_models(dilution_run: dict, integration_run: dict, concentration: dict,
               cell_volume: dict, temperature: dict | None = None,
               budget: dict | None = None) -> dict:
    bud = {**DEFAULT_BUDGET, **(budget or {})}
    data = build_dataset(dilution_run, integration_run, concentration, cell_volume)
    temp_k = temperature["temperature"]["canonical_value"] if temperature else None

    fitters = [
        ("one_site", lambda: _fit_one_site(data, bud)),
        ("two_site", lambda: _fit_two_site(data, bud)),
        ("dilution_const", lambda: _fit_dilution(data, False)),
        ("dilution_linear", lambda: _fit_dilution(data, True)),
    ]
    rows = []
    errors = []
    for family, fn in fitters:
        try:
            rows.append(_model_row(family, fn(), data, len(data["q"]), temp_k))
        except BudgetExhausted as exc:
            errors.append({"family": family, **exc.to_dict()})
        except SemanticUnresolved as exc:
            errors.append({"family": family, **exc.to_dict()})

    if not rows:
        raise SemanticUnresolved(
            "所有模型族都无法给出合法候选",
            [Diagnostic("no_candidates", "无合法模型候选", "/models", errors)],
        )

    rows.sort(key=lambda r: (round(r["aic"], 9), FAMILY_RANK[r["family"]],
                             tuple(sorted((k, round(v, 9)) for k, v in r["params"].items()))))
    best_aic = rows[0]["aic"]
    for r in rows:
        r["equivalent"] = (r["aic"] - best_aic) <= bud["tie_aic"]
        r["candidate_id"] = stable_id("model", {
            "family": r["family"],
            "aic": round(r["aic"], 9),
            "params": {k: round(float(v), 9) for k, v in r["params"].items()},
        })

    chosen = rows[0]
    ambiguous = [r["family"] for r in rows if r["equivalent"]]
    findings = []
    if len(ambiguous) > 1:
        findings.append({
            "finding_id": "fit::multiple_equivalent_models",
            "severity": "warning",
            "code": "multiple_equivalent_models",
            "message": f"ΔAIC ≤ {bud['tie_aic']} 的等优模型: {', '.join(ambiguous)}；全部保留为合法候选",
            "loc": "/models", "families": ambiguous, "tie_aic": bud["tie_aic"],
        })

    return {
        "kind": "model_fit",
        "inputs": {
            "injection_ids": data["injection_ids"],
            "injection_indices": data["indices"],
            "ratio_X": data["X"].tolist(),
            "heat_cal": data["q"].tolist(),
            "cell_conc_M": data["M"],
            "syringe_conc_M": data["syr"],
            "V0_L": data["V"],
            "temperature_K": temp_k,
            "coordinate_basis": "t 单位 s；X 为池内总配体/大分子摩尔比",
        },
        "candidates": rows,
        "ranked_families": [r["family"] for r in rows],
        "preferred_family": chosen["family"],
        "ambiguous": len(ambiguous) > 1,
        "equivalent_families": ambiguous,
        "budget": {"max_evals": bud["max_evals"], "tie_aic": bud["tie_aic"]},
        "errors": errors,
        "findings": findings,
    }
