"""绑定模型拟合与等优候选（确定性、固定求值预算、无随机性）。

- 一位点 / 多位点（两位点）/ 稀释假设同时比较
- 无法唯一决定时保留全部合法候选：|ΔAICc| <= tie_epsilon 视为等优
- 拟合在固定求值预算内进行；预算不足以完成网格+下山 => BUDGET_EXHAUSTED
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import BudgetExhausted, Diagnostic
from .ids import digest, stable_deps
from .models import dilution_only, m_titles, one_site, two_site

MIN_EVAL_BUDGET = 300


@dataclass
class FitPoint:
    index: int
    volume_ul: float
    net_ucal: float


def _predict(model: str, p: np.ndarray, mt: np.ndarray, cell_mm: float,
             v0: float, vols: np.ndarray) -> np.ndarray:
    if model == "one_site":
        n, lk, dh = p
        return one_site(mt, cell_mm, v0, n, 10 ** lk, dh, vols)
    if model == "two_site":
        n1, lk1, dh1, n2, lk2, dh2 = p
        return two_site(mt, cell_mm, v0, n1, 10 ** lk1, dh1, n2, 10 ** lk2, dh2, vols)
    return dilution_only(mt, p[0], v0, vols)


def _objective(model: str, p: np.ndarray, mt, cell_mm, v0, vols, y) -> float:
    try:
        yhat = _predict(model, p, mt, cell_mm, v0, vols)
    except (ValueError, FloatingPointError):
        return np.inf
    if not np.all(np.isfinite(yhat)):
        return np.inf
    return float(np.sum((yhat - y) ** 2))


def _minimize(model, p0, mt, cell_mm, v0, vols, y, budget) -> tuple[np.ndarray, float, bool, int]:
    """Nelder-Mead（无 scipy）。返回 (参数, SSE, converged, 已用求值次数)。"""
    n = len(p0)
    simplex = [np.array(p0, dtype=float)]
    step = np.array([0.25] * n)
    for i in range(n):
        v = np.array(p0, dtype=float)
        v[i] += step[i]
        simplex.append(v)
    vals = [_objective(model, v, mt, cell_mm, v0, vols, y) for v in simplex]
    used = len(simplex)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    converged = False
    max_iters = max(1, (budget - used) // (2 * n + 2))
    for _ in range(max_iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        spread = max(vals) - min(vals)
        if spread < 1e-10 * (1.0 + abs(vals[0])):
            converged = True
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = _objective(model, xr, mt, cell_mm, v0, vols, y)
        used += 1
        if used > budget:
            break
        if vals[0] <= fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
            continue
        if fr < vals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = _objective(model, xe, mt, cell_mm, v0, vols, y)
            used += 1
            if used > budget:
                simplex[-1] = xr
                vals[-1] = fr
                break
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = _objective(model, xc, mt, cell_mm, v0, vols, y)
            used += 1
            if used > budget:
                break
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = _objective(model, simplex[i], mt, cell_mm, v0, vols, y)
                    used += 1
                if used > budget:
                    break
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]], converged, used


BOUNDS = {
    "one_site": {"n": (0.05, 10.0), "log10K": (2.0, 9.0), "dH": (-50.0, 50.0)},
    "two_site": {"n1": (0.05, 10.0), "log10K1": (2.0, 9.0), "dH1": (-50.0, 50.0),
                 "n2": (0.05, 10.0), "log10K2": (2.0, 9.0), "dH2": (-50.0, 50.0)},
    "dilution": {"q_dil_per_uL": (-50.0, 50.0)},
}

GRID = {
    "one_site": [
        [0.5, 1.0, 2.0],      # n
        [3.0, 5.0, 7.0],      # log10K
        [-10.0, 0.0],         # dH
    ],
    "two_site": [
        [0.5, 1.5], [3.0, 6.0], [-8.0, 0.0],
        [0.5, 1.5], [5.0, 8.0], [-8.0, 0.0],
    ],
    "dilution": [[-1.0, 0.0, 1.0]],
}

K_COUNTS = {"one_site": 3, "two_site": 6, "dilution": 1}
DISPLAY = ("one_site", "two_site", "dilution")


def _grid_starts(model: str) -> list[np.ndarray]:
    axes = GRID[model]
    starts = [[]]
    for ax in axes:
        starts = [s + [v] for s in starts for v in ax]
    return [np.array(s, dtype=float) for s in starts]


def _clip(model: str, p: np.ndarray) -> np.ndarray:
    names = list(BOUNDS[model].keys())
    out = np.array(p, dtype=float)
    for i, name in enumerate(names):
        lo, hi = BOUNDS[model][name]
        if out[i] < lo:
            out[i] = lo
        elif out[i] > hi:
            out[i] = hi
    return out


def _fit_one_model(model, mt, cell_mm, v0, vols, y, total_budget) -> dict:
    k = K_COUNTS[model]
    starts = _grid_starts(model)
    per = max(8, total_budget // (len(starts) * 3))  # 三个模型分总预算

    def obj(p, *_a):
        return _objective(model, _clip(model, p), mt, cell_mm, v0, vols, y)

    best_p, best_sse, used_total, converged = None, np.inf, 0, False
    for p0 in starts:
        p, sse, conv, used = _minimize(model, p0, mt, cell_mm, v0, vols, y, per)
        p = _clip(model, p)
        sse = obj(p)
        used_total += used + 1
        if sse < best_sse:
            best_sse, best_p, converged = sse, p, conv
        if used_total >= total_budget and best_sse == np.inf:
            break
    budget_exhausted = used_total >= total_budget and not converged
    names = list(BOUNDS[model].keys())
    params = {name: float(v) for name, v in zip(names, best_p)} if best_p is not None else {}
    return {"model": model, "params": params, "sse": float(best_sse), "k": k,
            "converged": bool(converged), "evaluations": used_total,
            "budget_exhausted": bool(budget_exhausted)}


def aicc(sse: float, n_pts: int, k: int) -> float:
    if sse <= 0 or not math.isfinite(sse):
        sse = 1e-12
    base = n_pts * math.log(sse / n_pts) + 2 * k
    denom = n_pts - k - 1
    if denom > 0:
        base += 2 * k * (k + 1) / denom
    return base


def fit_binding(net_set: dict, schedule: dict, concentration: dict, conditions: dict,
                rules: dict, integral_flags: dict[int, list[str]] | None = None,
                deps_extra: dict | None = None) -> dict:
    """组装拟合点：积分有效、首注按规则舍弃、空白对应成功、非重叠污染。

    integral_flags: injection_index -> flags（来自积分层），用于首注/有效性排除并说明。
    """
    rv = rules["values"]
    budget = int(rv["fit_evaluations"])
    if budget < MIN_EVAL_BUDGET:
        raise BudgetExhausted(
            f"拟合求值预算 {budget} < 最小预算 {MIN_EVAL_BUDGET}",
            [Diagnostic("FIT_BUDGET", "rules.fit_evaluations",
                        f"need>={MIN_EVAL_BUDGET}", budget)],
        )

    vols_by_idx = {
        inj["index"]: inj["volume"]["canonical"]
        for inj in schedule["injections"] if inj["volume"]["valid"]
    }
    points: list[FitPoint] = []
    excluded: list[dict] = []
    integral_flags = integral_flags or {}
    for r in net_set["results"]:
        idx = r["injection_index"]
        flags = integral_flags.get(idx, [])
        reason = None
        if not r["valid"]:
            reason = "net_invalid"
        elif rv["discard_first"] and idx == 1:
            reason = "discarded_first"
        elif "NO_SEGMENT" in flags or "INVALID_INJECTION" in flags:
            reason = "integral_invalid"
        elif idx not in vols_by_idx:
            reason = "missing_volume"
        if reason:
            excluded.append({"injection_index": idx, "reason": reason, "flags": flags})
            continue
        points.append(FitPoint(idx, vols_by_idx[idx], r["net_heat_ucal"]))

    from .errors import SemanticUnresolvable
    if len(points) < 4:
        raise SemanticUnresolvable(
            f"可用于拟合的注射点仅 {len(points)} 个（至少需要 4 个）",
            [Diagnostic("INSUFFICIENT_FIT_POINTS", "binding",
                        f"excluded={excluded}", len(points))],
        )

    syringe_mm = concentration["syringe"]["canonical"]
    cell_mm = concentration["cell"]["canonical"]
    v0 = conditions["cell_volume"]["canonical"]
    if syringe_mm is None or cell_mm is None or v0 is None:
        raise SemanticUnresolvable(
            "浓度或池体积缺少有效规范值，无法绑定建模",
            [Diagnostic("MISSING_CANONICAL", "binding", "syringe/cell/v0")],
        )

    idx = np.array([p.index for p in points])
    vols = np.array([p.volume_ul for p in points])
    y = np.array([p.net_ucal for p in points])
    # m_titles 需要按完整注射顺序累计；用序号映射回累计值
    all_vols = np.array([inj["volume"]["canonical"]
                         for inj in sorted(schedule["injections"], key=lambda z: z["index"])
                         if inj["volume"]["valid"]])
    all_mt = m_titles(all_vols, syringe_mm, v0)
    mt = np.array([all_mt[i - 1] for i in idx])

    per_model_budget = budget // 3
    candidates = []
    for model in DISPLAY:
        fit = _fit_one_model(model, mt, cell_mm, v0, vols, y, per_model_budget)
        n_pts = len(points)
        fit["aicc"] = aicc(fit["sse"], n_pts, fit["k"])
        yhat = _predict(model, _clip(model, np.array([fit["params"][nm]
                         for nm in BOUNDS[model]])), mt, cell_mm, v0, vols)
        fit["residuals_ucal"] = [float(z) for z in (y - yhat)]
        fit["predicted_ucal"] = [float(z) for z in yhat]
        fit["observed_ucal"] = [float(z) for z in y]
        fit["per_mole_kcal"] = _per_mole_params(model, fit["params"])
        candidates.append(fit)

    best_aicc = min(c["aicc"] for c in candidates)
    eps = float(rv["tie_epsilon_aicc"])
    for c in candidates:
        c["delta_aicc"] = c["aicc"] - best_aicc
        c["tied_best"] = abs(c["delta_aicc"]) <= eps
    # 稳定顺序：aicc 升序，平手按固定模型名
    candidates.sort(key=lambda c: (round(c["aicc"], 10), DISPLAY.index(c["model"])))
    winners = [c["model"] for c in candidates if c["tied_best"]]

    deps = {
        "net_set": net_set.get("id") or _net_id(net_set),
        "schedule": schedule["id"],
        "concentration": concentration["id"],
        "conditions": conditions["id"],
        "rules": rules["id"],
        "points": [int(i) for i in idx],
    }
    if deps_extra:
        deps.update(deps_extra)
    cid = digest("fit", stable_deps(deps))
    return {
        "id": cid,
        "kind": "fit_report",
        "points": [{"injection_index": int(i), "volume_ul": float(v), "net_heat_ucal": float(q)}
                   for i, v, q in zip(idx, vols, y)],
        "m_titles_mM": [float(z) for z in mt],
        "excluded": excluded,
        "candidates": candidates,
        "winners": winners,
        "unique_winner": winners[0] if len(winners) == 1 else None,
        "tie": len(winners) > 1,
        "tie_epsilon_aicc": eps,
        "n_points": len(points),
        "budget_total": budget,
        "deps": deps,
        "events": [{
            "type": "MODEL_SELECTION",
            "rule": "MULTIPLE_EQUAL_BEST" if len(winners) > 1 else "UNIQUE_BEST",
            "winners": winners,
            "detail": {"delta_aicc": {c["model"]: round(c["delta_aicc"], 6) for c in candidates},
                       "epsilon": eps},
            "rule_version": rules["id"],
        }],
    }


def _net_id(net_set: dict) -> str:
    return digest("netset", [[r["id"]] for r in net_set["results"]])


def _per_mole_params(model: str, params: dict) -> dict:
    if model == "one_site":
        return {"n": params["n"], "K_M": 10 ** params["log10K"], "dH_kcal_per_mol": params["dH"]}
    if model == "two_site":
        return {"n1": params["n1"], "K1_M": 10 ** params["log10K1"], "dH1_kcal_per_mol": params["dH1"],
                "n2": params["n2"], "K2_M": 10 ** params["log10K2"], "dH2_kcal_per_mol": params["dH2"]}
    return {"q_dil_per_uL_ucal": params["q_dil_per_uL"]}
