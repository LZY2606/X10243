"""绑定模型候选：一位点 Wiseman、两位点独立位点、稀释假设（常热）。

三种互异结果：
- converged            拟合收敛，含全部候选与等优集合
- budget_exhausted     评估预算耗尽仍未收敛（资源耗尽 != 语义无解）
- semantic_unsolvable  输入语义上无法拟合（不足 2 点、缺浓度等）
等优（ΔAICc <= 2.0，或一位点退化为稀释的不可辨识情形 ΔAICc <= 10）
时保留全部合法候选，按内容 id 稳定排序，不替审阅者做唯一选择。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .canonicalid import cid, key_hash

TIE_DAICC = 2.0
NONIDENT_DAICC = 10.0
MAX_EVALS_PER_START = 350
MAX_STARTS_TOTAL = 12


class SemanticUnsolvable(Exception):
    def __init__(self, reason: str, location: str):
        super().__init__(f"{reason} @ {location}")
        self.reason = reason
        self.location = location


@dataclass
class FitBudget:
    max_evals: int = MAX_EVALS_PER_START * MAX_STARTS_TOTAL
    used: int = 0


def assemble_titration(integration_records: list[dict], concentration: dict,
                       conditions: dict) -> dict:
    cond = {c["key"]: c["quantity"] for c in conditions.get("conditions", [])}
    v0 = cond.get("cell_volume")
    if not v0 or v0.get("si_value") is None:
        raise SemanticUnsolvable("missing_cell_volume: 池体积缺失或不可转换",
                                 "conditions#cell_volume")
    cmap = {c["index"]: c for c in concentration.get("concentrations", [])
            if c.get("index") is not None}
    rows, missing = [], []
    for r in integration_records:
        if r["status"] == "discarded":
            continue
        if r.get("net_heat_uj") is None:
            continue
        c = cmap.get(r["index"])
        if not c or c["ligand"].get("si_value") is None \
                or c["macromolecule"].get("si_value") is None:
            missing.append(r["index"])
            continue
        rows.append((r["index"], r["net_heat_uj"], r["injection_volume_ul"],
                     c["ligand"]["si_value"], c["macromolecule"]["si_value"]))
    if missing:
        # 缺浓度不是格式错误：这些点不能进拟合，带位置返回
        pass
    if len(rows) < 2:
        raise SemanticUnsolvable(
            "insufficient_points: 至少需要 2 个既有净热又有浓度的注射",
            f"injections:{missing or 'all'}")
    rows.sort(key=lambda z: z[0])
    idx = np.array([z[0] for z in rows], dtype=float)
    q = np.array([z[1] for z in rows], dtype=float)
    dv = np.array([z[2] for z in rows], dtype=float) / v0["si_value"]
    lig0 = np.array([z[3] for z in rows], dtype=float)
    mt = np.array([z[4] for z in rows], dtype=float)
    ratios = np.divide(lig0, mt, out=np.full_like(lig0, np.nan), where=mt > 0)
    if np.isnan(ratios).any() or (ratios <= 0).any():
        raise SemanticUnsolvable("bad_concentration: 浓度比率非正或缺失",
                                 f"injections:{missing}")
    X = np.cumsum(dv)
    L = ratios * X / (1.0 + X)
    M = mt / (1.0 + X)
    return {"index": idx, "q_uJ": q, "dV": dv, "L_uM": L, "M_uM": M,
            "v0_uL": v0["si_value"], "lig0_uM": lig0, "mt_uM": mt}


def _bound_theta(L, K):
    return K * L / (1.0 + K * L)


def predict_one_site(p, data):
    n, logk, dh, qd = p
    K, V = 10.0 ** logk, data["v0_uL"]
    Mtot = V * data["M_uM"]
    Ltot = V * data["L_uM"]
    theta = np.empty(len(Ltot))
    for i, (mt, lt) in enumerate(zip(Mtot, Ltot)):
        a, b, c = n * mt, lt + n * mt + 1.0 / K, lt
        root = (b - math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
        theta[i] = root
    bind = n * Mtot * dh * np.diff(np.concatenate([[0.0], theta]))
    w = data["dV"] / np.median(data["dV"])
    return bind + qd * w


def predict_two_site(p, data):
    n1, logk1, dh1, n2, logk2, dh2, qd = p
    K1, K2, V = 10.0 ** logk1, 10.0 ** logk2, data["v0_uL"]
    Mtot = V * data["M_uM"]
    Ltot = V * data["L_uM"]
    th1 = _bound_theta(Ltot / V, K1)
    th2 = _bound_theta(Ltot / V, K2)
    h = n1 * Mtot * dh1 * np.diff(np.concatenate([[0.0], th1])) \
        + n2 * Mtot * dh2 * np.diff(np.concatenate([[0.0], th2]))
    w = data["dV"] / np.median(data["dV"])
    return h + qd * w


def predict_dilution(p, data):
    (qd,) = p
    w = data["dV"] / np.median(data["dV"])
    return np.full(len(data["q_uJ"]), qd) * w


def nelder_mead(fn, x0, steps, budget: FitBudget, max_iter=200):
    """返回 (x, f, converged, budget_exhausted)。"""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for j, st in enumerate(steps):
        v = np.array(x0, dtype=float)
        v[j] += st
        simplex.append(v)

    def cost(v):
        budget.used += 1
        return fn(v)

    vals = [cost(v) for v in simplex]
    for _ in range(max_iter):
        order = np.argsort(vals, kind="stable")
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        spread = float(np.max(np.abs(simplex[-1] - simplex[0])))
        if spread < 1e-7:
            return simplex[0], vals[0], True, False
        if budget.used >= budget.max_evals:
            return simplex[0], vals[0], False, True
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = cost(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        elif fr < vals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = cost(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid - 0.5 * (centroid - simplex[-1])
            fc = cost(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                best = simplex[0]
                simplex = [best] + [best + 0.5 * (v - best) for v in simplex[1:]]
                vals = [cost(v) for v in simplex]
    return simplex[0], vals[0], False, budget.used >= budget.max_evals


def _aicc(sse, n, k):
    if n - k - 1 <= 0:
        return float("inf")
    return n * math.log(max(sse, 1e-300) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _bounds_penalty(x, lo, hi):
    pen = 0.0
    for v, a, b in zip(x, lo, hi):
        if v < a:
            pen += (a - v) ** 2 * 1e12
        elif v > b:
            pen += (v - b) ** 2 * 1e12
    return pen


SPECS = {
    "one_site": {
        "predict": predict_one_site,
        "lo": np.array([1e-4, 2.0, -500.0, -200.0]),
        "hi": np.array([20.0, 10.0, 500.0, 200.0]),
        "steps": np.array([0.5, 0.5, 20.0, 10.0]),
        "starts": [
            [1.0, 4.0, -50.0, 0.0], [1.0, 6.0, -50.0, 0.0],
            [0.5, 5.0, -100.0, -5.0], [2.0, 7.0, -20.0, 5.0],
            [1.0, 3.0, 0.0, -20.0],
        ],
        "k": 4,
        "param_names": ["n", "log10_K", "deltaH_uJ_per_pmol", "q_dilute_uJ"],
    },
    "two_site": {
        "predict": predict_two_site,
        "lo": np.array([1e-4, 2.0, -500.0, 1e-4, 2.0, -500.0, -200.0]),
        "hi": np.array([20.0, 10.0, 500.0, 20.0, 10.0, 500.0, 200.0]),
        "steps": np.array([0.5, 0.5, 20.0, 0.5, 0.5, 20.0, 10.0]),
        "starts": [
            [0.5, 5.0, -30.0, 0.5, 7.0, -20.0, 0.0],
            [1.0, 4.0, -50.0, 1.0, 8.0, 10.0, -5.0],
        ],
        "k": 7,
        "param_names": ["n1", "log10_K1", "dH1_uJ_per_pmol",
                        "n2", "log10_K2", "dH2_uJ_per_pmol", "q_dilute_uJ"],
    },
    "dilution": {
        "predict": predict_dilution,
        "lo": np.array([-500.0]), "hi": np.array([500.0]),
        "steps": np.array([20.0]),
        "starts": [[0.0], [-20.0], [20.0]],
        "k": 1,
        "param_names": ["q_dilute_uJ"],
    },
}


def fit_candidate(name, data, budget: FitBudget) -> dict | None:
    spec = SPECS[name]
    lo, hi, steps = spec["lo"], spec["hi"], spec["steps"]
    q = data["q_uJ"]

    def fn(x):
        pen = _bounds_penalty(x, lo, hi)
        try:
            y = spec["predict"](np.array(x, dtype=float), data)
        except (ValueError, FloatingPointError, OverflowError):
            return 1e30 + pen
        if np.any(~np.isfinite(y)):
            return 1e30 + pen
        return float(np.sum((y - q) ** 2)) + pen

    best = None
    for x0 in spec["starts"]:
        x, f, converged, exhausted = nelder_mead(fn, x0, steps, budget)
        x = np.minimum(np.maximum(x, lo), hi)
        f_clean = fn(x) - _bounds_penalty(x, lo, hi)
        cand = {"x": x, "sse": float(f_clean), "converged": converged,
                "exhausted": exhausted}
        if best is None or cand["sse"] < best["sse"]:
            best = cand
        if exhausted:
            break
    if best is None:
        return None
    y = spec["predict"](best["x"], data)
    n = len(q)
    return {
        "model": name,
        "param_names": spec["param_names"],
        "params": [float(v) for v in best["x"]],
        "predicted_uJ": [float(v) for v in y],
        "residuals_uJ": [float(v) for v in (q - y)],
        "sse_uJ2": best["sse"],
        "aicc": _aicc(best["sse"], n, spec["k"]),
        "converged": bool(best["converged"]),
        "budget_exhausted": bool(best["exhausted"]),
        "n_points": n,
        "k_params": spec["k"],
    }


def _annotate_identifiability(cands: list[dict], data) -> None:
    by = {c["model"]: c for c in cands}
    one = by.get("one_site")
    dil = by.get("dilution")
    if not one:
        return
    logk = one["params"][1]
    weak_boundary = logk <= SPECS["one_site"]["lo"][1] + 0.35
    if weak_boundary:
        one["non_identifiable"] = True
        one["identifiability_note"] = (
            "log10_K 贴下界：一位点无法与纯稀释假设区分，"
            "保留稀释模型为合法候选")
    else:
        one["non_identifiable"] = False


def fit_all_models(data, max_evals: int | None = None) -> dict:
    budget = FitBudget(max_evals=max_evals or (MAX_EVALS_PER_START * MAX_STARTS_TOTAL))
    cands = []
    exhausted_any = False
    for name in ("one_site", "two_site", "dilution"):
        c = fit_candidate(name, data, budget)
        if c is None:
            continue
        exhausted_any = exhausted_any or c["budget_exhausted"]
        cands.append(c)
    if not cands:
        return {"outcome": "budget_exhausted", "candidates": [],
                "budget_used": budget.used, "reason": "预算内未得到任何候选"}
    if all(c["budget_exhausted"] and not c["converged"] for c in cands) and exhausted_any:
        return {"outcome": "budget_exhausted", "candidates": [],
                "budget_used": budget.used, "reason": "评估预算耗尽，无候选收敛"}
    _annotate_identifiability(cands, data)
    best_aicc = min(c["aicc"] for c in cands)
    for c in cands:
        c["delta_aicc"] = c["aicc"] - best_aicc
        c["candidate_id"] = cid("fit", {
            "model": c["model"], "params": [round(v, 9) for v in c["params"]],
            "sse_uJ2": round(c["sse_uJ2"], 12)})
    tied = [c for c in cands if c["delta_aicc"] <= TIE_DAICC]
    by = {c["model"]: c for c in cands}
    if by.get("one_site", {}).get("non_identifiable") and "dilution" in by:
        gap = abs(by["one_site"]["aicc"] - by["dilution"]["aicc"])
        if gap <= NONIDENT_DAICC and by["dilution"] not in tied:
            tied.append(by["dilution"])
    order = {"one_site": 0, "two_site": 1, "dilution": 2}
    tied.sort(key=lambda c: (round(c["delta_aicc"], 6), order.get(c["model"], 9),
                             key_hash(c["candidate_id"])))
    cands.sort(key=lambda c: (round(c["delta_aicc"], 6), order.get(c["model"], 9),
                              key_hash(c["candidate_id"])))
    for i, c in enumerate(tied):
        c["tied_rank"] = i
    return {"outcome": "converged", "candidates": cands,
            "tied_candidate_ids": [c["candidate_id"] for c in tied],
            "unique_winner": tied[0]["candidate_id"] if len(tied) == 1 else None,
            "budget_used": budget.used}
