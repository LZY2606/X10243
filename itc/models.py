"""结合模型候选与确定性拟合。

- one_site:   Wiseman 1:1 模型（K, ΔH, n）
- two_site:   两个独立位点（K1, ΔH1, K2, ΔH2，各 n=1）
- dilution:   仅稀释热（每摩尔注射物恒定热）
单位约定：浓度 mol/L，体积 L，热量 J，归一化热 J/mol（注射物）。
拟合：Nelder-Mead，多组确定性起点；评估次数受预算约束，耗尽即 BudgetExhausted。
"""
from __future__ import annotations

import numpy as np

from .domain import BudgetExhausted

MODEL_KINDS = ("one_site", "two_site", "dilution")


def displacement_concs(M0: float, Xs: float, V0_L: float, dvs_L: np.ndarray):
    """置换式滴定：每次注射后池内大分子/配体总浓度。"""
    Mt = np.empty(len(dvs_L))
    Xt = np.empty(len(dvs_L))
    m, x = M0, 0.0
    for i, dv in enumerate(dvs_L):
        f = dv / V0_L
        x = x * (1.0 - f) + f * Xs
        m = m * (1.0 - f)
        Mt[i], Xt[i] = m, x
    return Mt, Xt


def _per_injection(Q: np.ndarray, V0_L: float, dvs_L: np.ndarray, Xs: float) -> np.ndarray:
    """累计热 → 每次注射归一化热（J/mol），含置换校正。"""
    Qprev = np.concatenate([[0.0], Q[:-1]])
    dQ = Q - Qprev + (dvs_L / (2.0 * V0_L)) * (Q + Qprev)
    moles = Xs * dvs_L
    return dQ / moles


def heats_one_site(logK: float, dH: float, n: float,
                   M0: float, Xs: float, V0_L: float, dvs_L: np.ndarray) -> np.ndarray:
    K = 10.0 ** logK
    Mt, Xt = displacement_concs(M0, Xs, V0_L, dvs_L)
    b = 1.0 + K * (n * Mt - Xt)
    L = (-b + np.sqrt(b * b + 4.0 * K * Xt)) / (2.0 * K)
    Q = dH * V0_L * (Xt - L)  # J
    return _per_injection(Q, V0_L, dvs_L, Xs)


def _free_ligand_two_site(K1: float, K2: float, Mt: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    lo = np.zeros_like(Xt)
    hi = Xt.copy()
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        bound = Mt * (K1 * mid / (1.0 + K1 * mid) + K2 * mid / (1.0 + K2 * mid))
        need = mid + bound < Xt
        lo = np.where(need, mid, lo)
        hi = np.where(need, hi, mid)
    return 0.5 * (lo + hi)


def heats_two_site(logK1: float, dH1: float, logK2: float, dH2: float,
                   M0: float, Xs: float, V0_L: float, dvs_L: np.ndarray) -> np.ndarray:
    K1, K2 = 10.0 ** logK1, 10.0 ** logK2
    Mt, Xt = displacement_concs(M0, Xs, V0_L, dvs_L)
    L = _free_ligand_two_site(K1, K2, Mt, Xt)
    b1 = Mt * K1 * L / (1.0 + K1 * L)
    b2 = Mt * K2 * L / (1.0 + K2 * L)
    Q = V0_L * (dH1 * b1 + dH2 * b2)
    return _per_injection(Q, V0_L, dvs_L, Xs)


def heats_dilution(q0: float, count: int) -> np.ndarray:
    return np.full(count, q0)


def nelder_mead(f, x0, steps, max_evals: int, tol: float = 1e-7):
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += steps[i]
        simplex.append(x)
    simplex = np.array(simplex)
    evals = 0

    def ev(x):
        nonlocal evals
        if evals >= max_evals:
            raise BudgetExhausted(f"拟合评估预算 {max_evals} 次耗尽", location="nelder_mead")
        evals += 1
        return float(f(x))

    vals = np.array([ev(x) for x in simplex])
    for _ in range(2000):
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]
        spread = np.max(np.abs(simplex[1:] - simplex[0]) / (np.abs(simplex[0]) + 1e-30))
        if spread < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = ev(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = ev(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = ev(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                for i in range(1, n + 1):
                    vals[i] = ev(simplex[i])
    order = np.argsort(vals)
    return simplex[order[0]], float(vals[order[0]]), evals


def _rss(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sum((pred - obs) ** 2))


def fit_one_site(obs, M0, Xs, V0_L, dvs_L, max_evals):
    dH0 = -4.0e4 if float(np.mean(obs)) < 0 else 4.0e4

    def objective(x):
        logK, dH, n = x
        if not (1.0 <= logK <= 9.0 and 0.3 <= n <= 2.0 and abs(dH) < 1e7):
            return 1e30
        return _rss(heats_one_site(logK, dH, n, M0, Xs, V0_L, dvs_L), obs)

    starts = [(lk, dH0, 1.0) for lk in (4.5, 5.5, 6.5)]
    per = max(1, max_evals // len(starts))
    best = None
    for s in starts:
        x, v, _ = nelder_mead(objective, s, [0.5, abs(dH0) * 0.2, 0.1], per)
        if best is None or v < best[1]:
            best = (x, v)
    (logK, dH, n), rss = best
    return {"model": "one_site",
            "params": {"log10K": float(logK), "dH_J_mol": float(dH), "n": float(n)},
            "rss": rss,
            "predicted": heats_one_site(logK, dH, n, M0, Xs, V0_L, dvs_L).tolist()}


def fit_two_site(obs, M0, Xs, V0_L, dvs_L, max_evals, seed_params=None):
    dH0 = -4.0e4 if float(np.mean(obs)) < 0 else 4.0e4
    lk0 = float(seed_params["log10K"]) if seed_params else 6.0

    def objective(x):
        logK1, dH1, logK2, dH2 = x
        if not (1.0 <= logK1 <= 9.0 and 1.0 <= logK2 <= 9.0
                and abs(dH1) < 1e7 and abs(dH2) < 1e7):
            return 1e30
        return _rss(heats_two_site(logK1, dH1, logK2, dH2, M0, Xs, V0_L, dvs_L), obs)

    starts = [(lk0 + 0.3, dH0 * 0.7, lk0 - 1.2, dH0 * 0.3),
              (lk0 - 0.4, dH0 * 0.5, lk0 + 0.8, dH0 * 0.5)]
    per = max(1, max_evals // len(starts))
    best = None
    for s in starts:
        x, v, _ = nelder_mead(objective, s, [0.4, abs(dH0) * 0.2, 0.4, abs(dH0) * 0.2], per)
        if best is None or v < best[1]:
            best = (x, v)
    (logK1, dH1, logK2, dH2), rss = best
    return {"model": "two_site",
            "params": {"log10K1": float(logK1), "dH1_J_mol": float(dH1),
                       "log10K2": float(logK2), "dH2_J_mol": float(dH2)},
            "rss": rss,
            "predicted": heats_two_site(logK1, dH1, logK2, dH2, M0, Xs, V0_L, dvs_L).tolist()}


def fit_dilution(obs, count, max_evals):
    if max_evals < 1:
        raise BudgetExhausted("拟合评估预算耗尽", location="fit_dilution")
    q0 = float(np.mean(obs))
    return {"model": "dilution", "params": {"q_J_mol": q0},
            "rss": _rss(heats_dilution(q0, count), obs),
            "predicted": heats_dilution(q0, count).tolist()}


def fit_all(obs, M0, Xs, V0_L, dvs_L, max_evals: int = 20000) -> list[dict]:
    """拟合全部候选；预算由调用方给定，耗尽时抛出 BudgetExhausted。"""
    obs = np.asarray(obs, dtype=float)
    dvs_L = np.asarray(dvs_L, dtype=float)
    one = fit_one_site(obs, M0, Xs, V0_L, dvs_L, max_evals // 2)
    two = fit_two_site(obs, M0, Xs, V0_L, dvs_L, max_evals // 2,
                       seed_params=one["params"])
    dil = fit_dilution(obs, len(obs), max_evals)
    return [one, two, dil]
