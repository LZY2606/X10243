"""Binding-model candidates: one-site, two-site, dilution-only.

Wiseman / MicroCal convention with cell displacement correction:

* Q_i   = M_cell V_cell dH * theta(X_i)                     (cumulative heat)
* dQ_i  = Q_i - Q_{i-1} + v_i/(2V_cell) * (Q_i + Q_{i-1})
* y_i   = dQ_i / ([L]_inj v_i)                              (J per mol injected)

theta is nonlinear in K and stoichiometry n; dH is a linear coefficient solved
by least squares.  Deterministic grid + Nelder-Mead, NumPy only.
"""

from __future__ import annotations

import math

import numpy as np

from .budget import Budget
from .errors import SemanticNoSolution
from .util import digest

TIE_DELTA_AIC = 2.0
MIN_POINTS = 4
CAL4 = 4.184


def _geometry(norm):
    pts = sorted((r for r in norm["rows"] if r["usable"]), key=lambda r: r["ordinal"])
    if len(pts) < MIN_POINTS:
        raise SemanticNoSolution(
            f"need >= {MIN_POINTS} usable injections, found {len(pts)}",
            diagnostics=[{"code": "INSUFFICIENT_USABLE_POINTS",
                          "usable": len(pts), "required": MIN_POINTS}],
        )
    V = float(norm["cell_volume_l"])
    M = float(norm["macromolecule_concentration_m"])
    L0 = float(norm["ligand_concentration_m"])
    vols = np.asarray([float(r["volume_l"]) for r in pts])
    y = np.asarray([float(r["q_per_mol_j"]) for r in pts])
    cum_v = np.cumsum(vols)
    L_free0 = L0 * (cum_v / V) / (1.0 - cum_v / V)
    x = L_free0 / M
    return pts, V, M, L0, vols, cum_v, x, y


def _theta(x, logk, n, M):
    k = math.exp(min(logk, 600.0))
    c = k * M
    a = x + n + 1.0 / c
    root = np.maximum(a * a - 4.0 * n * x, 0.0)
    return 0.5 * (a - np.sqrt(root))


def _lstsq(cols, y, vols, L0):
    A = np.column_stack(cols) / (L0 * vols)[:, None]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    return coef, pred, resid, float(np.sum(resid * resid))


def _aic(sse, n, k):
    if sse <= 0 or not np.isfinite(sse):
        sse = 1e-30
    base = n * math.log(sse / n) + 2 * k
    if n - k - 1 > 0:
        base += 2 * k * (k + 1) / (n - k - 1)
    return base


def _nelder_mead(fn, x0, *, budget: Budget, steps=80):
    """Deterministic Nelder-Mead. fn(z)->float (sse)."""
    x0 = np.asarray(x0, dtype=float)
    m = x0.size
    simplex = [x0]
    for i in range(m):
        z = x0.copy()
        z[i] += 0.15 * max(abs(z[i]), 0.5)
        simplex.append(z)
    vals = [fn(z) for z in simplex]
    for _ in range(steps):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = np.mean(np.asarray(simplex[:-1]), axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        budget.tick()
        fr = fn(reflected)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = reflected, fr
        elif fr < vals[0]:
            expanded = centroid + 2 * (reflected - centroid)
            budget.tick()
            fe = fn(expanded)
            if fe < fr:
                simplex[-1], vals[-1] = expanded, fe
            else:
                simplex[-1], vals[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            budget.tick()
            fc = fn(contracted)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = contracted, fc
            else:
                best = simplex[0]
                simplex = [best] + [best + 0.5 * (z - best) for z in simplex[1:]]
                vals = [fn(z) for z in simplex]
                for _v in vals:
                    budget.tick()
    order = np.argsort(vals)
    return np.asarray(simplex[order[0]]), vals[order[0]]


def fit_one_site(norm, budget: Budget):
    pts, V, M, L0, vols, cum_v, x, y = _geometry(norm)

    def sse_of(z):
        logk, n = z
        n = max(n, 0.05)
        theta = _theta(x, logk, n, M)
        _, _, _, sse = _lstsq([M * V * _dq(theta, vols, V)], y, vols, L0)
        return sse

    best = None
    for lk in np.linspace(math.log(1e3), math.log(1e9), 41):
        for n in np.linspace(0.5, 6.0, 13):
            budget.tick()
            sse = sse_of((lk, n))
            if best is None or sse < best[0]:
                best = (sse, (float(lk), float(n)))
    z, sse = _nelder_mead(sse_of, best[1], budget=budget, steps=60)
    logk, n = float(z[0]), max(float(z[1]), 0.05)
    theta = _theta(x, logk, n, M)
    coef, pred, resid, sse = _lstsq([M * V * _dq(theta, vols, V)], y, vols, L0)
    return _result("one_site", {"K_m3": math.exp(logk), "n_sites": n,
                                "dH_j_mol": float(coef[0]),
                                "dH_kcal_mol": float(coef[0]) / CAL4 / 1000.0},
                   sse, len(pts), 3, pts, y, pred, resid)


def _dq(theta, vols, V):
    q = theta
    q_prev = np.concatenate([[0.0], q[:-1]])
    return (q - q_prev) + (vols / (2.0 * V)) * (q + q_prev)


def fit_two_site(norm, budget: Budget):
    pts, V, M, L0, vols, cum_v, x, y = _geometry(norm)

    def sse_of(z):
        lk1, lk2, n1, n2 = z
        n1, n2 = max(n1, 0.05), max(n2, 0.05)
        th1 = _theta(x, lk1, n1, M)
        th2 = _theta(x, lk2, n2, M)
        cols = [M * V * _dq(th1, vols, V), M * V * _dq(th2, vols, V)]
        _, _, _, sse = _lstsq(cols, y, vols, L0)
        return sse

    best = None
    grid = np.linspace(math.log(1e4), math.log(1e8), 9)
    for lk1 in grid:
        for lk2 in grid:
            budget.tick()
            sse = sse_of((lk1, lk2, 1.0, 1.0))
            if best is None or sse < best[0]:
                best = (sse, (float(lk1), float(lk2), 1.0, 1.0))
    z, sse = _nelder_mead(sse_of, best[1], budget=budget, steps=70)
    lk1, lk2 = float(z[0]), float(z[1])
    n1, n2 = max(float(z[2]), 0.05), max(float(z[3]), 0.05)
    th1, th2 = _theta(x, lk1, n1, M), _theta(x, lk2, n2, M)
    cols = [M * V * _dq(th1, vols, V), M * V * _dq(th2, vols, V)]
    coef, pred, resid, sse = _lstsq(cols, y, vols, L0)
    return _result("two_site", {
        "K1_m3": math.exp(lk1), "K2_m3": math.exp(lk2),
        "n1_sites": n1, "n2_sites": n2,
        "dH1_j_mol": float(coef[0]), "dH2_j_mol": float(coef[1]),
        "dH1_kcal_mol": float(coef[0]) / CAL4 / 1000.0,
        "dH2_kcal_mol": float(coef[1]) / CAL4 / 1000.0,
    }, sse, len(pts), 6, pts, y, pred, resid)


def fit_dilution(norm, budget: Budget):
    pts, V, M, L0, vols, cum_v, x, y = _geometry(norm)
    for _ in pts:
        budget.tick()
    A = np.ones((len(pts), 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    sse = float(np.sum(resid * resid))
    return _result("dilution", {"dilution_heat_j_mol": float(coef[0]),
                                "dilution_heat_kcal_mol": float(coef[0]) / CAL4 / 1000.0},
                   sse, len(pts), 1, pts, y, pred, resid)


def _result(model, params, sse, n, k, pts, y, pred, resid):
    sst = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
    r2 = 1.0 - sse / sst if sst > 0 else None
    return {
        "model": model, "params": params,
        "sse_j2_mol2": float(sse), "aic": _aic(sse, n, k),
        "n_points": n, "k_params": k, "r2": r2,
        "ordinals": [r["ordinal"] for r in pts],
        "observed_j_mol": [float(v) for v in y],
        "predicted_j_mol": [float(v) for v in pred],
        "residual_j_mol": [float(v) for v in resid],
    }


def fit_all(norm, budget: Budget | None = None, *, models=("one_site", "two_site", "dilution")):
    budget = budget or Budget("model_evaluations", 200_000)
    fitters = {"one_site": fit_one_site, "two_site": fit_two_site,
               "dilution": fit_dilution}
    results = [fitters[m](norm, budget) for m in models]
    best_aic = min(r["aic"] for r in results)
    for r in results:
        r["delta_aic"] = r["aic"] - best_aic
        r["equal_best"] = r["delta_aic"] <= TIE_DELTA_AIC
    results.sort(key=lambda r: (round(r["aic"], 9), r["model"]))
    equal = [r["model"] for r in results if r["equal_best"]]
    for rank, r in enumerate(results):
        r["rank"] = rank + 1
        r["tie_group"] = equal
    chosen = None if len(equal) > 1 else equal[0]
    return {"fits": results, "equal_best_models": equal,
            "unique_choice": chosen,
            "tie": len(equal) > 1,
            "tie_threshold_delta_aic": TIE_DELTA_AIC}


def stable_model_id(series_id, conc_digest, decision_digest, model, params):
    return digest(["model", series_id, conc_digest, decision_digest, model,
                   {k: round(v, 9) if isinstance(v, float) else v
                    for k, v in params.items()}], prefix="fit_")
