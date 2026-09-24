"""Binding models for ITC isotherms.

Models compared by the review page:

* ``one_site`` - single set of equivalent sites (Wiseman isotherm)
* ``two_site`` - two independent site sets (sum of two one-site terms)
* ``dilution`` - heat of dilution only: q_i = a + b*dV_i, no binding

Deterministic fitting: fixed log-space starts + a compact Nelder-Mead. A
finite ``budget`` of model evaluations is honored; running out before
convergence yields ``status="budget_exhausted"``, a distinct outcome from
unparseable input or a semantically impossible problem.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# dH [kcal/mol] * concentration [uM] * V0 [uL] -> heat [uJ]
# 1e3 cal/mol * 1e-6 mol/L * 1e-6 L = 1e-9 cal = 4.184e-3 uJ
KCAL_MOL_UM_UL_TO_UJ = 4.184e-3
CAL_TO_J = 4.184

DEFAULT_BUDGET = 5000
TIE_RTOL = 0.02  # within 2% SSE -> statistically equivalent candidates


def bound_fraction(k: float, ligand: float, macromol: float, n: float) -> float:
    """Theta from 1:1 mass action with k in 1/uM."""
    sites = n * macromol
    if sites <= 0:
        return 0.0
    kd = 1.0 / k
    disc = (ligand + sites + kd) ** 2 - 4.0 * ligand * sites
    lb = 0.5 * (ligand + sites + kd - math.sqrt(max(disc, 0.0)))
    return min(max(lb / sites, 0.0), 1.0)


def cell_heat(x_conc: float, m0: float, v0: float, params: tuple) -> float:
    """Total heat content of the cell at cumulative ligand concentration x."""
    total = 0.0
    if len(params) == 3:  # one site: (log10k, dH, n)
        logk, dh, n = params
        total = dh * KCAL_MOL_UM_UL_TO_UJ * n * m0 * \
            bound_fraction(10 ** logk, x_conc, m0, n) * v0
    else:                 # two sites
        for off in (0, 3):
            logk, dh, n = params[off:off + 3]
            total += dh * KCAL_MOL_UM_UL_TO_UJ * n * m0 * \
                bound_fraction(10 ** logk, x_conc, m0, n) * v0
    return total


def wiseman_q(x_conc: np.ndarray, m0: float, v0: float,
              dv: np.ndarray, params: tuple, dilution: tuple | None = None
              ) -> np.ndarray:
    """Per-injection heat with the standard displacement correction.

    Q_i = V0*[Q'(X_i) - Q'(X_{i-1])] + dV_i*Q'(X_{i-1})  (MicroCal form),
    where Q' is heat density per cell volume and dV_i is injection volume.
    """
    densities = np.array([cell_heat(x, m0, v0, params) / v0
                          for x in x_conc])
    prev = np.concatenate([[0.0], densities[:-1]])
    q = v0 * (densities - prev) + dv * prev
    if dilution is not None:
        a, b = dilution
        q = q + a + b * dv
    return q


# ---------------------------------------------------------------------------
# Dilution-only model
# ---------------------------------------------------------------------------

def fit_dilution(dv: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    """q_i = a + b*dV_i solved by least squares (always solvable)."""
    A = np.column_stack([np.ones_like(dv), dv])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    pred = A @ coef
    sse = float(np.sum((q - pred) ** 2))
    return {"model": "dilution", "params": {"a_uJ": float(coef[0]),
                                            "b_uJ_per_uL": float(coef[1])},
            "prediction": pred.tolist(), "sse": sse, "status": "converged",
            "evaluations": 1}


# ---------------------------------------------------------------------------
# Deterministic Nelder-Mead
# ---------------------------------------------------------------------------

def nelder_mead(func, x0: np.ndarray, budget: int, *,
                step: float = 0.35, alpha=1.0, gamma=2.0, rho=0.5,
                sigma=0.5) -> tuple[np.ndarray, float, int, bool]:
    simplex = [np.asarray(x0, dtype=float)]
    n = len(x0)
    for j in range(n):
        x = np.array(x0, dtype=float)
        x[j] += step * max(1.0, abs(x[j]))
        simplex.append(x)
    vals = [None] * (n + 1)
    used = 0

    def f(x):
        nonlocal used
        if used >= budget:
            raise _Budget()
        used += 1
        return func(x)

    try:
        vals = [f(x) for x in simplex]
        while used < budget:
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            if vals[0] <= fr < vals[-2]:
                simplex[-1], vals[-1] = xr, fr
            elif fr < vals[0]:
                xe = centroid + gamma * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1], vals[-1] = xe, fe
                else:
                    simplex[-1], vals[-1] = xr, fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < vals[-1]:
                    simplex[-1], vals[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        vals[i] = f(simplex[i])
            spread = max(np.max(np.abs(s - simplex[0])) for s in simplex[1:])
            if spread < 1e-6:
                return np.asarray(simplex[0]), float(vals[0]), used, True
    except _Budget:
        i = int(np.argmin(vals))
        return np.asarray(simplex[i]), float(vals[i]), used, False
    i = int(np.argmin(vals))
    return np.asarray(simplex[i]), float(vals[i]), used, True


class _Budget(Exception):
    pass


# ---------------------------------------------------------------------------
# Binding fits
# ---------------------------------------------------------------------------

LOGK_GRID = (5.0, 6.0, 7.0, 8.0)     # 1/M log10 later; here 1/uM range
# concentrations in uM -> typical Kd 0.01..10 uM => log10 k [1/uM] in -2..2
LOGK_UM_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0)
DH_GRID = (-10.0, -2.0, 2.0)        # kcal/mol
N_GRID = (0.5, 1.0, 2.0)


def _fit_binding(model: str, x_conc: np.ndarray, m0: float, v0: float,
                 dv: np.ndarray, q: np.ndarray, budget: int
                 ) -> dict[str, Any]:
    n_sites_params = 3 if model == "one_site" else 6

    def decode(z: np.ndarray):
        if model == "one_site":
            return (z[0], z[1], z[2]), None
        p1 = (z[0], z[1], z[2])
        p2 = (z[3], z[4], z[5])
        return p1 + p2, None

    def objective(z: np.ndarray) -> float:
        if model == "one_site":
            logk, dh, n = z
            if n <= 0:
                return 1e30
            params = (logk, dh, n)
        else:
            if z[2] <= 0 or z[5] <= 0:
                return 1e30
            # canonicalize site order by affinity to keep stable identity
            if z[0] > z[3]:
                z = np.array([z[3], z[4], z[5], z[0], z[1], z[2]])
            params = tuple(z)
        pred = wiseman_q(x_conc, m0, v0, dv, params)
        return float(np.sum((pred - q) ** 2))

    starts: list[np.ndarray] = []
    if model == "one_site":
        for lk in LOGK_UM_GRID:
            for dh in DH_GRID[1:]:
                starts.append(np.array([lk, dh, 1.0]))
        starts.append(np.array([0.0, q.sum() / max(m0 * v0 * KCAL_MOL_UM_UL_TO_UJ, 1e-12), 1.0]))
    else:
        for lk1, lk2 in ((-1.0, 1.0), (0.0, 0.0)):
            starts.append(np.array([lk1, -5.0, 0.5, lk2, -5.0, 0.5]))
        starts.append(np.array([-1.0, -2.0, 1.0, 1.0, -2.0, 1.0]))

    remaining = budget
    best = None
    total_evals = 0
    converged = True
    for si, x0 in enumerate(starts):
        sub_budget = max(60, remaining // max(1, len(starts) - si))
        x, val, used, ok = nelder_mead(objective, x0, min(remaining, sub_budget))
        total_evals += used
        remaining -= used
        if not ok:
            converged = False
        if best is None or val < best[1]:
            best = (x, val)
        if remaining <= 0:
            break
    x, sse = best
    if model == "two_site" and x[0] > x[3]:
        x = np.array([x[3], x[4], x[5], x[0], x[1], x[2]])
    if model == "one_site":
        params = (float(x[0]), float(x[1]), float(x[2]))
        pred = wiseman_q(x_conc, m0, v0, dv, params)
        param_dict = {"log10_K_1_per_uM": params[0], "K_1_per_M": 10 ** params[0] * 1e6,
                      "dH_1_kcal_per_mol": params[1], "N_1": params[2]}
    else:
        params = tuple(float(v) for v in x)
        pred = wiseman_q(x_conc, m0, v0, dv, params)
        param_dict = {
            "log10_K1_1_per_uM": params[0], "K1_1_per_M": 10 ** params[0] * 1e6,
            "dH1_kcal_per_mol": params[1], "N1": params[2],
            "log10_K2_1_per_uM": params[3], "K2_1_per_M": 10 ** params[3] * 1e6,
            "dH2_kcal_per_mol": params[4], "N2": params[5]}
    return {"model": model, "params": param_dict, "param_vector": list(params),
            "prediction": pred.tolist(), "sse": float(sse),
            "evaluations": total_evals,
            "status": "converged" if converged and total_evals < budget
            else "budget_exhausted"}


def fit_models(q: np.ndarray, dv: np.ndarray, x_conc: np.ndarray, m0: float,
               v0: float, *, budget: int = DEFAULT_BUDGET,
               models=("one_site", "two_site", "dilution"),
               discard: set[int] | None = None,
               rtol: float = TIE_RTOL) -> dict[str, Any]:
    """Fit all candidates; return per-model results and equivalence classes.

    Discarded injections (e.g. first injection) are removed before fitting.
    Candidates within ``rtol`` of the best SSE are all returned as tied -
    no unique winner is invented.
    """
    keep = np.array([i not in (discard or set()) for i in range(len(q))])
    if keep.sum() < 2:
        return {"status": "semantic_unresolvable",
                "reason": "fewer than two usable injections after discard",
                "candidates": []}
    qk, dvk, xk = q[keep], dv[keep], x_conc[keep]
    candidates = []
    exhausted = False
    for model in models:
        if model == "dilution":
            r = fit_dilution(dvk, qk)
        else:
            r = _fit_binding(model, xk, m0, v0, dvk, qk, budget)
        r["used_indices"] = [int(i) for i, k in enumerate(keep) if k]
        r["discarded_indices"] = sorted(discard or set())
        if r["status"] == "budget_exhausted":
            exhausted = True
        candidates.append(r)
    candidates.sort(key=lambda c: (round(c["sse"], 10), c["model"]))
    best_sse = candidates[0]["sse"] if candidates else 0.0

    def equiv_key(sse: float) -> bool:
        return sse <= best_sse * (1.0 + rtol) + 1e-12 if best_sse > 0 else \
            abs(sse) <= 1e-9

    tied = [c["model"] for c in candidates if equiv_key(c["sse"])]
    for c in candidates:
        c["tied_best"] = c["model"] in tied
        c["rank"] = next(i + 1 for i, cc in enumerate(candidates)
                         if cc is c)
    return {"status": "multiple_equivalent" if len(tied) > 1 else
            ("budget_exhausted" if exhausted else "converged"),
            "best_models": tied, "candidates": candidates,
            "tie_rtol": rtol}
