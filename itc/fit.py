"""Binding-model candidates: one-site, two-site and dilution-only.

All candidates whose RSS ties within tolerance are retained (equal-best
multiplicity is a first-class result, never silently collapsed).  Ordering
is fully deterministic: identical inputs always yield identical order.
"""
from __future__ import annotations

import math

PARAM_VERSION = "fit-1.0"
EQUAL_RTOL = 1e-6
EQUAL_ATOL = 1e-9
MAX_CANDIDATES = 8


class BudgetExhausted(Exception):
    pass


class _Budget:
    def __init__(self, max_evals):
        self.max_evals = max_evals
        self.used = 0

    def charge(self):
        self.used += 1
        if self.used > self.max_evals:
            raise BudgetExhausted()


def cell_conditions(injections, cell_conc_mM, syringe_conc_mM, cell_vol_uL):
    """Total cell / syringe-side concentrations after each injection
    (constant-volume displacement model)."""
    mt, xt, rows = cell_conc_mM, 0.0, []
    for inj in injections:
        v = inj["volume_uL"]
        keep = 1.0 - v / cell_vol_uL
        mt = mt * keep
        xt = xt * keep + syringe_conc_mM * v / cell_vol_uL
        rows.append({"mt_mM": mt, "xt_mM": xt, "v_uL": v})
    return rows


def _bound_mM(K, mt, xt):
    """Bound ligand concentration (mM) for 1:1 equilibrium."""
    a = K * (mt + xt) + 1.0
    disc = a * a - 4.0 * K * K * mt * xt
    return (a - math.sqrt(max(disc, 0.0))) / (2.0 * K)


def model_heats(model, theta, cond, cell_vol_uL):
    """theta layout:
    one_site:        [n, log10K, dH, qd]
    two_site:        [n1, log10K1, dH1, n2, log10K2, dH2, qd]
    dilution_const:  [qd]
    dH in uJ/nmol (== kJ/mol); heats in uJ."""
    heats = []
    prev_b = (0.0, 0.0) if model == "two_site" else 0.0
    for c in cond:
        keep = 1.0 - c["v_uL"] / cell_vol_uL
        if model == "one_site":
            n, logK, dH, qd = theta
            b = _bound_mM(10 ** logK, c["mt_mM"] * n, c["xt_mM"])
            q = cell_vol_uL * dH * (b - prev_b * keep) + qd
            prev_b = b
        elif model == "two_site":
            n1, logK1, dH1, n2, logK2, dH2, qd = theta
            b1 = _bound_mM(10 ** logK1, c["mt_mM"] * n1, c["xt_mM"])
            b2 = _bound_mM(10 ** logK2, c["mt_mM"] * n2, c["xt_mM"])
            q = (cell_vol_uL * dH1 * (b1 - prev_b[0] * keep)
                 + cell_vol_uL * dH2 * (b2 - prev_b[1] * keep) + qd)
            prev_b = (b1, b2)
        elif model == "dilution_const":
            q = theta[0]
        else:
            raise ValueError(model)
        heats.append(q)
    return heats


def rss(observed, predicted):
    return sum((o - p) ** 2 for o, p in zip(observed, predicted)
               if o is not None)


_GRIDS = {
    "one_site": [
        [0.8, 1.0, 1.2],          # n
        [1.0, 2.0, 3.0],          # log10 K (1/mM)
        [-20.0, -5.0, 5.0],       # dH uJ/nmol
        [0.0],                    # qd
    ],
    "two_site": [
        [1.0], [2.0], [-10.0, 5.0],
        [1.0], [2.0, 3.0], [-10.0, 5.0], [0.0],
    ],
    "dilution_const": [[-5.0, -1.0, 0.0, 1.0, 5.0]],
}

_STEPS = {
    "one_site": [0.1, 0.5, 2.0, 0.5],
    "two_site": [0.1, 0.5, 2.0, 0.1, 0.5, 2.0, 0.5],
    "dilution_const": [0.5],
}


def _grid_points(model):
    grids = _GRIDS[model]
    pts = [[]]
    for axis in grids:
        pts = [p + [v] for p in pts for v in axis]
    return pts


def _nelder_mead(fn, x0, steps, budget, tol=1e-12, max_iter=300):
    """Deterministic Nelder-Mead.  Ties broken by coordinates so identical
    inputs always take the identical path."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += steps[i]
        simplex.append(pt)
    vals = []
    for pt in simplex:
        vals.append(fn(pt))
        budget.charge()
    for _ in range(max_iter):
        order = sorted(range(n + 1),
                       key=lambda i: (vals[i], tuple(simplex[i])))
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * max(1.0, abs(vals[0])):
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n
                    for j in range(n)]
        worst = simplex[-1]
        xr = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = fn(xr)
        budget.charge()
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = [c + 2.0 * (r - c) for c, r in zip(centroid, xr)]
            fe = fn(xe)
            budget.charge()
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
            continue
        xc = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
        fc = fn(xc)
        budget.charge()
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = [b + 0.5 * (p - b) for b, p in zip(best, simplex[i])]
            vals[i] = fn(simplex[i])
            budget.charge()
    order = sorted(range(n + 1), key=lambda i: (vals[i], tuple(simplex[i])))
    return simplex[order[0]], vals[order[0]]


def _refine(model, theta, observed, cond, cell_vol_uL, budget):
    def fn(x):
        return rss(observed, model_heats(model, x, cond, cell_vol_uL))
    return _nelder_mead(fn, theta, _STEPS[model], budget)


def fit_models(observed, cond, cell_vol_uL, models=None, budget=20000):
    """observed: list of corrected heats (uJ), None = excluded.
    Returns dict with status, candidates (stable order), diagnostics."""
    models = models or ["one_site", "two_site", "dilution_const"]
    valid = [i for i, o in enumerate(observed) if o is not None]
    diagnostics = []
    if cell_vol_uL is None or cell_vol_uL <= 0:
        return {"status": "semantic_unsolvable", "candidates": [],
                "diagnostics": [{
                    "severity": "error", "position": "cell_volume",
                    "code": "cell_volume_invalid",
                    "message": "cell volume missing or non-positive; "
                               "binding isotherm undefined"}]}
    if len(valid) < 2:
        return {"status": "semantic_unsolvable", "candidates": [],
                "diagnostics": [{
                    "severity": "error", "position": "injections",
                    "code": "insufficient_valid_injections",
                    "message": f"only {len(valid)} usable injection(s); "
                               "at least 2 required for fitting"}]}
    cond_v = [cond[i] for i in valid]
    obs_v = [observed[i] for i in valid]
    pool, exhausted = [], False
    b = _Budget(budget)
    try:
        for model in models:
            for start in _grid_points(model):
                theta, val = _refine(model, start, obs_v, cond_v,
                                     cell_vol_uL, b)
                pool.append({"model": model, "theta": theta, "rss": val,
                             "n_params": len(theta)})
    except BudgetExhausted:
        exhausted = True
    if not pool:
        return {"status": "budget_exhausted", "candidates": [],
                "evaluations": b.used,
                "diagnostics": [{
                    "severity": "error", "position": "fit",
                    "code": "budget_exhausted",
                    "message": f"evaluation budget {budget} exhausted "
                               "before any candidate completed"}]}
    best = min(c["rss"] for c in pool)
    tol = EQUAL_ATOL + EQUAL_RTOL * max(abs(best), 1.0)
    kept, seen = [], set()
    for c in sorted(pool, key=_sort_key):
        if c["rss"] <= best + tol:
            sig = (c["model"],) + tuple(round(x, 6) for x in c["theta"])
            if sig not in seen:
                seen.add(sig)
                kept.append(c)
        if len(kept) >= MAX_CANDIDATES:
            break
    for rank, c in enumerate(kept):
        c["rank"] = rank
        c["equal_best"] = c["rss"] <= best + tol
        c["theta"] = [round(x, 9) for x in c["theta"]]
    status = "budget_exhausted" if exhausted else "ok"
    if exhausted:
        diagnostics.append({
            "severity": "warning", "position": "fit",
            "code": "budget_exhausted",
            "message": f"evaluation budget {budget} exhausted; candidates "
                       "are partial"})
    return {"status": status, "candidates": kept, "evaluations": b.used,
            "best_rss": best, "diagnostics": diagnostics}


def _sort_key(c):
    return (round(c["rss"], 9), c["model"],
            tuple(round(x, 6) for x in c["theta"]))
