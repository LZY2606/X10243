"""Binding-model candidates: one-site, two-site (multi-site) and a
dilution-only hypothesis, fitted with a NumPy Nelder-Mead.

Every fit records its input digest (integrals + concentration version +
blank rule). Revising concentrations makes existing fits stale but never
touches the underlying integrals. Models whose RSS is statistically
indistinguishable are ALL kept as equally-plausible candidates.
"""

import math

import numpy as np

from .errors import BudgetExhausted

MODEL_KINDS = ("one_site", "two_site", "dilution")


# ---------------------------------------------------------------- model math
def _cell_titration(volumes_uL, cell_conc_mM, syringe_conc_mM, cell_vol_mL):
    """Displacement dilution: cell concentration of macromolecule and
    ligand after each injection (constant cell volume)."""
    V0 = cell_vol_mL
    mt, xt = cell_conc_mM, 0.0
    mt_s, xt_s = [], []
    for v_uL in volumes_uL:
        v = v_uL / 1000.0
        frac = v / V0
        xt = xt * (1 - frac) + syringe_conc_mM * frac
        mt = mt * (1 - frac)
        mt_s.append(mt)
        xt_s.append(xt)
    return np.asarray(mt_s), np.asarray(xt_s)


def _site_heat(mt, xt, n, logK, dh_kJ, V0):
    """Cumulative heat (uJ) of one independent site class."""
    K = 10.0 ** logK                      # mM^-1
    nmt = n * mt
    nmt = np.where(nmt <= 0, 1e-12, nmt)
    a = 1.0 + xt / nmt + 1.0 / (K * nmt)
    root = np.sqrt(np.maximum(a * a - 4.0 * xt / nmt, 0.0))
    # mM * mL = umol ; umol * kJ/mol = 1e-3 J = 1000 uJ
    return nmt * V0 * dh_kJ * 1000.0 / 2.0 * (a - root)


def _injection_heats(cumulative, volumes_uL, cell_vol_mL):
    q = []
    prev = 0.0
    for i, v_uL in enumerate(volumes_uL):
        v = v_uL / 1000.0
        qi = cumulative[i] - prev + (v / cell_vol_mL) * (cumulative[i] + prev) / 2.0
        q.append(qi)
        prev = cumulative[i]
    return np.asarray(q)


def predict(model, params, volumes_uL, cell_conc_mM, syringe_conc_mM,
            cell_vol_mL):
    mt, xt = _cell_titration(volumes_uL, cell_conc_mM, syringe_conc_mM,
                             cell_vol_mL)
    if model == "one_site":
        n, logK, dh = params
        cum = _site_heat(mt, xt, n, logK, dh, cell_vol_mL)
        return _injection_heats(cum, volumes_uL, cell_vol_mL)
    if model == "two_site":
        n1, logK1, dh1, n2, logK2, dh2 = params
        cum = (_site_heat(mt, xt, n1, logK1, dh1, cell_vol_mL)
               + _site_heat(mt, xt, n2, logK2, dh2, cell_vol_mL))
        return _injection_heats(cum, volumes_uL, cell_vol_mL)
    if model == "dilution":
        (q_per_uL,) = params
        return np.asarray(volumes_uL) * q_per_uL
    raise ValueError(model)


PARAM_SPEC = {
    "one_site": (["n", "logK", "dH_kJ_mol"],
                 [1.0, 3.0, -40.0], [0.3, 1.0, 15.0]),
    "two_site": (["n1", "logK1", "dH1_kJ_mol", "n2", "logK2", "dH2_kJ_mol"],
                 [0.5, 3.5, -30.0, 0.5, 1.5, -10.0],
                 [0.3, 0.8, 15.0, 0.3, 0.8, 15.0]),
    "dilution": (["q_uJ_per_uL"], [-1.0], [1.0]),
}


# ---------------------------------------------------------------- optimizer
def nelder_mead(func, x0, step, max_evals=2000, tol=1e-9):
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += step[i]
        simplex.append(x)
    simplex = np.asarray(simplex)
    fvals = np.asarray([func(x) for x in simplex])
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(fvals, kind="stable")
        simplex, fvals = simplex[order], fvals[order]
        if np.max(np.abs(fvals - fvals[0])) < tol and \
           np.max(np.abs(simplex - simplex[0])) < 1e-7:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = func(xr)
        evals += 1
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = func(xe)
            evals += 1
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        xc = centroid + 0.5 * (worst - centroid)
        fc = func(xc)
        evals += 1
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        best = simplex[0].copy()
        simplex = best + 0.5 * (simplex - best)
        fvals = np.asarray([func(x) for x in simplex])
        evals += n
    order = np.argsort(fvals, kind="stable")
    return simplex[order[0]], fvals[order[0]], evals


def fit_model(model, obs_uJ, volumes_uL, cell_conc_mM, syringe_conc_mM,
              cell_vol_mL, max_evals=4000, starts=None):
    names, x0, step = PARAM_SPEC[model]
    obs = np.asarray(obs_uJ, dtype=float)

    def rss(params):
        pred = predict(model, params, volumes_uL, cell_conc_mM,
                       syringe_conc_mM, cell_vol_mL)
        r = pred - obs
        return float(np.dot(r, r))

    starts = starts or [x0]
    best = None
    total_evals = 0
    for s in starts:
        remaining = max_evals - total_evals
        if remaining <= len(s) + 1:
            raise BudgetExhausted(
                f"fit budget exhausted after {total_evals} evaluations",
                diagnostics=[{"code": "E_BUDGET_FIT",
                              "message": "max_evals reached during fit",
                              "location": {"model": model},
                              "severity": "error"}])
        x, f, ev = nelder_mead(rss, s, step, max_evals=remaining)
        total_evals += ev
        if best is None or f < best[1]:
            best = (x, f)
    params = {k: float(v) for k, v in zip(names, best[0])}
    pred = predict(model, best[0], volumes_uL, cell_conc_mM,
                   syringe_conc_mM, cell_vol_mL)
    return {"model": model, "params": params, "rss": float(best[1]),
            "evals": total_evals,
            "residuals": [float(v) for v in (obs - pred)],
            "predicted": [float(v) for v in pred]}


def group_equally_plausible(fits, rel_tol=0.02, abs_tol=1.0):
    """Mark every fit whose RSS is indistinguishable from the best."""
    if not fits:
        return fits
    best = min(f["rss"] for f in fits)
    gid = "eqg-1"
    for f in fits:
        f["equally_plausible"] = (f["rss"] - best) <= max(abs_tol,
                                                          rel_tol * best)
        f["group_id"] = gid if f["equally_plausible"] else None
    return fits
