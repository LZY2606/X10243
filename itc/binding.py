"""Binding-model candidates and least-squares fitting (NumPy only).

Models (per-injection heats in uJ, displacement-corrected):
  * one_site   - 1:1 binding, params logK (K in M^-1), dH (kJ/mol), n,
                 q_dil (uJ constant dilution offset)
  * two_site   - two independent sets of sites
  * dilution   - constant heat per injection (dilution-only hypothesis)

The fitter keeps *all* legal candidates: distinct optima whose RSS is
within tolerance of the best are flagged ``equally_good`` instead of
being silently discarded.  A fit that exceeds its evaluation budget
reports status ``budget_exhausted`` - distinct from ``invalid_format``
and ``unsatisfiable``.
"""

from __future__ import annotations

import math

import numpy as np

PARAM_VERSION = "binding/1"

MODEL_PARAMS = {
    "one_site": ["logK", "dH", "n", "q_dil"],
    "two_site": ["logK1", "dH1", "n1", "logK2", "dH2", "n2", "q_dil"],
    "dilution": ["q_dil"],
}

DEFAULT_STARTS = {
    "one_site": [
        [5.0, -30.0, 1.0, 0.0],
        [6.0, -40.0, 1.0, 0.0],
        [7.0, -20.0, 0.9, 0.0],
        [4.0, -60.0, 1.1, 0.0],
    ],
    "two_site": [
        [6.0, -30.0, 0.5, 4.5, -10.0, 0.5, 0.0],
        [7.0, -40.0, 0.6, 5.0, -15.0, 0.4, 0.0],
    ],
    "dilution": [[0.0]],
}

EQUAL_RSS_RTOL = 0.05
EQUAL_RSS_ATOL = 1e-9


def _sites(kind, params):
    if kind == "one_site":
        return [(params["logK"], params["dH"], params["n"])]
    if kind == "two_site":
        return [(params["logK1"], params["dH1"], params["n1"]),
                (params["logK2"], params["dH2"], params["n2"])]
    return []


def predict_heats(kind, params, volumes_uL, M0_mM, Xsyr_mM, V0_mL):
    """Per-injection heat (uJ) with displacement correction."""
    V0_uL = V0_mL * 1000.0
    Xt, Mt = 0.0, float(M0_mM)
    sites = _sites(kind, params)
    prev = [0.0] * len(sites)
    out = []
    for v in volumes_uL:
        f = float(v) / V0_uL
        Xt = Xt * (1.0 - f) + Xsyr_mM * f
        Mt = Mt * (1.0 - f)
        q = float(params.get("q_dil", 0.0))
        for s, (logK, dH, ns) in enumerate(sites):
            Kd_mM = 10.0 ** (3.0 - logK)
            Ms = ns * Mt
            S = Ms + Xt + Kd_mM
            disc = max(S * S - 4.0 * Ms * Xt, 0.0)
            b = 0.5 * (S - math.sqrt(disc))  # bound sites, mM
            q += 1000.0 * dH * V0_mL * (b - (1.0 - f) * prev[s])
            prev[s] = b
        out.append(q)
    return out


def _nelder_mead(func, x0, step, max_evals, tol=1e-10):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += step if np.isscalar(step) else step[i]
        simplex.append(x)
    simplex = np.array(simplex)
    f = np.array([func(x) for x in simplex])
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(f)
        simplex, f = simplex[order], f[order]
        if (np.max(np.abs(simplex[1:] - simplex[0])) < tol
                and np.max(np.abs(f[1:] - f[0])) < tol * max(1.0, abs(f[0]))):
            return simplex[0], float(f[0]), evals, True
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = func(xr)
        evals += 1
        if f[0] <= fr < f[-2]:
            simplex[-1], f[-1] = xr, fr
            continue
        if fr < f[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = func(xe)
            evals += 1
            if fe < fr:
                simplex[-1], f[-1] = xe, fe
            else:
                simplex[-1], f[-1] = xr, fr
            continue
        if fr < f[-1]:
            xc = centroid + 0.5 * (xr - centroid)
            fc = func(xc)
            evals += 1
            if fc <= fr:
                simplex[-1], f[-1] = xc, fc
                continue
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            evals += 1
            if fc < f[-1]:
                simplex[-1], f[-1] = xc, fc
                continue
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            f[i] = func(simplex[i])
            evals += 1
            if evals >= max_evals:
                break
    order = np.argsort(f)
    simplex, f = simplex[order], f[order]
    return simplex[0], float(f[0]), evals, False


def _rss(kind, vec, names, volumes, M0, Xsyr, V0, obs_idx, obs_heat):
    params = dict(zip(names, vec))
    pred = predict_heats(kind, params, volumes, M0, Xsyr, V0)
    return float(sum((pred[i] - h) ** 2 for i, h in zip(obs_idx, obs_heat)))


def fit_all(volumes_uL, obs_idx, obs_heat, M0_mM, Xsyr_mM, V0_mL,
            budget=4000, kinds=("one_site", "two_site", "dilution")):
    """Fit every requested model kind; keep all equally-good candidates.

    Returns {"status", "candidates", "diagnostics"} where status is one of
    ok / unsatisfiable / budget_exhausted.
    """
    diagnostics = []
    candidates = []
    min_points = min(len(MODEL_PARAMS[k]) for k in kinds)
    if len(obs_idx) < min_points:
        return {"status": "unsatisfiable", "candidates": [],
                "diagnostics": [{
                    "code": "insufficient_points",
                    "location": "model_fit",
                    "message": f"{len(obs_idx)} usable injection heats "
                               f"cannot constrain {min_points}+ parameters"}]}
    budget_hit = False
    for kind in kinds:
        names = MODEL_PARAMS[kind]
        if len(obs_idx) < len(names):
            diagnostics.append({
                "code": "model_skipped", "location": f"model_fit.{kind}",
                "message": f"not enough usable points ({len(obs_idx)}) for "
                           f"{len(names)} parameters"})
            continue
        best_for_kind = []
        for start in DEFAULT_STARTS[kind]:
            x, fval, evals, converged = _nelder_mead(
                lambda v: _rss(kind, v, names, volumes_uL, M0_mM, Xsyr_mM,
                               V0_mL, obs_idx, obs_heat),
                start, step=0.4, max_evals=budget)
            if not converged and evals >= budget:
                budget_hit = True
            best_for_kind.append((x, fval))
        # dedupe optima that landed on the same point
        uniq = []
        for x, fval in sorted(best_for_kind, key=lambda t: t[1]):
            if all(np.max(np.abs(x - ux)) > 1e-4 for ux, _ in uniq):
                uniq.append((x, fval))
        for x, fval in uniq:
            params = {n: float(v) for n, v in zip(names, x)}
            candidates.append({
                "kind": kind, "params": params, "rss": float(fval),
                "predicted": predict_heats(kind, params, volumes_uL,
                                           M0_mM, Xsyr_mM, V0_mL),
            })
    if not candidates:
        return {"status": "unsatisfiable", "candidates": [],
                "diagnostics": diagnostics or [{
                    "code": "no_candidate", "location": "model_fit",
                    "message": "no model kind could be fitted"}]}
    best = min(c["rss"] for c in candidates)
    for c in candidates:
        c["equally_good"] = bool(
            c["rss"] <= best * (1.0 + EQUAL_RSS_RTOL) + EQUAL_RSS_ATOL)
    # stable order: rss, then kind, then rounded parameters
    candidates.sort(key=lambda c: (
        round(c["rss"], 12), c["kind"],
        tuple(round(v, 8) for v in c["params"].values())))
    status = "budget_exhausted" if budget_hit else "ok"
    return {"status": status, "candidates": candidates,
            "diagnostics": diagnostics}
