"""Binding-model candidates: one-site, two-site (multi-site) and dilution.

NumPy-only deterministic multi-start Levenberg-Marquardt. Equivalent inputs
always produce the same candidates in the same order. Distinct local minima
are all kept; the equally-good decision is made by the caller with an
explicit SSE tolerance.
"""
from __future__ import annotations

import numpy as np

from .results import ResourceExhausted

MODEL_KINDS = ("one_site", "two_site", "dilution")


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_evals):
        self.left = max_evals

    def use(self, n=1):
        self.left -= n
        if self.left < 0:
            raise BudgetExceeded("model evaluation budget exhausted")


def dilution_factors(vols_uL, V0_uL):
    f = 1.0
    out = np.empty(len(vols_uL))
    for i, v in enumerate(vols_uL):
        f *= 1.0 - float(v) / V0_uL
        out[i] = f
    return out


def cell_concentrations(M0_uM, X0_uM, vols_uL, V0_uL):
    f = dilution_factors(vols_uL, V0_uL)
    return M0_uM * f, X0_uM * (1.0 - f)


def _q_cum_site(Mt, Xt, V0_uL, n, K_inv_uM, dH_uJ_mol):
    nMt = np.maximum(n * Mt, 1e-9)
    Xr = Xt / nMt
    c = np.maximum(K_inv_uM * nMt, 1e-9)
    a = 1.0 + Xr + 1.0 / c
    bracket = a - np.sqrt(np.maximum(a * a - 4.0 * Xr, 0.0))
    moles = nMt * V0_uL * 1e-12  # uM * uL -> mol
    return moles * dH_uJ_mol / 2.0 * bracket


def per_injection_heat(Qcum, vols_uL, V0_uL):
    dQ = np.empty(len(vols_uL))
    qprev = 0.0
    for i in range(len(vols_uL)):
        dQ[i] = Qcum[i] - qprev + (vols_uL[i] / (2.0 * V0_uL)) * (Qcum[i] + qprev)
        qprev = Qcum[i]
    return dQ


def model_heats(kind, params, Mt, Xt, vols_uL, V0_uL):
    """params use scaled forms: [n, log10(K/uM^-1), dH/1e10 (uJ/mol)]."""
    if kind == "one_site":
        n, logK, dHs = params
        q = _q_cum_site(Mt, Xt, V0_uL, n, 10.0 ** logK, dHs * 1e10)
        return per_injection_heat(q, vols_uL, V0_uL)
    if kind == "two_site":
        n1, l1, h1, n2, l2, h2 = params
        q = (_q_cum_site(Mt, Xt, V0_uL, n1, 10.0 ** l1, h1 * 1e10)
             + _q_cum_site(Mt, Xt, V0_uL, n2, 10.0 ** l2, h2 * 1e10))
        return per_injection_heat(q, vols_uL, V0_uL)
    if kind == "dilution":
        return np.full(len(vols_uL), params[0])
    raise ValueError(f"unknown model kind {kind}")


def named_params(kind, params):
    if kind == "one_site":
        n, logK, dHs = params
        return {"n": float(n), "K_inv_uM": float(10.0 ** logK), "dH_uJ_mol": float(dHs * 1e10)}
    if kind == "two_site":
        n1, l1, h1, n2, l2, h2 = params
        return {
            "n1": float(n1), "K1_inv_uM": float(10.0 ** l1), "dH1_uJ_mol": float(h1 * 1e10),
            "n2": float(n2), "K2_inv_uM": float(10.0 ** l2), "dH2_uJ_mol": float(h2 * 1e10),
        }
    if kind == "dilution":
        return {"q_dilution_uJ": float(params[0])}
    raise ValueError(kind)


def params_vector(kind, named):
    if kind == "one_site":
        return [named["n"], np.log10(named["K_inv_uM"]), named["dH_uJ_mol"] / 1e10]
    if kind == "two_site":
        return [named["n1"], np.log10(named["K1_inv_uM"]), named["dH1_uJ_mol"] / 1e10,
                named["n2"], np.log10(named["K2_inv_uM"]), named["dH2_uJ_mol"] / 1e10]
    if kind == "dilution":
        return [named["q_dilution_uJ"]]
    raise ValueError(kind)


def _lm(model, y, p0, budget, max_iter=200, tol=1e-12):
    p = np.asarray(p0, dtype=float)
    fp = model(p)
    budget.use()
    r = y - fp
    sse = float(r @ r)
    lam = 1e-2
    for _ in range(max_iter):
        J = np.empty((len(y), len(p)))
        for j in range(len(p)):
            h = 1e-6 * max(abs(p[j]), 1e-3)
            pj = p.copy()
            pj[j] += h
            J[:, j] = (model(pj) - fp) / h
            budget.use()
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(40):
            try:
                dp = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            pnew = p + dp
            fnew = model(pnew)
            budget.use()
            rnew = y - fnew
            sse_new = float(rnew @ rnew)
            if sse_new < sse:
                done = abs(sse - sse_new) < tol * max(sse, 1e-30)
                p, fp, r, sse = pnew, fnew, rnew, sse_new
                lam = max(lam / 3.0, 1e-9)
                improved = True
                if done:
                    return p, sse
                break
            lam *= 3.0
        if not improved:
            break
    return p, sse


_ONE_SITE_STARTS = tuple(
    (n, l, h)
    for n in (0.6, 1.0, 1.4)
    for l in (-4.0, -3.0, -2.0)
    for h in (-6.0, -2.0, 2.0, 6.0)
)
_TWO_SITE_STARTS = (
    (0.9, -3.0, -4.0, 0.3, -5.0, 1.0),
    (0.3, -5.0, 1.0, 0.9, -3.0, -4.0),
    (0.5, -3.0, -2.0, 0.5, -4.0, -2.0),
    (1.2, -3.5, -5.0, 0.2, -2.0, 2.0),
    (0.2, -2.0, 2.0, 1.2, -3.5, -5.0),
    (0.7, -2.5, -3.0, 0.7, -4.5, 0.5),
)


def _starts_for(kind, y):
    if kind == "one_site":
        return _ONE_SITE_STARTS
    if kind == "two_site":
        return _TWO_SITE_STARTS
    if kind == "dilution":
        return ((float(np.mean(y)),),)
    raise ValueError(kind)


def fit_models(kinds, y, Mt, Xt, vols_uL, V0_uL, max_evals=60_000):
    """Fit each model kind. Returns list of (kind, params, sse) — every
    distinct local minimum is kept — or ResourceExhausted."""
    y = np.asarray(y, dtype=float)
    budget = Budget(max_evals)
    out = []
    try:
        for kind in kinds:
            basins = []
            for start in _starts_for(kind, y):
                p, sse = _lm(lambda pp: model_heats(kind, pp, Mt, Xt, vols_uL, V0_uL),
                             y, start, budget)
                key = np.round(np.asarray(p, dtype=float), 4)
                if not any(np.allclose(key, kept, atol=1e-3) for kept, _ in basins):
                    basins.append((key, (np.asarray(p, dtype=float), float(sse))))
            for _, (p, sse) in basins:
                out.append((kind, tuple(float(v) for v in p), sse))
    except BudgetExceeded:
        return ResourceExhausted(
            detail="model fitting exceeded evaluation budget",
            budget=f"{max_evals} evaluations",
        )
    out.sort(key=lambda c: (round(c[2], 9), c[0], tuple(round(v, 6) for v in c[1])))
    return out
