"""Binding model candidates: one-site, two-site and dilution-only.

All legal candidates are kept; when models tie within tolerance every one of
them is reported as equipotent instead of silently picking a winner.
"""
from __future__ import annotations

import numpy as np

from .domain import RESULT_BUDGET, RESULT_OK

PARAM_VERSION = "binding/1"

K_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]
DH_GRID = [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0]
TIE_REL_TOL = 1e-6


def bound_conc(K, Mt, Xt):
    """Bound ligand concentration (mM) for 1:1 binding, quadratic solution."""
    K = float(K)
    b = 1.0 + K * Mt + K * Xt
    disc = np.maximum(b * b - 4.0 * K * K * Mt * Xt, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * K)


def predict_heats(model, params, Mt, ligand_concs, V0_uL):
    """Predicted per-injection heats in uJ (kJ/mol * mM * uL == uJ)."""
    X = np.asarray(ligand_concs, dtype=float)
    if model == "one_site":
        K, dH = params
        B = bound_conc(K, Mt, X)
    elif model == "two_site":
        K1, K2, dH = params
        B = bound_conc(K1, Mt, X) + bound_conc(K2, Mt, X)
    elif model == "dilution_const":
        return np.full(len(X), float(params[0]))
    else:
        raise ValueError(model)
    dB = np.diff(np.concatenate([[0.0], B]))
    scale = dH if model == "one_site" else dH / 2.0
    return scale * dB * V0_uL


def fit_models(heats, Mt, ligand_concs, V0_uL, budget=100000):
    """Returns (status, payload); payload keeps every model, tied or not."""
    y = np.asarray(heats, dtype=float)
    X = np.asarray(ligand_concs, dtype=float)
    counter = [0]

    def ssr(pred):
        return float(np.sum((pred - y) ** 2))

    one = None
    for K in K_GRID:
        for dH in DH_GRID:
            counter[0] += 1
            if counter[0] > budget:
                return RESULT_BUDGET, None
            s = ssr(predict_heats("one_site", (K, dH), Mt, X, V0_uL))
            if one is None or s < one["ssr"] - 1e-12:
                one = {"model": "one_site", "params": {"K": K, "dH": dH}, "ssr": s}

    two = None
    for K1 in K_GRID:
        for K2 in K_GRID:
            for dH in DH_GRID:
                counter[0] += 1
                if counter[0] > budget:
                    return RESULT_BUDGET, None
                s = ssr(predict_heats("two_site", (K1, K2, dH), Mt, X, V0_uL))
                if two is None or s < two["ssr"] - 1e-12:
                    two = {"model": "two_site",
                           "params": {"K1": K1, "K2": K2, "dH": dH}, "ssr": s}

    counter[0] += 1
    if counter[0] > budget:
        return RESULT_BUDGET, None
    c = float(np.mean(y))
    dil = {"model": "dilution_const", "params": {"c": c},
           "ssr": ssr(np.full(len(y), c))}

    fits = [one, two, dil]
    best_ssr = min(f["ssr"] for f in fits)
    tol = TIE_REL_TOL * max(1.0, abs(best_ssr))
    candidates = sorted(f["model"] for f in fits if f["ssr"] <= best_ssr + tol)
    for f in fits:
        if f["model"] == "one_site":
            params = (f["params"]["K"], f["params"]["dH"])
        elif f["model"] == "two_site":
            params = (f["params"]["K1"], f["params"]["K2"], f["params"]["dH"])
        else:
            params = (f["params"]["c"],)
        pred = predict_heats(f["model"], params, Mt, X, V0_uL)
        f["predicted_uJ"] = [float(v) for v in pred]
        f["residuals_uJ"] = [float(v) for v in (pred - y)]
        f["is_candidate"] = f["model"] in candidates
    return RESULT_OK, {"models": fits, "candidates": candidates,
                       "unique_decision": len(candidates) == 1}
