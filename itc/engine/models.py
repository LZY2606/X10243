"""Binding model candidates fit against per-injection binding heats.

Candidates:

* ``one_site``  - Wiseman 1:1 model (N, K [1/M], dH [cal/mol])
* ``two_site``  - independent two-class-sites model (N1,K1,dH1,N2,K2,dH2)
* ``dilution``  - constant per-injection dilution heat (no binding)

Equally good candidates are *all kept*.  Tie test uses the AIC delta rule.
Fitting uses a coordinate-descent Nelder-Mead bounded least squares with an
explicit iteration budget; exhausting it is ResourceExhausted, distinct from
non-convergence warnings.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..errors import ResourceExhausted
from .hashing import sha12

R_GAS = 1.987204258  # cal/(mol*K)


def _q_one_site(X, N, K, dH):
    """Cumulative heat µcal. X = array([Mt, Lt])"""
    Mt, Lt = X
    a = 1.0
    b = -(Lt + N + 1.0 / K)
    c = N * Lt
    disc = b * b - 4.0 * a * c
    disc = np.maximum(disc, 0.0)
    bound = 0.5 * (-b - np.sqrt(disc))
    return Mt * dH * bound


def delta_heats(model: str, theta: np.ndarray, Mt: float,
                Lt: np.ndarray, V0_ml: float) -> np.ndarray:
    """Model incremental heat per injection in µcal."""
    if model == "one_site":
        N, K, dH = theta
        Q = _q_one_site((Mt, Lt), N, K, dH)
        dQ = np.empty_like(Q)
        dQ[0] = Q[0]
        dQ[1:] = Q[1:] - Q[:-1] + (Lt[1:] - Lt[:-1]) / (2.0 * V0_ml) * (
            Q[:-1] / np.where(Lt[:-1] > 0, Lt[:-1], np.inf)) * V0_ml
        # standard displacement correction:
        corr = np.empty_like(Q)
        corr[0] = Q[0]
        corr[1:] = Q[1:] - Q[:-1] + (Lt[1:] - Lt[:-1]) / 2.0 * (
            Q[:-1] / np.where(Lt[:-1] > 0, Lt[:-1], np.inf))
        return corr
    if model == "two_site":
        N1, K1, dH1, N2, K2, dH2 = theta
        Q = (_q_one_site((Mt, Lt), N1, K1, dH1)
             + _q_one_site((Mt, Lt), N2, K2, dH2))
        corr = np.empty_like(Q)
        corr[0] = Q[0]
        corr[1:] = Q[1:] - Q[:-1] + (Lt[1:] - Lt[:-1]) / 2.0 * (
            Q[:-1] / np.where(Lt[:-1] > 0, Lt[:-1], np.inf))
        return corr
    if model == "dilution":
        q0 = theta[0]
        return np.full_like(Lt, q0)
    raise ValueError(model)


MODEL_SPECS: dict[str, dict] = {
    "one_site": {
        "params": ["N", "K_M-1", "dH_cal/mol"],
        "bounds": [(1e-6, 50.0), (1.0, 1.0e12), (-1.0e6, 1.0e6)],
        "starts": [
            (1.0, 1.0e5, -5000.0),
            (1.0, 1.0e7, -5000.0),
            (2.0, 1.0e6, -20000.0),
        ],
    },
    "two_site": {
        "params": ["N1", "K1_M-1", "dH1_cal/mol",
                   "N2", "K2_M-1", "dH2_cal/mol"],
        "bounds": [(1e-6, 50.0), (1.0, 1.0e12), (-1.0e6, 1.0e6),
                   (1e-6, 50.0), (1.0, 1.0e12), (-1.0e6, 1.0e6)],
        "starts": [
            (0.5, 1.0e5, -3000.0, 1.5, 1.0e8, -15000.0),
            (1.5, 1.0e8, -15000.0, 0.5, 1.0e5, -3000.0),
            (1.0, 1.0e6, -8000.0, 1.0, 1.0e6, -8000.0),
        ],
    },
    "dilution": {
        "params": ["q_dilution_ucal"],
        "bounds": [(-1.0e6, 1.0e6)],
        "starts": [(0.0,), (-500.0,), (500.0,)],
    },
}


def _clamp(theta, bounds):
    return np.array([min(max(x, lo), hi) for x, (lo, hi) in zip(theta, bounds)])


def _nelder_mead(fn, x0, bounds, max_iter: int, x_tol: float = 1e-10):
    """Log-space bounded NM over positive parameters; signed dH params use
    affine scaling. Returns (x, f, iterations, converged). Exhausting the
    budget raises ResourceExhausted."""
    x0 = np.asarray(_clamp(x0, bounds), dtype=float)
    n = len(x0)

    def to_z(x):
        z = []
        for v, (lo, hi) in zip(x, bounds):
            if lo < 0 < hi:  # signed parameter
                z.append(v / max(abs(hi), abs(-lo)))
            else:
                span = math.log(hi) - math.log(lo)
                z.append((math.log(max(v, lo)) - math.log(lo)) / span)
        return np.array(z)

    def to_x(z):
        x = []
        for zz, (lo, hi) in zip(z, bounds):
            if lo < 0 < hi:
                x.append(min(max(zz * max(abs(hi), abs(-lo)), lo), hi))
            else:
                span = math.log(hi) - math.log(lo)
                x.append(min(math.exp(math.log(lo) + min(max(zz, 0.0), 1.0) * span), hi))
        return np.array(x)

    z0 = to_z(x0)
    simplex = [z0]
    for j in range(n):
        z = z0.copy()
        z[j] += 0.25
        simplex.append(np.clip(z, 0.0, 1.0))
    fs = [fn(to_x(z)) for z in simplex]
    iters = 0
    while iters < max_iter:
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]
        spread = np.max(np.ptp(np.array(simplex), axis=0))
        if spread < x_tol:
            return to_x(simplex[0]), fs[0], iters, True
        centroid = np.mean(np.array(simplex[:-1]), axis=0)
        xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), 0.0, 1.0)
        fr = fn(to_x(xr))
        iters += 1
        if fs[0] <= fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
            continue
        if fr < fs[0]:
            xe = np.clip(centroid + 2.0 * (xr - centroid), 0.0, 1.0)
            fe = fn(to_x(xe))
            iters += 1
            if fe < fr:
                simplex[-1], fs[-1] = xe, fe
            else:
                simplex[-1], fs[-1] = xr, fr
            continue
        xc = np.clip(centroid - 0.5 * (simplex[-1] - centroid), 0.0, 1.0)
        fc = fn(to_x(xc))
        iters += 1
        if fc < fs[-1]:
            simplex[-1], fs[-1] = xc, fc
            continue
        best = simplex[0]
        simplex = [best] + [np.clip(best + 0.5 * (s - best), 0.0, 1.0)
                            for s in simplex[1:]]
        fs = [fn(to_x(z)) for z in simplex]
    raise ResourceExhausted(
        f"fit iteration budget {max_iter} exhausted",
    )


def _fit_one(model: str, Mt, Lt, y, V0_ml, budget: int,
             tie_delta: float) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    bounds = spec["bounds"]

    def sse(theta):
        try:
            pred = delta_heats(model, np.asarray(theta), Mt, Lt, V0_ml)
        except (ValueError, FloatingPointError):
            return 1.0e30
        r = pred - y
        val = float(np.dot(r, r))
        return val if math.isfinite(val) else 1.0e30

    best = None
    for start in spec["starts"]:
        x, f, iters, converged = _nelder_mead(
            sse, list(start), bounds, max_iter=budget)
        if best is None or f < best[1]:
            best = (x, f, iters, converged)
    theta, sse_val, iters, converged = best
    pred = delta_heats(model, theta, Mt, Lt, V0_ml)
    resid = y - pred
    n = len(y)
    k = len(theta) + 1
    var = max(sse_val / n, 1e-12)
    aic = n * math.log(max(sse_val / n, 1e-30)) + 2 * k
    rmse = math.sqrt(sse_val / n)
    return {
        "model": model,
        "parameter_names": spec["params"],
        "parameters": {name: float(v) for name, v in zip(spec["params"], theta)},
        "predicted_ucal": [float(v) for v in pred],
        "residuals_ucal": [float(v) for v in resid],
        "sse_ucal2": float(sse_val),
        "rmse_ucal": rmse,
        "aic": aic,
        "iterations": iters,
        "converged": converged,
    }


def fit_candidates(*, corrected: list[dict[str, Any]], injections: list[dict],
                   concentrations: dict[str, Any], conditions: dict[str, Any],
                   rules: dict[str, Any], budget: int | None = None) -> dict:
    """Fit all legal candidates from dilution-corrected heats.
    Injections flagged first-injection-discarded or dilution-unresolved are
    excluded; when too few points remain, a located diagnostic is returned."""
    budget = budget or rules["fit"]["max_iterations"]
    tie_delta = rules["fit"]["tie_aic_delta"]

    use = [c for c in corrected if not c.get("discarded")
           and c.get("net_heat_ucal") is not None]
    diagnostics = []
    if len(use) < 3:
        return {"candidates": [], "status": "semantic_unresolvable",
                "message": "fewer than 3 binding heats after discard/blanks",
                "used_ordinals": [c["injection_ordinal"] for c in use],
                "diagnostics": [{"code": "too_few_heats",
                                 "message": "need >=3 corrected injections to fit"}],
                "rules_version": rules.get("version")}

    ligand = _scalar(concentrations, "ligand")
    macro = _scalar(concentrations, "macromolecule")
    V0 = _scalar(conditions, "cell_volume")
    if ligand is None or macro is None or V0 is None:
        return {"candidates": [], "status": "semantic_unresolvable",
                "message": "ligand/M concentration or cell volume unresolved",
                "diagnostics": [{"code": "concentration_unresolved",
                                 "message": "cannot normalize concentrations"}],
                "rules_version": rules.get("version")}

    ordinals = [c["injection_ordinal"] for c in use]
    inj_by_ord = {r["ordinal"]: r for r in injections}
    vols = np.array([inj_by_ord[o]["volume"]["canonical"]["value"] for o in ordinals],
                    dtype=float)  # µL
    V0_ul = float(V0)
    V0_ml = V0_ul / 1000.0
    y = np.array([c["net_heat_ucal"] for c in use], dtype=float)

    Lt = np.zeros(len(use))
    cum = 0.0
    for i, v in enumerate(vols):
        cum += v
        Lt[i] = cum * ligand / V0_ul      # mM
    Mt = float(macro)                      # mM
    if len(use) < 7:
        models_to_fit = ["one_site", "dilution"]
    else:
        models_to_fit = ["one_site", "two_site", "dilution"]

    fits = [_fit_one(m, Mt, Lt, y, V0_ml, budget, tie_delta)
            for m in models_to_fit]
    best_aic = min(f["aic"] for f in fits)
    for f in fits:
        f["delta_aic"] = f["aic"] - best_aic
        f["tied_best"] = abs(f["delta_aic"]) <= tie_delta
        f["id"] = "fit_" + sha12({
            "model": f["model"], "params": {k: round(v, 8)
                                            for k, v in f["parameters"].items()},
            "sse": round(f["sse_ucal2"], 6),
            "ordinals": ordinals,
            "rules_version": rules.get("version"),
        })

    ranked = sorted(fits, key=lambda f: (round(f["aic"], 6),
                                         _model_order(f["model"])))
    tied = [f["model"] for f in ranked if f["tied_best"]]
    unique_best = tied[0] if len(tied) == 1 else None
    return {
        "candidates": ranked,
        "status": "ok",
        "used_ordinals": ordinals,
        "x_ratio": [float(v / Mt) for v in Lt],
        "Mt_mM": Mt,
        "Lt_mM": [float(v) for v in Lt],
        "observed_ucal": [float(v) for v in y],
        "tie": {"is_tied": len(tied) > 1, "models": tied,
                "aic_delta_threshold": tie_delta},
        "unique_best_model": unique_best,
        "rules_version": rules.get("version"),
    }


def _model_order(m: str) -> int:
    return {"one_site": 0, "two_site": 1, "dilution": 2}[m]


def _scalar(mat: dict[str, Any], role: str):
    for s in mat["payload"].get("scalars", []):
        if s["role"] == role:
            c = s["measured"]["canonical"]
            return None if c is None else c["value"]
    return None
