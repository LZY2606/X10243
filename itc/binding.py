"""Binding-model candidates: one-site, two-site (independent) and dilution.

Deterministic NumPy-only fitting (grid + linear least squares for the
enthalpies, which enter linearly). Every fit consumes from an explicit
evaluation budget; exhausting it yields Status.BUDGET_EXHAUSTED, which is a
distinct outcome from FORMAT_CORRUPT and SEMANTIC_UNSOLVABLE.

Equally-good models (SSR within a relative tolerance of the best) are ALL
kept as candidates; choosing between them is a human decision recorded as an
event, never an automatic silent pick.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .diagnostics import Diagnostic, Outcome, Status

UMOL_SCALE = 1e-6  # uM * uL -> 1e-6 umol

EQUAL_SSR_TOL = 0.05  # models within 5% of best SSR are "equally good"


class Budget:
    def __init__(self, max_evaluations: int):
        self.remaining = max_evaluations

    def take(self, n: int = 1) -> bool:
        if self.remaining < n:
            return False
        self.remaining -= n
        return True


def cell_concentrations(volumes_uL: np.ndarray, cell_volume_uL: float,
                        macromolecule_uM: float, syringe_uM: float):
    """Cumulative in-cell totals under the constant-volume (overflow) model."""
    n = len(volumes_uL)
    mt = np.zeros(n)
    xt = np.zeros(n)
    m = macromolecule_uM
    x = 0.0
    for i in range(n):
        v = volumes_uL[i]
        x = (x * (cell_volume_uL - v) + syringe_uM * v) / cell_volume_uL
        m = m * (1.0 - v / cell_volume_uL)
        mt[i] = m
        xt[i] = x
    return mt, xt


def bound_sites(xt: np.ndarray, mt: np.ndarray, n_sites: float, K: float) -> np.ndarray:
    """Concentration (uM) of bound complex for identical independent sites."""
    sites = n_sites * mt
    b = sites + xt + 1.0 / K
    q = 0.5 * (b - np.sqrt(np.maximum(b * b - 4.0 * sites * xt, 0.0)))
    return q


def site_shape(volumes_uL, cell_volume_uL, mt, xt, n_sites, K) -> np.ndarray:
    """Per-injection heat shape S_i (umol of complex formed), dH factored out."""
    q = bound_sites(xt, mt, n_sites, K)
    s = np.zeros(len(volumes_uL))
    prev = 0.0
    for i in range(len(volumes_uL)):
        displaced = prev * (1.0 - volumes_uL[i] / cell_volume_uL)
        s[i] = cell_volume_uL * UMOL_SCALE * (q[i] - displaced)
        prev = q[i]
    return s


def _solve_linear(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def fit_one_site(y, volumes, v0, mt, xt, budget: Budget,
                 n_grid=None, logk_grid=None) -> Optional[dict]:
    n_grid = n_grid if n_grid is not None else np.round(np.arange(0.7, 1.31, 0.1), 3)
    logk_grid = logk_grid if logk_grid is not None else np.round(np.arange(2.0, 8.01, 0.1), 3)
    total = len(n_grid) * len(logk_grid)
    if not budget.take(total):
        return None
    best = None
    for n_sites in n_grid:
        for logk in logk_grid:
            s = site_shape(volumes, v0, mt, xt, float(n_sites), 10.0 ** float(logk))
            denom = float(s @ s)
            if denom <= 0:
                continue
            dh = float(s @ y) / denom
            ssr = float(np.sum((y - dh * s) ** 2))
            if best is None or ssr < best["ssr"]:
                best = {"ssr": ssr, "n": float(n_sites), "logK": float(logk), "dH": dh}
    # local refinement pass (fixed extra budget, still accounted)
    refine = 41
    if best is None or not budget.take(refine):
        return best
    for logk in np.round(np.arange(best["logK"] - 0.1, best["logK"] + 0.101, 0.005), 4):
        s = site_shape(volumes, v0, mt, xt, best["n"], 10.0 ** float(logk))
        denom = float(s @ s)
        if denom <= 0:
            continue
        dh = float(s @ y) / denom
        ssr = float(np.sum((y - dh * s) ** 2))
        if ssr < best["ssr"]:
            best = {"ssr": ssr, "n": best["n"], "logK": float(logk), "dH": dh}
    return best


def fit_two_site(y, volumes, v0, mt, xt, budget: Budget) -> Optional[dict]:
    logk1_grid = np.round(np.arange(2.0, 5.01, 0.5), 3)
    logk2_grid = np.round(np.arange(5.5, 8.01, 0.5), 3)
    total = len(logk1_grid) * len(logk2_grid)
    if not budget.take(total):
        return None
    best = None
    for logk1 in logk1_grid:
        s1 = site_shape(volumes, v0, mt, xt, 1.0, 10.0 ** float(logk1))
        for logk2 in logk2_grid:
            s2 = site_shape(volumes, v0, mt, xt, 1.0, 10.0 ** float(logk2))
            design = np.column_stack([s1, s2])
            coef = _solve_linear(design, y)
            ssr = float(np.sum((y - design @ coef) ** 2))
            if best is None or ssr < best["ssr"]:
                best = {"ssr": ssr, "logK1": float(logk1), "logK2": float(logk2),
                        "dH1": float(coef[0]), "dH2": float(coef[1])}
    return best


def fit_dilution(y, volumes, budget: Budget) -> Optional[dict]:
    if not budget.take(1):
        return None
    design = np.column_stack([np.ones_like(volumes), volumes])
    coef = _solve_linear(design, y)
    ssr = float(np.sum((y - design @ coef) ** 2))
    return {"ssr": ssr, "offset_uJ": float(coef[0]), "slope_uJ_per_uL": float(coef[1])}


def predict_model(model: dict, volumes, v0, mt, xt) -> np.ndarray:
    kind = model["model"]
    if kind == "one_site":
        return model["dH"] * site_shape(volumes, v0, mt, xt, model["n"], 10.0 ** model["logK"])
    if kind == "two_site":
        s1 = site_shape(volumes, v0, mt, xt, 1.0, 10.0 ** model["logK1"])
        s2 = site_shape(volumes, v0, mt, xt, 1.0, 10.0 ** model["logK2"])
        return model["dH1"] * s1 + model["dH2"] * s2
    if kind == "dilution":
        return model["offset_uJ"] + model["slope_uJ_per_uL"] * np.asarray(volumes, dtype=float)
    raise ValueError(f"unknown model kind {kind!r}")


def fit_models(
    heats_uJ: List[float],
    volumes_uL: List[float],
    cell_volume_uL: float,
    macromolecule_uM: float,
    syringe_uM: float,
    max_evaluations: int = 2000,
    equal_tol: float = EQUAL_SSR_TOL,
) -> Outcome:
    """Fit all candidate models; keep every equally-good candidate."""
    y = np.asarray(heats_uJ, dtype=float)
    volumes = np.asarray(volumes_uL, dtype=float)
    if y.size < 3:
        return Outcome(Status.SEMANTIC_UNSOLVABLE, diagnostics=[Diagnostic(
            code="too_few_injections",
            message=f"need at least 3 usable injection integrals to compare models, got {y.size}",
            path="injections",
        )])
    if not np.isfinite(y).all() or float(np.sum(np.abs(y))) == 0.0:
        return Outcome(Status.SEMANTIC_UNSOLVABLE, diagnostics=[Diagnostic(
            code="heats_degenerate",
            message="injection heats are all zero or non-finite; no binding signal to fit",
            path="integrals",
        )])
    mt, xt = cell_concentrations(volumes, cell_volume_uL, macromolecule_uM, syringe_uM)
    budget = Budget(max_evaluations)
    candidates: List[dict] = []

    one = fit_one_site(y, volumes, cell_volume_uL, mt, xt, budget)
    if one is not None:
        one["model"] = "one_site"
        candidates.append(one)
    two = fit_two_site(y, volumes, cell_volume_uL, mt, xt, budget)
    if two is not None:
        two["model"] = "two_site"
        candidates.append(two)
    dil = fit_dilution(y, volumes, budget)
    if dil is not None:
        dil["model"] = "dilution"
        candidates.append(dil)

    if not candidates:
        return Outcome(Status.BUDGET_EXHAUSTED, diagnostics=[Diagnostic(
            code="budget_exhausted",
            message=f"evaluation budget ({max_evaluations}) ran out before any model converged",
            path="models",
        )])

    # Stable ordering: SSR first, then model name, so equivalent inputs always
    # serialize in the same order.
    candidates.sort(key=lambda c: (round(c["ssr"], 9), c["model"]))
    best_ssr = candidates[0]["ssr"]
    equally_good = [c for c in candidates if c["ssr"] <= best_ssr * (1.0 + equal_tol)]
    for c in candidates:
        c["equally_good"] = c in equally_good
        c["predicted_uJ"] = [round(v, 6) for v in predict_model(c, volumes, cell_volume_uL, mt, xt)]
        c["residuals_uJ"] = [round(float(a - b), 6) for a, b in zip(y, c["predicted_uJ"])]
    payload = {
        "candidates": candidates,
        "equally_good_count": len(equally_good),
        "budget": {"max_evaluations": max_evaluations, "remaining": budget.remaining},
        "equal_ssr_tol": equal_tol,
    }
    diags = []
    status = Status.OK
    if budget.remaining <= 0:
        diags.append(Diagnostic(
            code="budget_exhausted",
            message="budget fully consumed; some candidate grids may be truncated",
            path="models", severity="warning"))
        status = Status.DIAGNOSTIC
    if len(equally_good) > 1:
        diags.append(Diagnostic(
            code="multiple_equally_good_models",
            message=f"{len(equally_good)} models fit equally well within tol={equal_tol}; "
                    "all candidates retained for human decision",
            path="models", severity="warning"))
        status = Status.DIAGNOSTIC
    return Outcome(status, payload=payload, diagnostics=diags)
