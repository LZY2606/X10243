"""Binding-model candidates fitted entirely with NumPy.

Three hypotheses are compared on the same fitted rows:

* ``dilution``  : q_i = b0 + b1 * L_i      (heat proportional to the amount
                  of titrant injected; non-binding null hypothesis)
* ``one_site``  : Wiseman single-site model, parameters (n, log10 Ka, dH)
* ``two_site``  : two independent classes of sites, parameters
                  (n1, log10 Ka1, dH1, n2, log10 Ka2, dH2), canonicalised
                  Ka1 <= Ka2 so symmetric answers receive one stable id

Equally-good models are *all* kept as legal candidates.  The criterion is
noise-aware: the smallest SSE defines the best SSE ``s*`` and any model with
``sse <= s* + tol * nu * sigma2`` is part of the same tie group, where sigma2
is estimated from the best fit and ``tol`` defaults to 1e-3.  Ties are
presented in the fixed order dilution -> one_site -> two_site so equivalent
input always yields a stable, identical ordering.

Concentration never participates in integral/result ids; a concentration
revision creates a new fit while old integrals remain valid (that layering
lives in :mod:`itc.store`, the fitting input passed here is just numbers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ResourceExhausted

MODEL_RULE_VERSION = "binding-models-1"
TIE_TOLERANCE = 1e-3
MODEL_ORDER = {"dilution": 0, "one_site": 1, "two_site": 2}


@dataclass
class FitBudgets:
    max_evaluations: int = 2500
    max_points: int = 1000


@dataclass
class FitRow:
    """One fitted injection heat.

    q_J          integrated heat for this shot (baseline or blank corrected)
    v_inj_L      injection volume
    L_syringe_M  titrant concentration *used for this fit*
    M_cell_M     macromolecule concentration in the cell *for this fit*
    V0_L         cell (pool) volume
    injection    original injection index (for traceability)
    integral_id  stable id of the integral this row summarizes
    """

    q_J: float
    v_inj_L: float
    L_syringe_M: float
    M_cell_M: float
    V0_L: float
    injection: int
    integral_id: str


def _theta(total_M: float, ligand_M: float, ka: float, n: float) -> float:
    """Fraction of macromolecule with bound ligand at one site class."""
    a = n * total_M
    b = a + ligand_M + 1.0 / ka
    disc = b * b - 4.0 * a * ligand_M
    if disc < 0:
        disc = 0.0
    return (b - math.sqrt(disc)) / (2.0 * a) if a > 0 else 0.0


def _bound_curve(rows: list[FitRow], ka: float, n: float) -> np.ndarray:
    """Cumulative moles bound at site class after each injection."""
    V = rows[0].V0_L
    M0 = rows[0].M_cell_M
    bound = np.zeros(len(rows))
    moles_lig = 0.0
    for k, row in enumerate(rows):
        V += row.v_inj_L
        moles_lig += row.v_inj_L * row.L_syringe_M
        total_M = M0 * rows[0].V0_L / V
        ligand_M = moles_lig / V
        bound[k] = V * total_M * _theta(total_M, ligand_M, ka, n)
    return bound


def _lstsq_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(coef[0]), float(coef[1]), float(resid @ resid)


def _nnls2(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Non-negative least squares for y = dH1*x1 + dH2*x2 (no intercept)."""
    best = (0.0, 0.0, float(y @ y))
    X = np.column_stack([x1, x2])

    def sse(a: float, b: float) -> float:
        r = y - X @ np.array([a, b])
        return float(r @ r)

    candidates = [(0.0, 0.0)]
    a_alone = max(0.0, float(np.dot(x1, y) / np.dot(x1, x1))) if np.dot(x1, x1) else 0.0
    b_alone = max(0.0, float(np.dot(x2, y) / np.dot(x2, y))) if np.dot(x2, x2) else 0.0
    candidates.append((a_alone, 0.0))
    candidates.append((0.0, b_alone))
    for a0, b0 in candidates:
        a, b = a0, b0
        for _ in range(30):
            a_num = np.dot(x1, y - b * x2)
            a_den = np.dot(x1, x1)
            a = max(0.0, a_num / a_den) if a_den else 0.0
            b_num = np.dot(x2, y - a * x1)
            b_den = np.dot(x2, x2)
            b = max(0.0, b_num / b_den) if b_den else 0.0
        candidates.append((a, b))
    for a, b in candidates:
        score = sse(a, b)
        if score < best[2]:
            best = (a, b, score)
    return best


def fit(rows: list[FitRow], budgets: FitBudgets | None = None) -> dict[str, Any]:
    """Fit all three hypotheses; return ranked candidates + fit evidence."""
    budgets = budgets or FitBudgets()
    if len(rows) > budgets.max_points:
        raise ResourceExhausted(
            f"{len(rows)} fit rows exceed budget {budgets.max_points}", [])
    if len(rows) < 3:
        from .errors import SemanticImpasse, Diagnostic
        raise SemanticImpasse(
            "at least 3 integrable, matched injections are required to fit",
            [Diagnostic("M010",
                        f"only {len(rows)} usable rows; fitting is not defined")])
    evals = 0

    q = np.array([r.q_J for r in rows], dtype=float)
    L = np.array([r.v_inj_L * r.L_syringe_M for r in rows], dtype=float)

    # --- dilution: q = b0 + b1 * moles titrant ---------------------------
    b1, b0, sse_dil = _lstsq_intercept(L, q)
    evals += 1
    dilution = {
        "model": "dilution", "rule_version": MODEL_RULE_VERSION,
        "parameters": {"dilution_heat_J_per_mol": b1, "offset_J": b0},
        "sse": sse_dil, "k": 2,
    }

    # --- one site ----------------------------------------------------------
    one_best: dict[str, Any] | None = None
    log_ka_grid = np.linspace(2.0, 9.0, 29)
    n_grid = np.linspace(0.2, 5.0, 13)
    for log_ka in log_ka_grid:
        ka = 10 ** log_ka
        for n in n_grid:
            bound = _bound_curve(rows, ka, n)
            delta = np.diff(np.concatenate([[0.0], bound]))
            dH, intercept, sse = _lstsq_intercept(delta, q)
            evals += 1
            cand = {"sse": sse, "log10_Ka": float(log_ka), "n": float(n),
                    "dH_J_per_mol": dH, "intercept_J": intercept}
            if one_best is None or sse < one_best["sse"]:
                one_best = cand
    # local refinement on the coarse winner
    c = one_best
    for d_lk in (-0.25, -0.1, 0.1, 0.25):
        for d_n in (-0.15, 0.15):
            lk = min(9.5, max(1.5, c["log10_Ka"] + d_lk))
            n = max(0.05, c["n"] + d_n)
            bound = _bound_curve(rows, 10 ** lk, n)
            delta = np.diff(np.concatenate([[0.0], bound]))
            dH, intercept, sse = _lstsq_intercept(delta, q)
            evals += 1
            if sse < one_best["sse"]:
                one_best = {"sse": sse, "log10_Ka": lk, "n": n,
                            "dH_J_per_mol": dH, "intercept_J": intercept}
    one_site = {
        "model": "one_site", "rule_version": MODEL_RULE_VERSION,
        "parameters": {"n": one_best["n"], "log10_Ka": one_best["log10_Ka"],
                       "Ka_M_inv": 10 ** one_best["log10_Ka"],
                       "dH_J_per_mol": one_best["dH_J_per_mol"],
                       "intercept_J": one_best["intercept_J"]},
        "sse": one_best["sse"], "k": 3,
    }

    # --- two sites ----------------------------------------------------------
    two_best = None
    two_grid = np.linspace(2.5, 8.5, 13)
    n_grid2 = np.linspace(0.2, 3.0, 7)
    for log1 in two_grid:
        for log2 in two_grid:
            if log2 < log1:
                continue
            for n1 in n_grid2:
                ka1, ka2 = 10 ** log1, 10 ** log2
                b1v = _bound_curve(rows, ka1, n1)
                x1 = np.diff(np.concatenate([[0.0], b1v]))
                # second site shares n2=n1 in the coarse screen with dH2=0
                dH1, dH2, sse = _nnls2(x1, np.zeros_like(x1), q)
                evals += 1
                if two_best is None or sse < two_best[0]:
                    two_best = (sse, log1, log2, n1, n1, dH1, dH2)
    # refine winner with independent n2 and real second-curve basis
    _, l1, l2, wn1, _, _, _ = two_best
    for d1 in (-0.3, 0.0, 0.3):
        for d2 in (-0.3, 0.0, 0.3):
            for n2 in (0.5, 1.0, 2.0):
                log1, log2 = l1 + d1, max(l1 + d1 + 0.1, l2 + d2)
                x1 = np.diff(np.concatenate([[0.0],
                            _bound_curve(rows, 10 ** log1, wn1)]))
                x2 = np.diff(np.concatenate([[0.0],
                            _bound_curve(rows, 10 ** log2, n2)]))
                dH1, dH2, sse = _nnls2(x2 and x1, x2, q)
                evals += 1
                if sse < two_best[0]:
                    two_best = (sse, log1, log2, wn1, n2, dH1, dH2)
    if evals > budgets.max_evaluations:
        raise ResourceExhausted(
            f"fit used {evals} evaluations over budget {budgets.max_evaluations}",
            [])
    sse2, l1, l2, n1, n2, dH1, dH2 = two_best
    two_site = {
        "model": "two_site", "rule_version": MODEL_RULE_VERSION,
        "parameters": {"n1": n1, "log10_Ka1": l1, "Ka1_M_inv": 10 ** l1,
                       "dH1_J_per_mol": dH1, "n2": n2, "log10_Ka2": l2,
                       "Ka2_M_inv": 10 ** l2, "dH2_J_per_mol": dH2},
        "sse": sse2, "k": 6,
    }

    candidates = [dilution, one_site, two_site]
    n_rows = len(rows)
    best_sse = min(c["sse"] for c in candidates)
    sigma2 = max(best_sse / max(1, n_rows - 1), 1e-30)
    nu = n_rows - 1
    for c in candidates:
        c["aic"] = n_rows * math.log(max(c["sse"], 1e-30) / n_rows) + 2 * c["k"]
        c["is_best"] = c["sse"] == best_sse
        c["is_tied"] = c["sse"] <= best_sse + TIE_TOLERANCE * nu * sigma2
    tied = [c for c in candidates if c["is_tied"]]
    for c in candidates:
        c["tie_group_size"] = len(tied)
    candidates.sort(key=lambda c: (c["sse"], MODEL_ORDER[c["model"]]))
    for rank, c in enumerate(candidates, start=1):
        c["rank"] = rank
    residuals = _residuals(rows, candidates[0])
    return {
        "rule_version": MODEL_RULE_VERSION,
        "n_rows": n_rows,
        "evaluations": evals,
        "sigma2_estimate": sigma2,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_group": [c["model"] for c in tied],
        "candidates": candidates,
        "residuals": residuals,
        "rows": [{"injection": r.injection, "integral_id": r.integral_id,
                  "q_J": r.q_J, "moles_ligand": r.v_inj_L * r.L_syringe_M}
                 for r in rows],
    }


def predict(parameters: dict[str, Any], model: str,
            rows: list[FitRow]) -> np.ndarray:
    if model == "dilution":
        L = np.array([r.v_inj_L * r.L_syringe_M for r in rows])
        return parameters["dilution_heat_J_per_mol"] * L + parameters["offset_J"]
    if model == "one_site":
        bound = _bound_curve(rows, 10 ** parameters["log10_Ka"], parameters["n"])
        delta = np.diff(np.concatenate([[0.0], bound]))
        return parameters["dH_J_per_mol"] * delta + parameters["intercept_J"]
    if model == "two_site":
        b1 = _bound_curve(rows, 10 ** parameters["log10_Ka1"], parameters["n1"])
        b2 = _bound_curve(rows, 10 ** parameters["log10_Ka2"], parameters["n2"])
        x1 = np.diff(np.concatenate([[0.0], b1]))
        x2 = np.diff(np.concatenate([[0.0], b2]))
        return parameters["dH1_J_per_mol"] * x1 + parameters["dH2_J_per_mol"] * x2
    raise ValueError(f"unknown model {model}")


def _residuals(rows: list[FitRow], best: dict[str, Any]) -> list[dict[str, Any]]:
    yhat = predict(best["parameters"], best["model"], rows)
    return [{"injection": r.injection, "integral_id": r.integral_id,
             "q_J": r.q_J, "predicted_J": float(yhat[k]),
             "residual_J": float(r.q_J - yhat[k])}
            for k, r in enumerate(rows)]
