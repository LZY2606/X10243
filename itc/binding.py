"""Binding-model candidates: one-site, multi-site and dilution-only.

Fitting is a bounded computation: a finite evaluation budget produces either a
converged fit, an explicitly ``budget_exhausted`` outcome (with the partial
best), or ``insufficient_data`` for semantic impossibility.
"""
from __future__ import annotations

import numpy as np

from .errors import BudgetExhausted, Diagnostic, SemanticImpossibility
from .models import make_node

DEFAULT_SCORING = {
    "version": 1,
    "tie_tolerance_norm_sse": 1e-6,
    "fit_budget_evals": 4000,
    "starts": {"one_site": 3, "two_site": 4, "dilution": 1},
}

MODEL_ORDER = ["dilution", "one_site", "two_site"]


def titration_axis(corrections, schedule, concentration, runmeta):
    """Build x (ligand injected in mol per injection) and totals."""
    vt = runmeta["cell_volume"]["value_l"]
    cs = concentration["syringe"]["value_m"]
    if vt is None or cs is None:
        raise SemanticImpossibility(
            "cannot build titration axis: cell volume or syringe "
            "concentration has no normalized value")
    rows = []
    for node in corrections:
        ordinal = node.payload["ordinal"]
        if node.payload["status"] != "corrected":
            continue
        if node.payload.get("discarded"):
            continue
        inj = schedule["injections"][ordinal]
        if inj["volume_l"] is None:
            continue
        rows.append({
            "ordinal": ordinal,
            "x_mol": inj["volume_l"] * cs,
            "q_cal": node.payload["corrected_heat_cal"],
        })
    return rows, vt, cs


def cumulative_fraction(rows):
    x = np.array([r["x_mol"] for r in rows], dtype=float)
    q = np.array([r["q_cal"] for r in rows], dtype=float)
    return x, q


def _free_fraction_total(x_cum, macromol, k):
    """Bound fraction in a 1:1 model with total ligand/macromolecule.

    Solves K a^2 - (1 + K(Mt+Lt)) a + 1 = 0 for a = free ligand fraction.
    """
    km = k * macromol
    kl = k * x_cum
    b = 1.0 + km + kl
    disc = b * b - 4.0 * kl
    disc = np.maximum(disc, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * np.maximum(kl, 1e-30))


def predict_one_site(x, q_scale, log_k, macromol, vt):
    """q_i model with K in M^-1. q_scale is molar enthalpy * M0 * V0.

    Returns per-injection incremental heat.
    """
    k = np.exp(log_k)
    x_cum = np.cumsum(x)
    bound_frac = 1.0 - _free_fraction_total(x_cum, macromol, k)
    q_total = q_scale * bound_frac
    dq = np.diff(np.concatenate(([0.0], q_total)))
    return dq


def predict_two_site(x, q1, q2, log_k1, log_k2, macromol, vt):
    """Two independent sites: sum of two one-site saturation curves."""
    x_cum = np.cumsum(x)
    f1 = 1.0 - _free_fraction_total(x_cum, macromol, np.exp(log_k1))
    f2 = 1.0 - _free_fraction_total(x_cum, macromol, np.exp(log_k2))
    q_total = q1 * f1 + q2 * f2
    return np.diff(np.concatenate(([0.0], q_total)))


def predict_dilution(x, slope_cal_per_mol):
    return x * slope_cal_per_mol


class _Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def tick(self):
        self.used += 1
        if self.used > self.limit:
            raise BudgetExhausted(
                "model fit evaluation budget exhausted", self.limit,
                self.used)


def _nelder_mead(fn, x0, budget, max_iter=None):
    """Small dependency-free Nelder-Mead; raises BudgetExhausted."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    step = 0.05
    for i in range(n):
        v = np.array(x0, dtype=float)
        v[i] += step * max(1.0, abs(v[i]))
        simplex.append(v)
    vals = [fn(v) for v in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    iteration = 0
    max_iter = max_iter or budget.limit
    while iteration < max_iter:
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        spread = np.max(np.abs(simplex[-1] - simplex[0]))
        if spread < 1e-10 and abs(vals[-1] - vals[0]) < 1e-14:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = fn(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        elif fr < vals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = fn(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = fn(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = fn(simplex[i])
        iteration += 1
    return np.array(simplex[0]), vals[0], iteration


# second half appended below
