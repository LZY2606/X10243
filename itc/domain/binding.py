"""Binding-model candidates and fitting.

Three legal hypotheses are compared against the *net* per-injection heats:

* ``one_site``  - single-class independent sites (Wiseman)
* ``two_site``  - two independent site classes with separate K, dH, N
* ``dilution``  - no saturable binding; q linear in injected ligand amount

All candidates are returned. Equi-optimal candidates (relative SSE within a
declared tolerance) are all kept; the caller must not silently pick one.
A concentration revision invalidates (stales) fits upstream; it never edits
stored integrals.

Resource-budget exhaustion is its own terminal status, distinct from
corruption or semantic unresolved cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .diagnostics import BUDGET_EXHAUSTED, SEMANTIC_UNRESOLVED

ONE_SITE = "one_site"
TWO_SITE = "two_site"
DILUTION = "dilution"
MODEL_IDS = (ONE_SITE, TWO_SITE, DILUTION)


class FitStatus(str, Enum):
    FITTED = "FITTED"
    BUDGET_EXHAUSTED = BUDGET_EXHAUSTED
    SEMANTIC_UNRESOLVED = SEMANTIC_UNRESOLVED
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class FitRequest:
    net_heats: list[float]          # ucal, NaN excluded
    volumes_uL: list[float]
    v0_uL: float                    # cell/pool volume
    m0_um: float                    # macromolecule in cell
    l0_um: float                    # ligand in syringe
    discard_first: bool
    equiopt_tol: float = 0.10
    max_evals: int = 4000


@dataclass
class ModelFit:
    model_id: str
    status: str
    params: dict
    sse: float
    predicted: list[float]
    residuals: list[float]
    equi_optimal: bool = False
    diagnostics: list[dict] = field(default_factory=list)
    param_version: str = ""
    inputs_version: str = ""
    stable_id: str = ""

    def to_dict(self) -> dict:
        return {
            "stable_id": self.stable_id,
            "model_id": self.model_id,
            "status": self.status,
            "params": self.params,
            "sse": _f(self.sse),
            "predicted": [_f(x) for x in self.predicted],
            "residuals": [_f(x) for x in self.residuals],
            "equi_optimal": self.equi_optimal,
            "diagnostics": self.diagnostics,
            "param_version": self.param_version,
            "inputs_version": self.inputs_version,
        }


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def prepare_vectors(req: FitRequest):
    q, v = [], []
    for i, (heat, vol) in enumerate(zip(req.net_heats, req.volumes_uL)):
        if heat is None or (isinstance(heat, float) and math.isnan(heat)):
            continue
        if req.discard_first and i == 0:
            continue
        q.append(heat); v.append(vol)
    return np.array(q, dtype=float), np.array(v, dtype=float)


def _ligand_cumulative(v: np.ndarray, v0: float, l0: float) -> np.ndarray:
    # X_i ligand concentration in cell, accounting for displaced volume
    x = np.zeros_like(v)
    disp = 1.0
    total = 0.0
    for i, dvi in enumerate(v):
        total += dvi * disp
        disp *= (1.0 - dvi / v0)
        x[i] = l0 * total / v0
    return x


def _one_site_q(theta, x: np.ndarray, m0: float, v0: float, nfix=None
                ) -> np.ndarray:
    if nfix is None:
        n, lgk, dh = theta
    else:
        n, (lgk, dh) = nfix, theta
    # K stored per M; concentrations x,m0 are uM, so 1/K[M] expressed in uM
    k = 10.0 ** lgk
    mt = n * m0
    s_sum = mt + x + 1.0e6 / k
    bound = (s_sum - np.sqrt(np.maximum(s_sum ** 2 - 4.0 * mt * x, 0.0))) \
        / (2.0 * mt)
    # m0[uM]*v0[uL] is pmol; dh[uCal/umol]; -> heat in uCal needs 1e-6
    qtot = (m0 * v0 / 1_000_000.0) * n * dh * bound
    return _diff(qtot)


def _site_cumulative(n, lgk, dh, x, m0, v0) -> np.ndarray:
    # K stored per M; concentrations x,m0 are uM, so 1/K[M] expressed in uM
    k = 10.0 ** lgk
    mt = n * m0
    s_sum = mt + x + 1.0e6 / k
    bound = (s_sum - np.sqrt(np.maximum(s_sum ** 2 - 4.0 * mt * x, 0.0))) \
        / (2.0 * mt)
    return (m0 * v0 / 1_000_000.0) * n * dh * bound  # ucal cumulative


def _diff(qtot: np.ndarray) -> np.ndarray:
    q = np.empty_like(qtot)
    q[0] = qtot[0]
    q[1:] = qtot[1:] - qtot[:-1]
    return q


def _two_site_q(theta, x, m0, v0) -> np.ndarray:
    n1, lgk1, dh1, n2, lgk2, dh2 = theta
    qtot = (_site_cumulative(n1, lgk1, dh1, x, m0, v0)
            + _site_cumulative(n2, lgk2, dh2, x, m0, v0))
    return _diff(qtot)


def _dilution_q(theta, v, l0) -> np.ndarray:
    a, b = theta
    return a + b * (v * l0 / 1000.0)  # ucal vs nmol ligand injected


def _nelder_mead(fn, x0, max_evals, init_step=0.1):
    """Small bounded-by-transform Nelder-Mead. Works in raw positive space via
    internal log transform; returns (x, f, evals, converged)."""
    x = np.array(x0, dtype=float)
    evals = 0
    converged = False

    def fwrap(z):
        nonlocal evals
        evals += 1
        return fn(z)

    n = len(x)
    simplex = [x.copy()]
    for j in range(n):
        y = x.copy()
        y[j] += init_step * (abs(y[j]) + 1.0)
        simplex.append(y)
    fvals = [fwrap(s) for s in simplex]
    while evals < max_evals:
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) <= 1e-8 * (1.0 + abs(fvals[0])):
            converged = True
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = fwrap(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = fwrap(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue
        xc = centroid - 0.5 * (centroid - simplex[-1])
        fc = fwrap(xc)
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        # shrink
        for j in range(1, n + 1):
            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
            fvals[j] = fwrap(simplex[j])
    return simplex[0], fvals[0], evals, converged


def fit_model(model_id: str, req: FitRequest, param_version: str,
              inputs_version: str) -> ModelFit:
    q, v = prepare_vectors(req)
    if len(q) < 4 or not math.isfinite(req.v0_uL) or req.v0_uL <= 0:
        return ModelFit(model_id, FitStatus.INSUFFICIENT_DATA.value,
                        {}, math.nan, [_f(x) for x in q], [],
                        diagnostics=[{"code": "INSUFFICIENT_DATA",
                                      "message": "need >=4 valid injections and "
                                                 "finite pool volume"}],
                        param_version=param_version,
                        inputs_version=inputs_version)
    x = _ligand_cumulative(v, req.v0_uL, req.l0_um)

    if model_id == DILUTION:
        pred = _dilution_q
        design = np.column_stack([np.ones_like(v), v * req.l0_um / 1000.0])
        coef, *_ = np.linalg.lstsq(design, q, rcond=None)
        p = coef
        y = _dilution_q(p, v, req.l0_um)
        sse = float(np.sum((y - q) ** 2))
        return ModelFit(model_id, FitStatus.FITTED.value,
                        {"intercept_ucal": float(p[0]),
                         "slope_ucal_per_nmol": float(p[1])}, sse,
                         list(map(float, y)), list(map(float, y - q)),
                         param_version=param_version,
                         inputs_version=inputs_version)

    if model_id == ONE_SITE:
        starts = [(n, k, d) for n in (0.5, 1.0, 2.0)
                  for k in (3.5, 4.5, 5.0, 5.5, 6.5, 7.5)
                  for d in (-1.2e4, -6e3, -2e3, -500.0, 500.0, 3e3)]
        def objective(theta):
            n, lgk, dh = theta
            # measurable scientific domain; degenerate solutions are illegal
            if not (0.05 <= n <= 10.0 and 2.0 <= lgk <= 9.0
                    and abs(dh) <= 1e6):
                return 1e30
            y = _one_site_q(theta, x, req.m0_um, req.v0_uL)
            if not np.isfinite(y).all():
                return 1e30
            return float(np.sum((y - q) ** 2))
        best, best_sse, evals, converged = _multistart(
            objective, starts, req.max_evals)
        if best is None:
            return _semantic(model_id, q, param_version, inputs_version,
                             "one_site produced no finite solution")
        y = _one_site_q(best, x, req.m0_um, req.v0_uL)
        if not _legal_one_site(best):
            return _semantic(model_id, q, param_version, inputs_version,
                             "one_site optimum lies outside measurable domain")
        return ModelFit(
            model_id,
            FitStatus.FITTED.value if converged else
            FitStatus.BUDGET_EXHAUSTED.value,
            {"N": float(best[0]), "K_per_M": float(10 ** best[1]),
             "dH_ucal_per_umol": float(best[2])},
            best_sse, list(map(float, y)), list(map(float, y - q)),
            diagnostics=[] if converged else [{
                "code": BUDGET_EXHAUSTED,
                "message": f"optimizer eval budget {req.max_evals} exhausted "
                           f"before convergence"}],
            param_version=param_version, inputs_version=inputs_version)

    if model_id == TWO_SITE:
        starts = [(n1, k1, d1, n2, k2, d2)
                  for n1 in (0.5, 1.0) for n2 in (0.5, 1.5)
                  for k1, k2 in ((4.5, 6.0), (6.0, 4.5), (5.0, 5.5),
                                 (5.5, 5.0))
                  for d1, d2 in ((-6e3, -1e3), (-1e3, -6e3),
                                 (-4e3, -3e3))]
        def objective(theta):
            n1, lgk1, dh1, n2, lgk2, dh2 = theta
            # both classes must be distinct, populated and measurable; this
            # rejects the degenerate "empty second site" pseudo-candidate
            if not (0.05 <= min(n1, n2) and max(n1, n2) <= 10.0
                    and 2.0 <= min(lgk1, lgk2)
                    and max(lgk1, lgk2) <= 9.0
                    and abs(lgk1 - lgk2) >= 0.3
                    and max(abs(dh1), abs(dh2)) <= 1e6):
                return 1e30
            y = _two_site_q(theta, x, req.m0_um, req.v0_uL)
            if not np.isfinite(y).all():
                return 1e30
            return float(np.sum((y - q) ** 2))
        best, best_sse, evals, converged = _multistart(
            objective, starts, req.max_evals)
        if best is None:
            return _semantic(model_id, q, param_version, inputs_version,
                             "two_site produced no finite solution")
        y = _two_site_q(best, x, req.m0_um, req.v0_uL)
        if not _legal_two_site(best):
            return _semantic(model_id, q, param_version, inputs_version,
                             "two_site optimum is degenerate (an empty or "
                             "unresolved site class)")
        return ModelFit(
            model_id,
            FitStatus.FITTED.value if converged else
            FitStatus.BUDGET_EXHAUSTED.value,
            {"N1": float(best[0]), "K1_per_M": float(10 ** best[1]),
             "dH1_ucal_per_umol": float(best[2]),
             "N2": float(best[3]), "K2_per_M": float(10 ** best[4]),
             "dH2_ucal_per_umol": float(best[5])},
            best_sse, list(map(float, y)), list(map(float, y - q)),
            diagnostics=[] if converged else [{
                "code": BUDGET_EXHAUSTED,
                "message": f"optimizer eval budget {req.max_evals} exhausted "
                           f"before convergence"}],
            param_version=param_version, inputs_version=inputs_version)
    return _semantic(model_id, q, param_version, inputs_version,
                     f"unknown model {model_id}")



def _legal_one_site(theta) -> bool:
    n, lgk, _ = theta
    return (0.06 <= n <= 9.0 and 2.2 <= lgk <= 8.8)


def _legal_two_site(theta) -> bool:
    n1, k1, _, n2, k2, _ = theta
    if not (0.06 <= min(n1, n2) and max(n1, n2) <= 9.0):
        return False
    if not (2.2 <= min(k1, k2) and max(k1, k2) <= 8.8):
        return False
    # both classes carry appreciable occupancy and are resolvably distinct
    return min(n1, n2) >= 0.1 and abs(k1 - k2) >= 0.35


def _multistart(objective, starts, max_evals):
    # Coarse exploration from a handful of starts, then a final refinement of
    # the best point using the remaining budget. BUDGET_EXHAUSTED is reported
    # only if even the best point never reached the convergence test.
    coarse = starts[:6]
    budget_each = max(40, (max_evals // 2) // max(1, len(coarse)))
    used = 0
    best, best_sse, any_converged = None, math.inf, False
    for s0 in coarse:
        remain = max_evals - used
        if remain <= 0:
            break
        x, fx, ev, conv = _nelder_mead(objective, np.array(s0, float),
                                       min(budget_each, remain))
        used += ev
        if math.isfinite(fx) and fx < best_sse:
            best, best_sse, any_converged = x, fx, conv
        any_converged = any_converged or conv
    if best is not None and used < max_evals:
        x, fx, ev, conv = _nelder_mead(objective, best, max_evals - used,
                                       init_step=0.02)
        used += ev
        if math.isfinite(fx) and fx < best_sse:
            best, best_sse = x, fx
        any_converged = any_converged or conv
    return best, (best_sse if best is not None else math.nan), used, \
        any_converged


def _semantic(model_id, q, pv, iv, message):
    return ModelFit(model_id, FitStatus.SEMANTIC_UNRESOLVED.value, {},
                    math.nan, [_f(x) for x in q], [],
                    diagnostics=[{"code": SEMANTIC_UNRESOLVED,
                                  "message": message}],
                    param_version=pv, inputs_version=iv)


def compare(fits: list[ModelFit], tol: float) -> list[ModelFit]:
    """Mark every candidate within relative tolerance of the best SSE as an
    equi-optimal legal candidate."""
    valid = [f for f in fits if f.status == FitStatus.FITTED.value]
    if not valid:
        return fits
    best = min(f.sse for f in valid)
    for f in fits:
        f.equi_optimal = (f.status == FitStatus.FITTED.value
                          and (f.sse - best) <= tol * max(1.0, best))
    return fits
