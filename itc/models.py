"""Binding model candidates: one-site, two-site and dilution-only.

All fits record their input digest, parameter version and parent
dependencies at the storage layer (itc.store). Here we keep the pure
math. Fitting is budgeted: exceeding the evaluation budget raises
BudgetExhausted, which is a distinct outcome from format or semantic
failures.

Equally-good models: candidates whose AICc lies within EQUAL_AICC of
the best are all kept as legal candidates; nothing is auto-adopted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .errors import BudgetExhausted, SemanticUnsolvable

MODEL_ONE_SITE = "one_site"
MODEL_TWO_SITE = "two_site"
MODEL_DILUTION = "dilution_only"
MODEL_KINDS = (MODEL_ONE_SITE, MODEL_TWO_SITE, MODEL_DILUTION)

EQUAL_AICC = 2.0
PARAMS_VERSION = "itc-models/1.0"

DEFAULT_BUDGET = 4000


@dataclass
class FitInput:
    """Canonicalised per-injection data used by the fits."""
    volumes_uL: List[float]
    heats_uJ: List[float]
    cell_conc_mM: float
    syringe_conc_mM: float
    cell_volume_uL: float
    temperature_K: float


def _cell_concentrations(inp: FitInput):
    """Macromolecule / titrant concentration in the cell after each
    injection, with displacement dilution. Returns (Mt_i, Xt_i) in mM."""
    v0 = inp.cell_volume_uL
    mt = np.empty(len(inp.volumes_uL))
    xt = np.empty(len(inp.volumes_uL))
    m_prev, x_prev = inp.cell_conc_mM, 0.0
    for i, v in enumerate(inp.volumes_uL):
        f = v / v0
        m_prev = m_prev * (1.0 - f)
        x_prev = x_prev * (1.0 - f) + inp.syringe_conc_mM * f
        mt[i] = m_prev
        xt[i] = x_prev
    return mt, xt


def _theta(ka_per_mM: float, n: float, mt: np.ndarray, xt: np.ndarray):
    """Occupancy solving Ka*(Xt - n*Mt*th)*(1-th) - th = 0 (quadratic)."""
    a = ka_per_mM * n * mt
    b = -(ka_per_mM * xt + ka_per_mM * n * mt + 1.0)
    c = ka_per_mM * xt
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    sqrt_disc = np.sqrt(disc)
    th = np.where(
        np.abs(a) > 1e-12,
        (-b - sqrt_disc) / (2.0 * np.where(np.abs(a) > 1e-12, a, 1.0)),
        c / np.maximum(-b, 1e-12),
    )
    return np.clip(th, 0.0, 1.0)


def _incremental(q_cum_uJ: np.ndarray, volumes_uL: Sequence[float],
                 cell_volume_uL: float) -> np.ndarray:
    """Displacement-corrected incremental heats from cumulative heats."""
    dq = np.empty_like(q_cum_uJ)
    q_prev = 0.0
    for i, v in enumerate(volumes_uL):
        dq[i] = q_cum_uJ[i] - q_prev * (1.0 - v / cell_volume_uL)
        q_prev = q_cum_uJ[i]
    return dq


def one_site_heats(params, inp: FitInput) -> np.ndarray:
    log_ka, n, dh = params
    ka = 10.0 ** log_ka
    mt, xt = _cell_concentrations(inp)
    th = _theta(ka, n, mt, xt)
    q_cum = inp.cell_volume_uL * mt * n * dh * th  # uL*mM=nmol; nmol*kJ/mol = uJ
    return _incremental(q_cum, inp.volumes_uL, inp.cell_volume_uL)


def two_site_heats(params, inp: FitInput) -> np.ndarray:
    log_ka1, n1, dh1, log_ka2, n2, dh2 = params
    mt, xt = _cell_concentrations(inp)
    th1 = _theta(10.0 ** log_ka1, n1, mt, xt)
    th2 = _theta(10.0 ** log_ka2, n2, mt, xt)
    q_cum = inp.cell_volume_uL * mt * (n1 * dh1 * th1 + n2 * dh2 * th2)
    return _incremental(q_cum, inp.volumes_uL, inp.cell_volume_uL)


def dilution_heats(params, inp: FitInput) -> np.ndarray:
    (q_const,) = params
    return np.full(len(inp.volumes_uL), q_const, dtype=float)


_PREDICTORS: Dict[str, Callable] = {
    MODEL_ONE_SITE: one_site_heats,
    MODEL_TWO_SITE: two_site_heats,
    MODEL_DILUTION: dilution_heats,
}
_NPARAMS = {MODEL_ONE_SITE: 3, MODEL_TWO_SITE: 6, MODEL_DILUTION: 1}


class _Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self, n: int = 1):
        self.used += n
        if self.used > self.limit:
            raise BudgetExhausted(
                f"evaluation budget exhausted ({self.used} > {self.limit})",
                details={"budget": self.limit})


def _nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray,
                 step: np.ndarray, budget: _Budget,
                 tol: float = 1e-9, max_iter: int = 600) -> tuple:
    n = len(x0)
    simplex = [x0.astype(float)]
    for i in range(n):
        x = x0.astype(float).copy()
        x[i] += step[i]
        simplex.append(x)
    simplex = np.array(simplex)
    vals = np.array([f(x) for x in simplex])
    budget.spend(n + 1)
    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]
        if np.max(np.abs(vals - vals[0])) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = f(xr)
        budget.spend()
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = f(xe)
            budget.spend()
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
            continue
        xc = centroid + 0.5 * (worst - centroid)
        fc = f(xc)
        budget.spend()
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue
        best = simplex[0].copy()
        simplex = np.array([best] + [best + 0.5 * (s - best)
                                     for s in simplex[1:]])
        vals = np.array([vals[0]] + [f(s) for s in simplex[1:]])
        budget.spend(n)
    order = np.argsort(vals)
    return simplex[order][0], float(vals[order][0])


def _rss(predict_fn, params, inp: FitInput) -> float:
    pred = predict_fn(params, inp)
    if not np.all(np.isfinite(pred)):
        return float("inf")
    resid = pred - np.asarray(inp.heats_uJ, dtype=float)
    return float(resid @ resid)


def _aicc(rss: float, n_points: int, k: int) -> float:
    if n_points <= k + 1 or rss <= 0:
        return float("inf")
    return (n_points * math.log(rss / n_points) + 2.0 * k
            + 2.0 * k * (k + 1.0) / (n_points - k - 1.0))


@dataclass
class Candidate:
    model: str
    params: Dict[str, float]
    rss: float
    aicc: float
    status: str = "candidate"        # candidate | equally_good
    diagnostics: list = field(default_factory=list)

    def to_dict(self):
        return {"model": self.model, "params": self.params, "rss": self.rss,
                "aicc": self.aicc, "status": self.status,
                "diagnostics": self.diagnostics}


_PARAM_NAMES = {
    MODEL_ONE_SITE: ("log10_Ka_per_mM", "n", "dH_kJ_mol"),
    MODEL_TWO_SITE: ("log10_Ka1_per_mM", "n1", "dH1_kJ_mol",
                     "log10_Ka2_per_mM", "n2", "dH2_kJ_mol"),
    MODEL_DILUTION: ("q_dilution_uJ",),
}


def fit_model(kind: str, inp: FitInput, budget: _Budget,
              x0: Optional[Sequence[float]] = None) -> Candidate:
    if kind not in _PREDICTORS:
        raise SemanticUnsolvable(f"unknown model kind {kind!r}")
    n_points = len(inp.heats_uJ)
    k = _NPARAMS[kind]
    if n_points < k + 2:
        raise SemanticUnsolvable(
            f"model {kind} needs >= {k + 2} points, got {n_points}",
            details={"model": kind, "n_points": n_points})
    predict = _PREDICTORS[kind]
    heats = np.asarray(inp.heats_uJ, dtype=float)
    scale = float(np.max(np.abs(heats))) or 1.0

    if kind == MODEL_DILUTION:
        q = float(np.mean(heats))
        rss = _rss(predict, (q,), inp)
        return Candidate(model=kind, params={"q_dilution_uJ": q},
                         rss=rss, aicc=_aicc(rss, n_points, k))

    if x0 is None:
        if kind == MODEL_ONE_SITE:
            x0 = (2.0, 1.0, -math.copysign(scale, heats[0] if heats[0] else -1.0))
        else:
            x0 = (2.3, 0.9, -scale, 1.0, 0.9, -scale * 0.3)
    x0 = np.asarray(x0, dtype=float)
    step = np.array([0.3 if abs(v) < 1e-6 else abs(v) * 0.2 for v in x0])
    best_x, best_rss = _nelder_mead(lambda x: _rss(predict, x, inp),
                                    x0, step, budget)
    params = {name: float(v) for name, v in zip(_PARAM_NAMES[kind], best_x)}
    return Candidate(model=kind, params=params, rss=best_rss,
                     aicc=_aicc(best_rss, n_points, k))


def fit_candidates(inp: FitInput, budget_limit: int = DEFAULT_BUDGET,
                   kinds: Sequence[str] = MODEL_KINDS) -> List[Candidate]:
    """Fit all requested model kinds and mark equally-good candidates.

    Deterministic ordering: sorted by (aicc, model name). All legal
    candidates are retained; adoption is a separate human decision.
    """
    budget = _Budget(budget_limit)
    cands: List[Candidate] = []
    one_site_x0 = None
    for kind in kinds:
        if kind == MODEL_TWO_SITE and one_site_x0 is not None:
            lk, n, dh = one_site_x0
            x0 = (lk + 0.3, n * 0.9, dh * 0.8, lk - 0.7, n * 0.5, dh * 0.2)
            cands.append(fit_model(kind, inp, budget, x0=x0))
        else:
            cand = fit_model(kind, inp, budget)
            if kind == MODEL_ONE_SITE:
                one_site_x0 = (cand.params["log10_Ka_per_mM"],
                               cand.params["n"], cand.params["dH_kJ_mol"])
            cands.append(cand)
    cands.sort(key=lambda c: (c.aicc, c.model))
    best = cands[0].aicc
    for c in cands:
        if math.isfinite(c.aicc) and c.aicc - best <= EQUAL_AICC:
            c.status = "equally_good"
    return cands
