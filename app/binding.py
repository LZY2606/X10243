"""Binding model candidates fitted with NumPy-only deterministic optimizers.

Models compared during review:
  * one_site  - Wiseman single-site isotherm (n, K, dH);
  * two_site  - two independent site classes (n1,n2,K1,K2,dH1,dH2);
  * dilution  - dilution-only null hypothesis (constant mean heat).

All candidates are always retained.  Candidates within an AICc tolerance of
the best score are jointly "co-optimal"; none is fabricated into a unique
winner.  Model order for equal scores is stable (lexicographic by key).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .errors import BudgetExhausted, IncompleteInput

AICC_TIE_TOLERANCE = 2.0


@dataclass
class FitResult:
    model: str
    parameters: dict[str, float]
    parameter_units: dict[str, str]
    aic: float
    aicc: float
    rss: float
    n: int
    k: int
    converged: bool
    evaluations: int
    budget: int
    predicted: list[float]
    status: str
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    co_optimal: bool = False

    def to_dict(self):
        return asdict(self)

    def summary(self):
        d = self.to_dict()
        d.pop("predicted", None)
        return d


def _wiseman_one_site(x, n, logk, dh, m0, v0, dvol):
    """Heat per injection (µcal), MicroCal Wiseman with volume displacement."""
    ka = 10.0 ** logk
    a = x / (n * m0) + 1.0 + 1.0 / (ka * n * m0)
    theta = np.clip(0.5 * (a - np.sqrt(np.maximum(a * a - 4.0 * x / (n * m0), 0.0))),
                    0.0, 1.0)
    q = v0 * dh * m0 * n * theta
    dq = np.empty_like(q)
    dq[0] = q[0]
    for i in range(1, len(q)):
        prev = 0.5 * (float(q[i]) + float(q[i - 1]))
        dq[i] = q[i] + float(dvol[i]) / v0 * prev - q[i - 1]
    return dq


def _wiseman_two_site(x, n1, n2, logk1, logk2, dh1, dh2, m0, v0, dvol):
    def theta(x, nn, ka):
        a = x / (nn * m0) + 1.0 + 1.0 / (ka * nn * m0)
        return np.clip(0.5 * (a - np.sqrt(np.maximum(a * a - 4.0 * x / (nn * m0), 0.0))),
                       0.0, 1.0)
    t1 = theta(x, n1, 10.0 ** logk1)
    t2 = theta(x, n2, 10.0 ** logk2)
    q = v0 * (dh1 * m0 * n1 * t1 + dh2 * m0 * n2 * t2)
    dq = np.empty_like(q)
    dq[0] = q[0]
    for i in range(1, len(q)):
        prev = 0.5 * (float(q[i]) + float(q[i - 1]))
        dq[i] = q[i] + float(dvol[i]) / v0 * prev - q[i - 1]
    return dq


def _aicc(rss, n, k):
    if rss <= 0 or not math.isfinite(rss):
        rss = 1e-12
    aic = n * math.log(rss / n) + 2 * k
    corr = (2 * k * (k + 1)) / max(n - k - 1, 1)
    return aic, aic + corr


class _Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def tick(self):
        self.used += 1
        if self.used > self.limit:
            raise BudgetExhausted(
                "fitting budget exhausted",
                [{"code": "budget_exhausted", "budget": self.limit,
                  "used": self.used}])


def _grid_minimize(predict, y, grid, budget):
    best = None
    for combo in grid:
        budget.tick()
        try:
            pred = predict(combo)
        except (FloatingPointError, OverflowError, ValueError):
            continue
        if not np.all(np.isfinite(pred)):
            continue
        rss = float(np.sum((pred - y) ** 2))
        if best is None or rss < best[0]:
            best = (rss, combo, pred)
    return best


def _coarse_grids():
    n_grid = np.array([0.5, 1.0, 1.5, 2.0])
    logk_grid = np.arange(3.0, 9.01, 0.5)
    dh_grid = np.array([-2e7, -1e7, -5e6, -2e6, -1e6, 0.0,
                        1e6, 2e6, 5e6, 1e7, 2e7])
    return n_grid, logk_grid, dh_grid


def _local_refine(predict, y, p0, scales, budget, max_steps=12):
    p = np.array(p0, dtype=float)
    eps_rel = 1e-5
    lam = 1e-3
    converged = False
    rss = None
    pred = None
    for _ in range(max_steps):
        budget.tick()
        pred = predict(p)
        r = pred - y
        rss = float(r @ r)
        jac = np.empty((len(y), len(p)))
        for j in range(len(p)):
            dp = np.zeros_like(p)
            step = eps_rel * max(abs(p[j]), scales[j])
            dp[j] = step
            jac[:, j] = (predict(p + dp) - pred) / step
        jtj = jac.T @ jac
        try:
            delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-300),
                                    -jac.T @ r)
        except np.linalg.LinAlgError:
            break
        p_new = p + delta
        budget.tick()
        try:
            pred_new = predict(p_new)
        except (FloatingPointError, OverflowError, ValueError):
            lam *= 10
            continue
        if not np.all(np.isfinite(pred_new)):
            lam *= 10
            continue
        rss_new = float((pred_new - y) @ (pred_new - y))
        if rss_new < rss:
            p, pred, rss = p_new, pred_new, rss_new
            lam = max(lam / 3, 1e-9)
            if float(np.linalg.norm(delta / np.maximum(scales, 1e-30))) < 1e-7:
                converged = True
                break
        else:
            lam *= 10
    return p, float(rss), pred.tolist(), converged


def _fit_one_site(x, y, dvol, m0, v0, budget):
    n_grid, logk_grid, dh_grid = _coarse_grids()
    best = None
    for n in n_grid:
        for logk in logk_grid:
            for dh in dh_grid:
                budget.tick()
                pred = _wiseman_one_site(x, n, logk, dh, m0, v0, dvol)
                if not np.all(np.isfinite(pred)):
                    continue
                rss = float(np.sum((pred - y) ** 2))
                if best is None or rss < best[0]:
                    best = (rss, (n, logk, dh), pred)
    if best is None:
        raise BudgetExhausted("no feasible one-site evaluation",
                              [{"code": "no_feasible_point"}])
    p, rss, pred, conv = _local_refine(
        lambda pp: _wiseman_one_site(x, pp[0], pp[1], pp[2], m0, v0, dvol),
        y, best[1], np.array([1.0, 2.0, 1e7]), budget)
    aic, aicc = _aicc(rss, len(y), 3)
    return FitResult("one_site",
                     {"n": float(p[0]), "log10_K_M-1": float(p[1]),
                      "dH_ucal_per_mol": float(p[2])},
                     {"n": "sites/molecule", "log10_K_M-1": "log10(M^-1)",
                      "dH_ucal_per_mol": "µcal/mol"},
                     aic, aicc, rss, len(y), 3, conv, budget.used,
                     budget.limit, list(pred), "ok")


def _fit_two_site(x, y, dvol, m0, v0, budget):
    # Stage 1: best one-site seed (shared coarse grid).
    n_grid, logk_grid, dh_grid = _coarse_grids()
    seed = None
    for n in n_grid:
        for logk in logk_grid:
            for dh in dh_grid:
                budget.tick()
                pred = _wiseman_one_site(x, n, logk, dh, m0, v0, dvol)
                if not np.all(np.isfinite(pred)):
                    continue
                rss = float(np.sum((pred - y) ** 2))
                if seed is None or rss < seed[0]:
                    seed = (rss, n, logk, dh)
    n0, k0, h0 = seed[1], seed[2], seed[3]
    # Stage 2: local perturbation of a second site around the first.
    n2_grid = np.array([0.25, 0.5, 1.0, 1.5])
    k2_grid = k0 + np.array([-1.5, -0.5, 0.5, 1.5])
    h2_grid = h0 * np.array([0.5, 1.0, 1.5])
    best = None
    for n2 in n2_grid:
        for k2 in k2_grid:
            for h2 in h2_grid:
                budget.tick()
                p = (n0, n2, k0, np.clip(k2, 2.0, 10.0), h0, h2)
                pred = _wiseman_two_site(x, *p, m0, v0, dvol)
                if not np.all(np.isfinite(pred)):
                    continue
                rss = float(np.sum((pred - y) ** 2))
                if best is None or rss < best[0]:
                    best = (rss, p, pred)
    p, rss, pred, conv = _local_refine(
        lambda pp: _wiseman_two_site(x, *pp, m0, v0, dvol),
        y, np.array(best[1], dtype=float),
        np.array([1.0, 1.0, 2.0, 2.0, 1e7, 1e7]), budget)
    aic, aicc = _aicc(rss, len(y), 6)
    return FitResult("two_site",
                     {"n1": float(p[0]), "n2": float(p[1]),
                      "log10_K1_M-1": float(p[2]), "log10_K2_M-1": float(p[3]),
                      "dH1_ucal_per_mol": float(p[4]),
                      "dH2_ucal_per_mol": float(p[5])},
                     {"n1": "sites/molecule", "n2": "sites/molecule",
                      "log10_K1_M-1": "log10(M^-1)",
                      "log10_K2_M-1": "log10(M^-1)",
                      "dH1_ucal_per_mol": "µcal/mol",
                      "dH2_ucal_per_mol": "µcal/mol"},
                     aic, aicc, rss, len(y), 6, conv, budget.used,
                     budget.limit, list(pred), "ok")


def _fit_dilution(x, y, dvol, m0, v0, budget):
    budget.tick()
    mean = float(np.mean(y))
    pred = np.full_like(y, mean)
    rss = float(np.sum((pred - y) ** 2))
    aic, aicc = _aicc(rss, len(y), 1)
    return FitResult("dilution", {"mean_q_ucal": mean},
                     {"mean_q_ucal": "µcal"}, aic, aicc, rss, len(y), 1,
                     True, budget.used, budget.limit, pred.tolist(), "ok")


def cumulative_ligand_concentration(volumes_ul, v0_ul, syringe_m):
    """Exact cumulative ligand concentration after each injection (M).

    Accounts for cell volume displacement; canonical MicroCal recursion.
    """
    x = []
    conc = 0.0
    for dv in volumes_ul:
        conc = conc + (dv / v0_ul) * (syringe_m - conc)
        x.append(conc)
    return np.array(x, dtype=float)


def fit_models(net_q_ucal, volumes_ul, syringe_m, macromolecule_m, v0_ul,
               discarded_indices=(), budget=20000):
    """Fit every candidate and return them all with co-optimality flags.

    Diagnostics are positional: rejected injections and missing values are
    reported with injection indices, never silently skipped with defaults.
    """
    if syringe_m is None or macromolecule_m is None or v0_ul is None:
        raise IncompleteInput(
            "binding fit requires syringe concentration, cell concentration "
            "and cell volume",
            [{"code": "missing_fit_input",
              "position": {"syringe_m": syringe_m,
                           "macromolecule_m": macromolecule_m,
                           "v0_ul": v0_ul}}])
    if len(net_q_ucal) != len(volumes_ul):
        raise IncompleteInput(
            "heat and volume vectors differ in length",
            [{"code": "length_mismatch",
              "position": {"n_heat": len(net_q_ucal),
                           "n_volume": len(volumes_ul)}}])
    keep = [i for i in range(len(net_q_ucal))
            if i not in set(discarded_indices)
            and net_q_ucal[i] is not None and volumes_ul[i] is not None]
    diag = []
    for i in range(len(net_q_ucal)):
        if i in set(discarded_indices):
            diag.append({"code": "injection_discarded", "index": i,
                         "detail": "excluded by first-injection/decision rule"})
        elif net_q_ucal[i] is None or volumes_ul[i] is None:
            diag.append({"code": "point_unavailable", "index": i,
                         "detail": "no net heat (gap or missing blank)"})
    if len(keep) < 4:
        raise IncompleteInput(
            "fewer than 4 usable injections; binding models are not identifiable",
            diag + [{"code": "insufficient_points", "usable": len(keep)}])
    y = np.array([net_q_ucal[i] for i in keep], dtype=float)
    vols = np.array([volumes_ul[i] for i in keep], dtype=float)
    x = cumulative_ligand_concentration(vols, v0_ul, syringe_m)
    dvol = vols
    b = _Budget(budget)
    results = []
    for fitter in (_fit_one_site, _fit_two_site, _fit_dilution):
        try:
            results.append(fitter(x, y, dvol, macromolecule_m, v0_ul, b))
        except BudgetExhausted:
            # Budget exhaustion is a distinct outcome: surface per model and
            # also re-raise so callers can choose hard/soft handling.
            partial = FitResult(
                fitter.__name__, {}, {}, float("inf"), float("inf"),
                float("inf"), len(keep), 0, False, b.used, b.limit, [],
                "budget_exhausted",
                [{"code": "budget_exhausted", "budget": budget, "used": b.used}])
            results.append(partial)
    exhausted = [r for r in results if r.status == "budget_exhausted"]
    feasible = [r for r in results if r.status == "ok"]
    if not feasible:
        raise BudgetExhausted(
            "all model fits exhausted the budget before any feasible point",
            [{"code": "budget_exhausted", "budget": budget,
              "evaluations": b.used}])
    best_aicc = min(r.aicc for r in feasible)
    for r in feasible:
        r.co_optimal = (r.aicc - best_aicc) <= AICC_TIE_TOLERANCE
        if diag:
            r.diagnostics.extend(diag)
    # stable order: aicc then lexicographic model key
    feasible.sort(key=lambda r: (round(r.aicc, 9), r.model))
    return {"results": feasible + exhausted,
            "best_aicc": best_aicc,
            "tie_tolerance": AICC_TIE_TOLERANCE,
            "co_optimal_models": [r.model for r in feasible if r.co_optimal],
            "usable_indices": keep,
            "x_ligand_M": x.tolist(),
            "y_net_q_ucal": y.tolist(),
            "volumes_ul": vols.tolist()}
