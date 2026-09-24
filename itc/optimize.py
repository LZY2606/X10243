"""Deterministic bounded nonlinear least squares in pure NumPy.

No random starts: replay must be bit-stable.  ``BudgetExhausted`` is a distinct
result from format corruption and semantic impossibility (see units).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BudgetExhausted(RuntimeError):
    """The iteration/evaluation budget ran out before convergence."""

    def __init__(self, message: str, *, evaluations: int, iterations: int):
        super().__init__(message)
        self.evaluations = evaluations
        self.iterations = iterations


@dataclass
class FitResult:
    params: np.ndarray
    rss: float
    evaluations: int
    iterations: int
    converged: bool
    budget_exhausted: bool


_EPS = 1e-12


def least_squares(
    residuals_fn,
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    max_iter: int = 200,
    max_evals: int = 2000,
    tol: float = 1e-10,
) -> FitResult:
    """Levenberg–Marquardt on a logit transform of the bounded parameters."""
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    x = np.array(x0, dtype=float)
    x = np.minimum(np.maximum(x, lo + 1e-9 * (hi - lo)), hi - 1e-9 * (hi - lo))

    def to_internal(p: np.ndarray) -> np.ndarray:
        u = (p - lo) / (hi - lo)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        return np.log(u / (1.0 - u))

    def to_external(z: np.ndarray) -> np.ndarray:
        return lo + (hi - lo) / (1.0 + np.exp(-z))

    def cost(z: np.ndarray) -> float:
        r = residuals_fn(to_external(z))
        return float(np.dot(r, r))

    z = to_internal(x)
    lam = 1e-3
    rss = cost(z)
    evals = 1
    iters = 0
    converged = False
    for it in range(1, max_iter + 1):
        iters = it
        p = to_external(z)
        r = residuals_fn(p)
        evals += 1
        # Numerical Jacobian in external coordinates.
        jac = np.empty((len(r), len(p)))
        step = 1e-7 * np.maximum(np.abs(p), 1e-6)
        for j in range(len(p)):
            p2 = p.copy()
            p2[j] += step[j]
            if p2[j] >= hi[j]:
                p2[j] = p[j] - step[j]
                step[j] = -step[j]
            jac[:, j] = (residuals_fn(p2) - r) / step[j]
            evals += 1
            if evals >= max_evals:
                break
        jtj = jac.T @ jac
        g = jac.T @ r
        # Gauss-Newton step in external space, damped, then reflect into bounds.
        accepted = False
        for _ in range(30):
            try:
                dz = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + _EPS), -g)
            except np.linalg.LinAlgError:
                dz = -g / (np.diag(jtj) + _EPS)
            z_new = z + dz
            rss_new = cost(z_new)
            evals += 1
            if rss_new < rss:
                z, rss = z_new, rss_new
                lam = max(lam / 3.0, 1e-12)
                accepted = True
                break
            lam *= 4.0
            if evals >= max_evals:
                break
        if evals >= max_evals:
            break
        if accepted and np.max(np.abs(dz)) < tol:
            converged = True
            break
    budget_exhausted = not converged
    if budget_exhausted and evals >= max_evals:
        raise BudgetExhausted(
            "nonlinear fit reached its evaluation budget",
            evaluations=evals,
            iterations=iters,
        )
    return FitResult(to_external(z), rss, evals, iters, converged, budget_exhausted)
