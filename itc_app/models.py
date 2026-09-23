"""Binding model candidates: one-site, two-site, dilution.

Inputs are canonicalised in the derivation layer only. Each candidate records
its input summary, rule version and parent dependency (material versions +
integration result). Equivalent inputs produce an identical, stable ordering
of candidates. Budget exhaustion is a distinct result from parse/semantic
failures upstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .units import RULES_VERSION, CANONICAL_UNIT
from .materials import MaterialSet
from .integration import IntegrationRun
from .materials import BudgetExhausted

AICC_EQUIFINAL_TOL = 2.0
DEFAULT_BUDGET = 4000  # total Levenberg-Marquardt iterations across multistarts

# dH in kcal/mol, concentrations mM, volume uL, heat uJ
KDH_TO_UJ_PER_UMOL = 4184.0


@dataclass
class FitPoint:
    injection_index: int
    ratio: float          # syringe/cell concentration ratio after injection
    heat_uJ: float        # observed net heat

    def to_dict(self) -> dict:
        return {"injection_index": self.injection_index, "ratio": self.ratio, "heat_uJ": self.heat_uJ}


@dataclass
class FitCandidate:
    name: str
    params: dict
    sse: float
    aicc: float
    converged: bool
    residuals: list[float]
    predictions: list[float]
    k: int
    tied: bool = False
    staleness_token: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "sse": self.sse,
            "aicc": self.aicc,
            "converged": self.converged,
            "residuals": self.residuals,
            "predictions": self.predictions,
            "k": self.k,
            "tied": self.tied,
            "staleness_token": self.staleness_token,
        }


@dataclass
class FitRun:
    rule_version: str
    input_summary: dict
    parents: dict
    points: list[FitPoint]
    candidates: list[FitCandidate]
    selected: list[str]
    blocked: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule_version": self.rule_version,
            "input_summary": self.input_summary,
            "parents": self.parents,
            "points": [p.to_dict() for p in self.points],
            "candidates": [c.to_dict() for c in self.candidates],
            "selected": self.selected,
            "blocked": self.blocked,
        }


def prepare_points(mat: MaterialSet, integ: IntegrationRun):
    """Build fitting points + blocked list. First injection is excluded."""
    if mat.cell_conc is None or mat.syringe_conc is None or mat.cell_volume is None:
        missing = [
            n
            for n, q in (
                ("cell_concentration", mat.cell_conc),
                ("syringe_concentration", mat.syringe_conc),
                ("cell_volume", mat.cell_volume),
            )
            if q is None or not q.convertible
        ]
        return [], {}, [
            {"code": "E_FIT_INPUT_MISSING", "message": f"missing/invalid {m}", "position": "@" + m.split("_")[0]}
            for m in missing
        ]
    M0 = mat.cell_conc.canonical
    X0 = mat.syringe_conc.canonical
    V = mat.cell_volume.canonical
    injected = 0.0
    vol_by_idx = {inj.index: inj.volume.canonical for inj in mat.injections}
    points, blocked = [], []
    for r in integ.results:
        vi = vol_by_idx.get(r.injection_index, 0.0)
        injected += vi or 0.0
        denom = V + injected
        ratio = X0 * injected / (M0 * denom) if denom > 0 else float("nan")
        if r.discarded:
            blocked.append({"injection_index": r.injection_index, "reason": "first_injection_discarded"})
            continue
        if r.status.startswith("blocked"):
            blocked.append({"injection_index": r.injection_index, "reason": r.status})
            continue
        if r.net_heat_uJ is None:
            blocked.append({"injection_index": r.injection_index, "reason": "blank_unmatched_no_net_heat"})
            continue
        points.append(FitPoint(r.injection_index, ratio, r.net_heat_uJ))
    summary = {
        "M0": {"value": M0, "unit": CANONICAL_UNIT["concentration"]},
        "X0": {"value": X0, "unit": CANONICAL_UNIT["concentration"]},
        "V": {"value": V, "unit": CANONICAL_UNIT["volume"]},
        "n_points": len(points),
        "n_blocked": len(blocked),
        "temperature_K": None if mat.temperature is None else mat.temperature.canonical,
    }
    return points, summary, blocked


def _bound_Q(ratio: np.ndarray, K, n, M0_mM, V_uL):
    """Bound-site fill fraction theta for one site with ligand depletion."""
    Mt = M0_mM
    Lt = ratio * Mt
    nK = n * K
    b = nK * Mt + nK * Lt + 1.0
    disc = np.maximum(b * b - 4.0 * (nK ** 2) * Mt * Lt, 0.0)
    theta = (b - np.sqrt(disc)) / (2.0 * nK * Mt)
    theta = np.clip(theta, 0.0, 1.0)
    return theta


def onesite_predict(params, ratios, M0_mM, V_uL):
    logK, n, dH, q0 = params
    K = np.exp(logK)
    theta = _bound_Q(np.asarray(ratios, float), K, n, M0_mM, V_uL)
    total = theta * n * M0_mM * V_uL * 1e-3 * dH * KDH_TO_UJ_PER_UMOL
    qi = np.empty_like(total)
    qi[0] = total[0]
    qi[1:] = np.diff(total)
    return qi + q0


def dilution_predict(params, ratios):
    return np.full_like(np.asarray(ratios, float), params[0], dtype=float)


def twosite_predict(params, ratios, M0_mM, V_uL):
    logK1, logK2, n1, n2, dH1, dH2, q0 = params
    M = M0_mM
    Lt = np.asarray(ratios, float) * M
    occ = np.zeros_like(Lt)
    Ks = [(np.exp(logK1), n1, dH1), (np.exp(logK2), n2, dH2)]
    for K, nn, dH in Ks:
        b = K * nn * M + K * Lt + 1.0
        disc = np.maximum(b * b - 4.0 * K * K * nn * M * Lt, 0.0)
        theta = np.clip((b - np.sqrt(disc)) / (2.0 * K * nn * M), 0.0, 1.0)
        occ += theta * nn * dH
    total = occ * M * V_uL * 1e-3 * KDH_TO_UJ_PER_UMOL
    qi = np.empty_like(total)
    qi[0] = total[0]
    qi[1:] = np.diff(total)
    return qi + q0


def _aicc(sse: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return float("inf")
    return n * math.log(max(sse, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _lm_fit(model, p0, x, y, *, lo, hi, max_iter, budget_box):
    """Damped Gauss-Newton with parameter scaling; returns (p, sse, iters, ok)."""
    p = np.asarray(p0, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    scale = np.where(hi - lo > 0, hi - lo, 1.0)

    def unbox(z):
        u = 1.0 / (1.0 + np.exp(-z))
        return lo + (hi - lo) * u

    def box(z):
        q = np.clip((z - lo) / scale, 1e-9, 1 - 1e-9)
        return np.log(q / (1.0 - q))

    z = box(p)
    lam = 1e-3
    sse = float(np.sum((model(unbox(z)) - y) ** 2))
    iters = 0
    while iters < max_iter:
        if budget_box[0] <= 0:
            raise BudgetExhausted("LM iteration budget exhausted before convergence", DEFAULT_BUDGET)
        budget_box[0] -= 1
        iters += 1
        r = model(unbox(z)) - y
        J = np.zeros((len(y), len(z)))
        eps = 1e-6
        for j in range(len(z)):
            dz = z.copy()
            dz[j] += eps
            J[:, j] = (model(unbox(dz)) - y - r) / eps
        g = J.T @ r
        H = J.T @ J
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        z_new = z + step
        p_new = unbox(z_new)
        r_new = model(p_new) - y
        sse_new = float(np.sum(r_new ** 2))
        if sse_new < sse:
            if abs(sse - sse_new) <= 1e-10 * max(1.0, sse):
                z = z_new
                sse = sse_new
                break
            z, sse, lam = z_new, sse_new, max(lam * 0.4, 1e-9)
        else:
            lam *= 3.0
            if lam > 1e12:
                break
    converged = iters < max_iter and np.isfinite(sse)
    return unbox(z), sse, iters, converged
