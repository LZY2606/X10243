"""Binding-model candidates and budgeted fitting.

Models compared on the same integrated injection heats:

* ``one_site``  - Wiseman one-set-of-sites isotherm
* ``two_site``  - two independent sets of sites (multi-site candidate)
* ``dilution``  - pure dilution-heat hypothesis (constant heat per mole)

Fitting is a deterministic coarse grid + local refinement whose number of
model evaluations is bounded by an explicit budget.  Exhausting the budget
raises ``BudgetExhausted``; ill-posed inputs raise ``SemanticUnresolved``;
neither is ever reported as a successful fit.  When several candidates are
statistically indistinguishable (delta AICc <= 2) all of them are kept.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .errors import BudgetExhausted, SemanticUnresolved
from .ids import content_id

MODEL_RULES_VERSION = "binding-models/v1"
EQUIV_AICC_DELTA = 2.0

MODEL_PARAM_COUNTS = {"one_site": 3, "two_site": 4, "dilution": 1}


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def spend(self, n: int = 1) -> None:
        self.used += n
        if self.used > self.limit:
            raise BudgetExhausted(
                f"model evaluation budget exhausted ({self.used} > {self.limit})",
                diagnostics=[{"reason": "BUDGET_EXHAUSTED", "used": self.used, "limit": self.limit}],
            )


def cumulative_composition(
    volumes_L: np.ndarray, cell_volume_L: float, ligand_M: float, macromolecule_M: float
) -> tuple[np.ndarray, np.ndarray]:
    """Total ligand and macromolecule concentration after each injection."""
    cum_v = np.cumsum(volumes_L)
    xt = ligand_M * cum_v / cell_volume_L
    mt = macromolecule_M * (1.0 - cum_v / cell_volume_L)
    return xt, mt


def _theta_one_site(xt: np.ndarray, mt: np.ndarray, n: float, K: float) -> np.ndarray:
    a = 1.0 + K * (xt + n * mt)
    disc = np.sqrt(np.maximum(a * a - 4.0 * K * K * xt * n * mt, 0.0))
    return (a - disc) / (2.0 * K * n * mt)


def heats_one_site(
    xt: np.ndarray, mt: np.ndarray, cell_volume_L: float, n: float, K: float, dH: float
) -> np.ndarray:
    theta = _theta_one_site(xt, mt, n, K)
    Q = cell_volume_L * mt * n * dH * theta
    dQ = np.empty_like(Q)
    dQ[0] = Q[0]
    dQ[1:] = Q[1:] - Q[:-1]
    return dQ


def heats_two_site(
    xt: np.ndarray,
    mt: np.ndarray,
    cell_volume_L: float,
    n1: float,
    K1: float,
    dH1: float,
    n2: float,
    K2: float,
    dH2: float,
) -> np.ndarray:
    return heats_one_site(xt, mt, cell_volume_L, n1, K1, dH1) + heats_one_site(
        xt, mt, cell_volume_L, n2, K2, dH2
    )


def heats_dilution(volumes_L: np.ndarray, ligand_M: float, q_dil_J_per_mol: float) -> np.ndarray:
    return q_dil_J_per_mol * ligand_M * volumes_L


def _rss(pred: np.ndarray, obs: np.ndarray) -> float:
    diff = pred - obs
    return float(diff @ diff)


def _refine(
    model: Callable[..., np.ndarray],
    params: list[float],
    steps: list[float],
    obs: np.ndarray,
    budget: _Budget,
    max_rounds: int = 60,
) -> tuple[list[float], float]:
    best = list(params)
    best_rss = _rss(model(*best), obs)
    budget.spend()
    for _ in range(max_rounds):
        improved = False
        for i, step in enumerate(steps):
            if step < 1e-12:
                continue
            for direction in (1.0, -1.0):
                trial = list(best)
                trial[i] += direction * step
                rss = _rss(model(*trial), obs)
                budget.spend()
                if rss < best_rss - 1e-18:
                    best, best_rss = trial, rss
                    improved = True
        if not improved:
            steps = [s / 2.0 for s in steps]
            if max(steps) < 1e-9:
                break
    return best, best_rss


def _fit_one_site(
    xt: np.ndarray, mt: np.ndarray, obs: np.ndarray, cell_volume_L: float, budget: _Budget
) -> tuple[dict[str, float], float]:
    def model(n: float, logK: float, dH: float) -> np.ndarray:
        return heats_one_site(xt, mt, cell_volume_L, n, math.exp(logK), dH)

    best_params: list[float] | None = None
    best_rss = math.inf
    for n in (0.6, 0.8, 1.0, 1.2, 1.5):
        for logK in (4.0, 6.0, 8.0, 10.0, 12.0, 14.0):
            for dH in (-8e4, -4e4, -2e4, -1e4, 1e4, 4e4):
                rss = _rss(model(n, logK, dH), obs)
                budget.spend()
                if rss < best_rss:
                    best_rss = rss
                    best_params = [n, logK, dH]
    assert best_params is not None
    params, rss = _refine(model, best_params, [0.05, 0.25, 2e3], obs, budget)
    return {"n": params[0], "logK": params[1], "K": math.exp(params[1]), "dH_J_mol": params[2]}, rss


def _fit_two_site(
    xt: np.ndarray, mt: np.ndarray, obs: np.ndarray, cell_volume_L: float, budget: _Budget
) -> tuple[dict[str, float], float]:
    def model(logK1: float, dH1: float, logK2: float, dH2: float) -> np.ndarray:
        return heats_two_site(
            xt, mt, cell_volume_L, 1.0, math.exp(logK1), dH1, 1.0, math.exp(logK2), dH2
        )

    best_params: list[float] | None = None
    best_rss = math.inf
    for logK1 in (8.0, 11.0, 14.0):
        for logK2 in (4.0, 6.5, 9.0):
            if logK2 >= logK1:
                continue
            for dH1 in (-6e4, -3e4, 3e4):
                for dH2 in (-6e4, -1.5e4, 1.5e4):
                    rss = _rss(model(logK1, dH1, logK2, dH2), obs)
                    budget.spend()
                    if rss < best_rss:
                        best_rss = rss
                        best_params = [logK1, dH1, logK2, dH2]
    if best_params is None:
        raise SemanticUnresolved("two-site grid produced no candidate")
    params, rss = _refine(model, best_params, [0.3, 3e3, 0.3, 3e3], obs, budget)
    logK1, dH1, logK2, dH2 = params
    if logK2 > logK1:  # canonical site ordering
        logK1, dH1, logK2, dH2 = logK2, dH2, logK1, dH1
    return {
        "n1": 1.0,
        "logK1": logK1,
        "K1": math.exp(logK1),
        "dH1_J_mol": dH1,
        "n2": 1.0,
        "logK2": logK2,
        "K2": math.exp(logK2),
        "dH2_J_mol": dH2,
    }, rss


def _fit_dilution(volumes_L: np.ndarray, ligand_M: float, obs: np.ndarray, budget: _Budget) -> tuple[dict[str, float], float]:
    x = ligand_M * volumes_L
    denom = float(x @ x)
    if denom <= 0:
        raise SemanticUnresolved("dilution model has zero design norm")
    q = float(x @ obs) / denom
    budget.spend()
    rss = _rss(q * x, obs)
    return {"q_dil_J_per_mol": q}, rss


def _aicc(rss: float, n_points: int, k: int) -> float:
    if n_points <= k + 1:
        return math.inf
    return n_points * math.log(max(rss, 1e-300) / n_points) + 2 * k + (2 * k * (k + 1)) / (n_points - k - 1)


def fit_models(
    integration: dict[str, Any],
    concentration: dict[str, Any],
    cell_volume: dict[str, Any],
    *,
    budget: int = 4000,
) -> dict[str, Any]:
    """Fit all candidate models; keep every statistically equivalent one."""
    usable = [
        p
        for p in integration["per_injection"]
        if p.get("status") == "ok" and not p.get("excluded_from_fit") and p.get("net_heat_J") is not None
    ]
    diagnostics: list[dict[str, Any]] = []
    for p in integration["per_injection"]:
        if p.get("excluded_from_fit"):
            diagnostics.append({"reason": "FIRST_INJECTION_DISCARDED", "ordinal": p["ordinal"]})
        elif p.get("status") != "ok" or p.get("net_heat_J") is None:
            diagnostics.append({"reason": "INJECTION_UNUSABLE_FOR_FIT", "ordinal": p.get("ordinal")})

    conc_entries = [e for e in concentration["payload"]["entries"] if e.get("convertible")]
    if not conc_entries:
        raise SemanticUnresolved(
            "no convertible concentration entries",
            diagnostics=[{"reason": "CONCENTRATION_MISSING", "material": concentration["id"]}],
        )
    ligand_M = conc_entries[0].get("syringe_M")
    macro_M = conc_entries[0].get("macromolecule_M")
    if ligand_M is None or macro_M is None:
        raise SemanticUnresolved(
            "ligand and macromolecule concentrations are both required for binding models",
            diagnostics=[{"reason": "CONCENTRATION_INCOMPLETE", "material": concentration["id"]}],
        )
    cv = cell_volume["payload"].get("value")
    if cv is None:
        raise SemanticUnresolved(
            "cell volume is unconvertible",
            diagnostics=[{"reason": "CELL_VOLUME_MISSING", "material": cell_volume["id"]}],
        )
    cell_volume_L = cv["value"]

    if len(usable) < 3:
        raise SemanticUnresolved(
            "fewer than three usable injections; binding models are unidentifiable",
            diagnostics=[{"reason": "TOO_FEW_POINTS", "usable": len(usable)}],
        )

    volumes = np.array([p["volume_L"] for p in usable], dtype=float)
    obs = np.array([p["net_heat_J"] for p in usable], dtype=float)
    xt, mt = cumulative_composition(volumes, cell_volume_L, ligand_M, macro_M)

    meter = _Budget(budget)
    candidates: list[dict[str, Any]] = []

    params, rss = _fit_one_site(xt, mt, obs, cell_volume_L, meter)
    pred = heats_one_site(xt, mt, cell_volume_L, params["n"], params["K"], params["dH_J_mol"])
    candidates.append(
        {
            "model": "one_site",
            "params": params,
            "rss": rss,
            "aicc": _aicc(rss, len(obs), MODEL_PARAM_COUNTS["one_site"]),
            "predicted_J": pred.tolist(),
        }
    )
    try:
        params2, rss2 = _fit_two_site(xt, mt, obs, cell_volume_L, meter)
        pred2 = heats_two_site(
            xt, mt, cell_volume_L, 1.0, params2["K1"], params2["dH1_J_mol"], 1.0, params2["K2"], params2["dH2_J_mol"]
        )
        candidates.append(
            {
                "model": "two_site",
                "params": params2,
                "rss": rss2,
                "aicc": _aicc(rss2, len(obs), MODEL_PARAM_COUNTS["two_site"]),
                "predicted_J": pred2.tolist(),
            }
        )
    except SemanticUnresolved as exc:
        diagnostics.extend(exc.diagnostics)

    params3, rss3 = _fit_dilution(volumes, ligand_M, obs, meter)
    pred3 = heats_dilution(volumes, ligand_M, params3["q_dil_J_per_mol"])
    candidates.append(
        {
            "model": "dilution",
            "params": params3,
            "rss": rss3,
            "aicc": _aicc(rss3, len(obs), MODEL_PARAM_COUNTS["dilution"]),
            "predicted_J": pred3.tolist(),
        }
    )

    finite = [c for c in candidates if math.isfinite(c["aicc"])]
    if not finite:
        raise SemanticUnresolved(
            "no candidate model has a finite information criterion",
            diagnostics=[{"reason": "ALL_MODELS_UNIDENTIFIABLE"}],
        )
    best_aicc = min(c["aicc"] for c in finite)
    for c in candidates:
        c["equi_optimal"] = math.isfinite(c["aicc"]) and (c["aicc"] - best_aicc) <= EQUIV_AICC_DELTA
    candidates.sort(key=lambda c: (c["aicc"], c["model"]))

    residuals = []
    for i, p in enumerate(usable):
        residuals.append(
            {
                "ordinal": p["ordinal"],
                "observed_J": float(obs[i]),
                "per_model": {c["model"]: float(obs[i] - c["predicted_J"][i]) for c in candidates},
            }
        )

    result = {
        "kind": "fit",
        "inputs": {
            "integration_id": integration["id"],
            "concentration_material_id": concentration["id"],
            "cell_volume_material_id": cell_volume["id"],
            "rules_version": MODEL_RULES_VERSION,
            "budget": budget,
            "budget_used": meter.used,
        },
        "ligand_M": ligand_M,
        "macromolecule_M": macro_M,
        "cell_volume_L": cell_volume_L,
        "candidates": candidates,
        "residuals": residuals,
        "diagnostics": diagnostics,
        "unit": "J",
    }
    result["id"] = content_id("fit", result)
    return result
