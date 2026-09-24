"""Binding-model candidates compared per injection.

Models (predict per-injection heat in cal/mol of injectant):

* ``one_site``          — 1:1 mass action, exact tight-binding root
* ``two_site``          — two independent sites with K1, K2, dH1, dH2
* ``dilution_const``    — no binding: constant dilution heat
* ``dilution_linear``   — no binding: heat linear in injection volume

Equally good models are all retained (BIC within ``tie_delta``), never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .optimize import BudgetExhausted, least_squares
from .units import QuantityError

MODEL_KIND = "binding_model"
TIE_BIC = 2.0
EQUIVALENT_BIC = 4.0

# Canonical unit conversions: heat uJ, volume uL, concentration mM.
_UJ_PER_CAL = 4.184e6
_UL_IN_L = 1e6


@dataclass
class Titration:
    volumes_uL: np.ndarray
    heats_uJ: np.ndarray
    ligand_mM: float
    macromolecule_mM: float
    cell_volume_uL: float
    temperature_K: float

    @property
    def mole_per_injection(self) -> np.ndarray:
        # mmol/L * uL = 1e-6 mmol = 1e-9 mol -> umol = 1e-3
        return self.ligand_mM * self.volumes_uL * 1e-3

    def cumulative_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-injection cumulative ligand X (mM), macromolecule M (mM), V (uL)."""
        dv = self.volumes_uL
        v0 = self.cell_volume_uL
        vol = v0 + np.concatenate([[0.0], np.cumsum(dv)[:-1]])
        x_lig = self.ligand_mM * np.cumsum(dv) / vol
        m_mac = self.macromolecule_mM * v0 / vol
        return x_lig, m_mac, vol


def _free_ligand_1to1(x_total: float, m_total: float, k: float) -> float:
    """Exact root of [L]: [L]^2 + (M + 1/K - X)[L] - X/K = 0, with a budget."""
    if k <= 0:
        raise QuantityError(
            "semantic-impossible",
            "model.one_site.K",
            "association constant must be positive",
        )
    inv = 1.0 / k
    b = m_total + inv - x_total
    disc = b * b + 4.0 * x_total * inv
    return 0.5 * (np.sqrt(max(disc, 0.0)) - b)


def _free_two_sites(x_total: float, m_total: float, k1: float, k2: float) -> float:
    """Bisection for two independent sites; bounded iteration budget."""
    lo, hi = 0.0, max(x_total, 1e-30)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        bound = m_total * (k1 * mid / (1 + k1 * mid) + k2 * mid / (1 + k2 * mid))
        if mid + bound - x_total > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * max(1.0, x_total):
            break
    return 0.5 * (lo + hi)


def _heat_per_mole(q_uJ_per_inj: np.ndarray, titr: Titration) -> np.ndarray:
    # cal/mol = uJ / umol / 4.184
    return q_uJ_per_inj / titr.mole_per_injection / _UJ_PER_CAL


def predict_one_site(theta: np.ndarray, titr: Titration) -> np.ndarray:
    k, dh, _ = theta  # K in /mM, dH in cal/mol; third slot unused offset=0
    x_cum, m_cum, vol = titr.cumulative_state()
    bound = np.array([m * k * _free_ligand_1to1(x, m, k) / (1 + k * _free_ligand_1to1(x, m, k))
                      for x, m in zip(x_cum, m_cum)])
    q_total = dh * _UJ_PER_CAL * bound * titr.macromolecule_mM * titr.cell_volume_uL / _UL_IN_L
    # displacement correction
    v_before = vol
    q_before = np.concatenate([[0.0], q_total[:-1]])
    fraction = v_before / (v_before + titr.volumes_uL)
    q_inj = q_total - q_before * fraction
    return _heat_per_mole(q_inj, titr)


def predict_two_site(theta: np.ndarray, titr: Titration) -> np.ndarray:
    k1, k2, dh1, dh2 = theta
    x_cum, m_cum, vol = titr.cumulative_state()
    q_total = np.zeros_like(x_cum)
    for i, (x, m) in enumerate(zip(x_cum, m_cum)):
        f = _free_two_sites(x, m, k1, k2)
        frac1 = k1 * f / (1 + k1 * f)
        frac2 = k2 * f / (1 + k2 * f)
        q_total[i] = (dh1 * frac1 + dh2 * frac2) * _UJ_PER_CAL * (
            m * titr.cell_volume_uL / _UL_IN_L
        )
    v_before = vol
    q_before = np.concatenate([[0.0], q_total[:-1]])
    fraction = v_before / (v_before + titr.volumes_uL)
    q_inj = q_total - q_before * fraction
    return _heat_per_mole(q_inj, titr)


def _linear_design(titr: Titration) -> np.ndarray:
    return np.column_stack([np.ones_like(titr.volumes_uL), titr.volumes_uL])


def _linear_fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    rss = float(np.sum((y - design @ coef) ** 2))
    return coef, rss


@dataclass
class ModelSpec:
    model_id: str
    family: str
    description: str
    predict: Callable
    n_params: int
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    starts: list[np.ndarray] | None = None


def build_titration(heat_set: dict[str, Any], concentrations: dict[str, Any],
                    pool: dict[str, Any], temperature_K: float,
                    *, include_open: bool = False) -> tuple[Titrration_placeholder()] if False else Titration:  # noqa
    entries = heat_set["entries"]
    usable = [
        e for e in entries
        if e["status"] == "integrated" and e["corrected_heat_uJ"] is not None
    ]
    if not include_open:
        usable = [e for e in usable if "open_endpoint" not in e["flags"]]
    if len(usable) < 2:
        raise QuantityError(
            "semantic-impossible",
            "fit.entries",
            "need at least two integrated, non-open injections to fit a model",
            raw=len(usable),
        )
    volumes = np.array([float(e.get("volume_uJ_placeholder", e.get("_volume_uL"))) for e in usable]) \
        if False else np.array([_entry_volume(heat_set, e) for e in usable], dtype=float)
    heats = np.array([float(e["corrected_heat_uJ"]) for e in usable])
    return Titration(
        volumes_uL=volumes,
        heats_uJ=heats,
        ligand_mM=float(concentrations["ligand_conc_mM"]),
        macromolecule_mM=float(concentrations["macromolecule_conc_mM"]),
        cell_volume_uL=float(pool["cell_volume_uL"]),
        temperature_K=float(temperature_K),
    )


def Titration_placeholder():  # kept out of runtime
    raise NotImplementedError


def _entry_volume(heat_set: dict[str, Any], entry: dict[str, Any]) -> float:
    volumes = heat_set.get("_volumes_uL")
    if volumes is None:
        raise QuantityError(
            "corrupt-input", "fit.entries.volume", "injection volumes missing on heat set"
        )
    return float(volumes[entry["ordinal"]])


def _observed(titr: Titration) -> np.ndarray:
    return _heat_per_mole(titr.heats_uJ, titr)


def _specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "one_site", "binding", "single independent sites (1:1 mass action)",
            predict_one_site, 2,
            bounds=(np.array([1e-3, -1.0e5]), np.array([1e6, 1.0e5])),
            starts=[
                np.array([1.0, -5000.0]),
                np.array([100.0, -5000.0]),
                np.array([1.0e4, -5000.0]),
            ],
        ),
        ModelSpec(
            "two_site", "binding", "two independent site classes",
            predict_two_site, 4,
            bounds=(np.array([1e-3, 1e-3, -1.0e5, -1.0e5]),
                    np.array([1e6, 1e6, 1.0e5, 1.0e5])),
            starts=[
                np.array([1.0, 100.0, -3000.0, -8000.0]),
                np.array([10.0, 1000.0, -2000.0, -6000.0]),
            ],
        ),
        ModelSpec("dilution_const", "dilution", "constant dilution heat, no binding",
                  None, 1),
        ModelSpec("dilution_linear", "dilution", "heat linear in injection volume, no binding",
                  None, 2),
    ]


def fit_models(heat_set: dict[str, Any], concentrations: dict[str, Any], pool: dict[str, Any],
               temperature_K: float, *, max_iter: int = 300, max_evals: int = 6000) -> dict[str, Any]:
    """Fit every candidate; deterministic, no random starts."""
    titr = build_titration(heat_set, concentrations, pool, temperature_K)
    y = _observed(titr)
    n = len(y)
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def bic_of(rss: float, k: int) -> float:
        rss = max(rss, 1e-30)
        return float(n * np.log(rss / n) + k * np.log(n))

    for spec in _specs():
        try:
            if spec.model_id == "dilution_const":
                design = np.ones((n, 1))
                coef, rss = _linear_fit(design, y)
                params = {"const_cal_per_mol": float(coef[0])}
                yhat = design @ coef
            elif spec.model_id == "dilution_linear":
                design = _linear_design(titr)
                coef, rss = _linear_fit(design, y)
                params = {"intercept_cal_per_mol": float(coef[0]),
                          "slope_cal_per_mol_per_uL": float(coef[1])}
                yhat = design @ coef
            else:
                best = None
                budget_fail = None
                for x0 in spec.starts:
                    try:
                        fr = least_squares(
                            lambda th, s=spec: s.predict(th, titr) - y,
                            x0, spec.bounds, max_iter=max_iter, max_evals=max_evals,
                        )
                    except BudgetExhausted as exc:
                        budget_fail = exc
                        continue
                    if best is None or fr.rss < best.rss:
                        best = fr
                if best is None:
                    raise budget_fail or BudgetExhausted(
                        "no start evaluated", evaluations=0, iterations=0
                    )
                theta = best.params
                rss = float(best.rss)
                yhat = spec.predict(theta, titr)
                if spec.model_id == "one_site":
                    params = {"K_per_mM": float(theta[0]), "dH_cal_per_mol": float(theta[1])}
                else:
                    params = {"K1_per_mM": float(theta[0]), "K2_per_mM": float(theta[1]),
                              "dH1_cal_per_mol": float(theta[2]), "dH2_cal_per_mol": float(theta[3])}
        except QuantityError as exc:
            failures.append({"model_id": spec.model_id, "outcome": "semantic_unresolvable",
                             "diagnostic": exc.diagnostic})
            continue
        except BudgetExhausted as exc:
            failures.append({"model_id": spec.model_id, "outcome": "budget_exhausted",
                             "diagnostic": {
                                 "code": "fit-budget-exhausted",
                                 "where": f"model.{spec.model_id}",
                                 "message": str(exc),
                                 "evaluations": exc.evaluations,
                                 "iterations": exc.iterations,
                             }})
            continue
        bic = bic_of(rss, spec.n_params)
        candidates.append({
            "model_id": spec.model_id,
            "family": spec.family,
            "description": spec.description,
            "params": params,
            "rss_uJ2": None,
            "rss_cal2_per_mol2": float(rss),
            "bic": bic,
            "n_points": n,
            "n_params": spec.n_params,
            "predicted_cal_per_mol": [float(v) for v in yhat],
            "observed_cal_per_mol": [float(v) for v in y],
            "residuals_cal_per_mol": [float(v) for v in (yhat - y)],
            "injection_ids": [e["injection_id"] for e in
                              [e2 for e2 in heat_set["entries"] if e2["status"] == "integrated"
                               and "open_endpoint" not in e2["flags"]]],
        })

    if not candidates:
        return {
            "kind": "model_fit",
            "status": "failed",
            "candidates": [],
            "failures": failures,
        }

    best_bic = min(c["bic"] for c in candidates)
    rank = 0
    for c in sorted(candidates, key=lambda z: (z["bic"], z["model_id"])):
        c["bic_delta"] = c["bic"] - best_bic
        if c["bic_delta"] <= TIE_BIC:
            c["equivalence"] = "co_optimal"
        elif c["bic_delta"] <= EQUIVALENT_BIC:
            c["equivalence"] = "equivalent_within_evidence"
        else:
            c["equivalence"] = "worse"
        c["bic_rank"] = rank
        rank += 1
    co = [c for c in candidates if c["equivalence"] == "co_optimal"]
    status = "multi_co_optimal" if len(co) > 1 else "fitted"
    return {
        "kind": "model_fit",
        "status": status,
        "co_optimal_model_ids": sorted(c["model_id"] for c in co),
        "candidates": candidates,
        "failures": failures,
    }
