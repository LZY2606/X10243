"""Binding-model candidates fitted to per-injection integrated heats.

Models compared by the review page:
  * ``one_site``  - Wiseman/MicroCal single set of identical sites
  * ``two_site``  - two independent site populations
  * ``dilution``  - constant dilution heat (K fixed to a negligible value)

All candidates are retained.  "Equally good" is deterministic: every model
within ``TIE_AICC_DELTA`` of the minimum AICc is flagged ``tied`` and every
legal candidate is returned -- no unique winner is fabricated.

Fits depend on the concentration + cell-volume versions but *not* on the raw
power samples: a concentration revision makes the fit stale and must be
redone, while the original integration stays valid and untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .errors import Diagnostic, SemanticUnresolvable, diag

MODEL_RULE_VERSION = "binding-v1.0"
TIE_AICC_DELTA = 2.0
R_GAS = 8.314462618                       # J mol^-1 K^-1
DILUTION_K = 1.0                          # m^3 mol^-1 => 1 kM^-1 equivalent


@dataclass
class BindingInput:
    heats_joules: list[float]             # per accepted injection, in order
    injection_volumes_m3: list[float]
    ligand_conc_syringe_molm3: float      # mol/m^3 == mmol/L numeric scale
    macromolecule_conc_cell_molm3: float
    cell_volume_m3: float
    temperature_k: float
    injection_indices: list[int]
    sign_convention: str = "heat_released_positive"

    def to_summary(self) -> dict[str, Any]:
        return {
            "n_points": len(self.heats_joules),
            "cell_volume_m3": self.cell_volume_m3,
            "temperature_k": self.temperature_k,
            "ligand_syringe_mol_per_l": self.ligand_conc_syringe_molm3 / 1000.0,
            "macromolecule_mol_per_l": self.macromolecule_conc_cell_molm3 / 1000.0,
            "injection_indices": list(self.injection_indices),
            "sign_convention": self.sign_convention,
        }


def _concentrations(bi: BindingInput) -> tuple[np.ndarray, np.ndarray]:
    """Total ligand in cell after each injection (MicroCal recursion)."""
    x = np.zeros(len(bi.heats_joules))
    v0 = bi.cell_volume_m3
    total = 0.0
    vol = v0
    for i, dvi in enumerate(bi.injection_volumes_m3):
        total = total + dvi * bi.ligand_conc_syringe_molm3
        vol = vol + dvi
        x[i] = total / vol
    macromol = (bi.macromolecule_conc_cell_molm3 * v0) / vol
    return x, np.full_like(x, macromol)


def _q_one_site(xt: np.ndarray, mt: np.ndarray, k: float, n: float,
                dh: float, v0: float) -> np.ndarray:
    """Total cell heat Q_i (J) for a single set of sites."""
    a = n * mt + xt + 1.0 / k
    theta = (a - np.sqrt(np.maximum(a * a - 4.0 * n * mt * xt, 0.0))) / (2.0 * n)
    return v0 * mt * dh * theta


def incremental_heats(q: np.ndarray, volumes: np.ndarray, v0: float) -> np.ndarray:
    """MicroCal dQ_i correcting for displaced volume."""
    dq = np.empty_like(q)
    dq[0] = q[0]
    dq[1:] = q[1:] - q[:-1] + (volumes[1:] / v0) * (q[1:] + q[:-1]) / 2.0
    return dq


def predict_one_site(bi: BindingInput, k: float, n: float, dh: float):
    xt, mt = _concentrations(bi)
    q = _q_one_site(xt, mt, k, n, dh, bi.cell_volume_m3)
    return incremental_heats(q, np.asarray(bi.injection_volumes_m3),
                             bi.cell_volume_m3)


def _q_two_site(xt, mt, k1, k2, dh1, dh2, v0):
    a1 = mt + xt + 1.0 / k1
    th1 = (a1 - np.sqrt(np.maximum(a1 * a1 - 4.0 * mt * xt, 0.0))) / 2.0
    a2 = mt + xt + 1.0 / k2
    th2 = (a2 - np.sqrt(np.maximum(a2 * a2 - 4.0 * mt * xt, 0.0))) / 2.0
    return v0 * mt * (dh1 * th1 + dh2 * th2)


def predict_two_site(bi: BindingInput, k1, k2, dh1, dh2):
    xt, mt = _concentrations(bi)
    q = _q_two_site(xt, mt, k1, k2, dh1, dh2, bi.cell_volume_m3)
    return incremental_heats(q, np.asarray(bi.injection_volumes_m3),
                             bi.cell_volume_m3)


def _grid(n: int, lo: float, hi: float) -> np.ndarray:
    return np.geomspace(lo, hi, n)


def _linear_dh(y, basis):
    """Non-negative? No - least squares dh; allow exothermic/endothermic."""
    dh, *_ = np.linalg.lstsq(basis.reshape(-1, 1), y, rcond=None)
    pred = basis * dh[0]
    return float(dh[0]), pred


def _aicc(rss: float, n: int, k_params: int) -> float:
    if rss <= 0 or not math.isfinite(rss):
        rss = 1e-30
    aic = n * math.log(rss / n) + 2 * k_params
    denom = n - k_params - 1
    if denom > 0:
        aic += 2 * k_params * (k_params + 1) / denom
    return aic


def fit_one_site(bi: BindingInput, *, grid_n: int = 96, refine: int = 41):
    y = np.asarray(bi.heats_joules)
    best = None
    k_grid = _grid(grid_n, 1e3, 1e9)          # m^3 mol^-1 = 1e3(M^-1)...1e9
    n_grid = np.linspace(0.5, 2.0, 16)
    for k in k_grid:
        for n in n_grid:
            basis = predict_one_site(bi, float(k), float(n), 1.0)
            dh, pred = _linear_dh(y, basis)
            rss = float(np.sum((y - pred) ** 2))
            if best is None or rss < best[0]:
                best = (rss, k, n, dh, pred)
    rss, k0, n0, _, _ = best
    for k in _grid(refine, max(k0 / 3, 1e3), min(k0 * 3, 1e9)):
        for n in np.linspace(max(0.5, n0 - 0.12), min(2.0, n0 + 0.12), 13):
            basis = predict_one_site(bi, float(k), float(n), 1.0)
            dh, pred = _linear_dh(y, basis)
            r = float(np.sum((y - pred) ** 2))
            if r < rss:
                rss, k0, n0, best_pred = r, k, n, pred
                best = (r, k, n, dh, pred)
    rss, k, n, dh, pred = best
    return dict(k=k, n=n, dh=dh, pred=pred, rss=rss)


def fit_two_site(bi: BindingInput, *, grid_n: int = 60, refine: int = 33):
    y = np.asarray(bi.heats_joules)
    best = None
    g = _grid(grid_n, 1e3, 1e9)

    def evaluate(k1, k2):
        b1 = predict_two_site(bi, k1, k2, 1.0, 0.0)
        b2 = predict_two_site(bi, k1, k2, 0.0, 1.0)
        coef, *_ = np.linalg.lstsq(np.column_stack([b1, b2]), y, rcond=None)
        pred = coef[0] * b1 + coef[1] * b2
        return float(np.sum((y - pred) ** 2)), float(coef[0]), float(coef[1]), pred

    for k1 in g:
        for k2 in g:
            rss, dh1, dh2, pred = evaluate(float(k1), float(k2))
            if best is None or rss < best[0]:
                best = (rss, float(k1), float(k2), dh1, dh2, pred)
    rss, k10, k20, *_ = best
    for k1 in _grid(refine, max(k10 / 2.5, 1e3), min(k10 * 2.5, 1e9)):
        for k2 in _grid(refine, max(k20 / 2.5, 1e3), min(k20 * 2.5, 1e9)):
            r, d1, d2, pred = evaluate(float(k1), float(k2))
            if r < rss:
                rss, k10, k20 = r, float(k1), float(k2)
                best = (r, float(k1), float(k2), d1, d2, pred)
    rss, k1, k2, dh1, dh2, pred = best
    return dict(k1=k1, k2=k2, dh1=dh1, dh2=dh2, pred=pred, rss=rss)


def fit_dilution(bi: BindingInput):
    y = np.asarray(bi.heats_joules)
    dh = float(np.mean(y))
    pred = np.full_like(y, dh)
    rss = float(np.sum((y - pred) ** 2))
    return dict(dh=dh, k=DILUTION_K, pred=pred, rss=rss)


def fit_models(bi: BindingInput) -> dict[str, Any]:
    n = len(bi.heats_joules)
    if n < 3:
        raise SemanticUnresolvable(
            f"need >= 3 integrated injections for model comparison, got {n}",
            [diag("too_few_injections", "model comparison is underdetermined",
                  f"n={n}")])
    y = np.asarray(bi.heats_joules)
    if not np.isfinite(y).all():
        bad = [i for i, v in enumerate(bi.injection_indices)
               if not math.isfinite(bi.heats_joules[i])]
        raise SemanticUnresolvable(
            "non-finite heats in model input",
            [diag("non_finite_heat", "rejected/gap injections must be excluded",
                  f"injection positions {bad}")])

    candidates_raw = []
    one = fit_one_site(bi)
    candidates_raw.append(("one_site", 3, one,
                           {"K": one["k"], "n": one["n"], "dH": one["dh"]},
                           {"K": "m^3 mol^-1", "n": "sites",
                            "dH": "J mol^-1"}))
    dil = fit_dilution(bi)
    candidates_raw.append(("dilution", 1, dil, {"dH": dil["dh"]},
                           {"dH": "J per injection"}))
    two = fit_two_site(bi)
    candidates_raw.append(("two_site", 4, two,
                           {"K1": two["k1"], "K2": two["k2"],
                            "dH1": two["dh1"], "dH2": two["dh2"]},
                           {"K1": "m^3 mol^-1", "K2": "m^3 mol^-1",
                            "dH1": "J mol^-1", "dH2": "J mol^-1"}))

    scored = []
    for name, kpar, res, params, units in candidates_raw:
        aicc = _aicc(res["rss"], n, kpar)
        scored.append((name, kpar, res, params, units, aicc))
    best_aicc = min(s[5] for s in scored)
    candidates = []
    winners = [s[0] for s in scored if s[5] <= best_aicc + TIE_AICC_DELTA]
    for name, kpar, res, params, units, aicc in scored:
        residuals = (y - res["pred"]).tolist()
        candidates.append({
            "model": name,
            "aic": aicc - (2 * kpar * (kpar + 1) / max(n - kpar - 1, 1)
                           if n - kpar - 1 > 0 else 0),
            "aicc": aicc,
            "n_params": kpar,
            "params": params,
            "param_units": units,
            "predicted": res["pred"].tolist(),
            "residuals": residuals,
            "tied": name in winners,
            "diagnostics": [],
        })
    order = {"one_site": 0, "two_site": 1, "dilution": 2}
    candidates.sort(key=lambda c: (0 if c["tied"] else 1, order[c["model"]],
                                   c["aicc"]))
    ambiguous = len(winners) > 1
    return {
        "rule_version": MODEL_RULE_VERSION,
        "tie_delta_aicc": TIE_AICC_DELTA,
        "candidates": candidates,
        "winners": sorted(winners, key=lambda m: order[m]),
        "ambiguous": ambiguous,
        "inputs": bi.to_summary(),
    }


def delta_g(k_m3_per_mol: float, temperature_k: float) -> float:
    """dG = -RT ln K with K converted to dimensionless standard state."""
    k_dimless = k_m3_per_mol * 1000.0        # per M reference
    return -R_GAS * temperature_k * math.log(max(k_dimless, 1e-300))
