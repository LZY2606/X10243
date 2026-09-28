from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import RULES_VERSION


def _ratios(injections: list[dict], integrations: list[dict], concentration_version: dict, cell_volume_uL: float):
    by_id = {item["injection_id"]: item for item in integrations}
    xs, ys, ids, volumes = [], [], [], []
    syringe = concentration_version["syringe_concentration"]["canonical_value"]
    macromolecule = concentration_version["macromolecule_concentration"]["canonical_value"]
    cumulative = 0.0
    for idx, injection in enumerate(injections):
        cumulative += by_id[injection["id"]]["injection_volume_uL"]
        ratio = cumulative * syringe / (cell_volume_uL * macromolecule) if macromolecule else math.nan
        item = by_id[injection["id"]]
        if item["status"] == "ok" and item["net_heat_uJ"] is not None and math.isfinite(ratio):
            xs.append(ratio)
            ys.append(item["net_heat_uJ"])
            ids.append(injection["id"])
            volumes.append(item["injection_volume_uL"])
    return np.asarray(xs), np.asarray(ys), ids, volumes


def _linear_design(x, basis):
    if basis == "constant":
        return np.column_stack([np.ones_like(x)])
    if basis == "one_site_zero":
        return np.column_stack([1.0 / (1.0 + x)])
    if basis == "one_site":
        return np.column_stack([1.0 / (1.0 + x), np.ones_like(x)])
    if basis == "two_site":
        return np.column_stack([1.0 / (1.0 + x), 1.0 / (1.0 + 2.0 * x), np.ones_like(x)])
    raise ValueError(basis)


def _linear_fit(x, y, basis):
    design = _linear_design(x, basis)
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeff
    sse = float(np.sum((y - pred) ** 2))
    return coeff, pred, sse


def _parameter_summary(basis, coeff, sse, x, n):
    a = float(coeff[0]) if len(coeff) > 0 else 0.0
    d = float(coeff[-1]) if "constant" not in basis and len(coeff) > 1 else (float(coeff[0]) if basis == "constant" else 0.0)
    params = {"dilution_heat_per_injection_uJ": d}
    if basis in ("one_site", "one_site_zero"):
        params.update({"binding_K_M": 0.0, "binding_enthalpy_uJ_per_injection_at_zero_ratio": a, "binding_sites": 1})
    if basis == "two_site":
        b = float(coeff[1])
        params.update({"site_1_K_M": 0.0, "site_2_K_M": 0.0, "site_1_amplitude_uJ": a, "site_2_amplitude_uJ": b, "binding_sites": 2})
    if basis == "constant":
        params = {"dilution_heat_per_injection_uJ": a}
    k = 2 if basis == "constant" else (3 if basis == "two_site" else (3 if basis == "one_site" else 2))
    aic = n * math.log(max(sse, 1e-12) / n) + 2 * k if n else math.inf
    return params, aic


def fit_models(injections, integrations, concentration_version, cell_volume_q, tie_tolerance=1e-9):
    cell_volume_uL = cell_volume_q["canonical_value"]
    x, y, ids, volumes = _ratios(injections, integrations, concentration_version, cell_volume_uL)
    if len(x) < 2:
        return {
            "status": "blocked",
            "reason": "insufficient_valid_injections",
            "valid_injection_ids": ids,
            "models": [],
            "rules_version": RULES_VERSION,
        }
    candidates = []
    basis_by_model = {
        "one_site_K_zero": "one_site",
        "two_site_K_zero": "two_site",
        "dilution_only": "constant",
    }
    for model_id, basis in basis_by_model.items():
        coeff, pred, sse = _linear_fit(x, y, basis)
        params, aic = _parameter_summary(basis, coeff, sse, x, len(x))
        candidates.append({
            "model_id": model_id,
            "hypothesis": "one_site" if model_id.startswith("one") else ("multi_site" if model_id.startswith("two") else "dilution"),
            "status": "legal_candidate",
            "sse_uJ2": sse,
            "aic": aic,
            "parameters": params,
            "predicted_by_injection": {inject_id: float(value) for inject_id, value in zip(ids, pred)},
            "residual_by_injection": {inject_id: float(value) for inject_id, value in zip(ids, y - pred)},
            "molar_ratio_by_injection": {inject_id: float(value) for inject_id, value in zip(ids, x)},
        })
    best_sse = min(item["sse_uJ2"] for item in candidates)
    optimal = [item for item in candidates if abs(item["sse_uJ2"] - best_sse) <= tie_tolerance]
    for item in candidates:
        item["equally_optimal"] = item in optimal
        item["selection_basis"] = f"normalized SSE within {tie_tolerance:g} uJ^2; mechanistically different candidates are retained"
    ambiguity = len(optimal) > 1
    return {
        "status": "ambiguous" if ambiguity else "ok",
        "reason": "multiple_equally_optimal_models" if ambiguity else "unique_best_sse",
        "valid_injection_ids": ids,
        "excluded_injections": [inj["id"] for inj in injections if inj["id"] not in ids],
        "models": candidates,
        "optimal_model_ids": [item["model_id"] for item in optimal],
        "concentration_material_id": concentration_version["material_id"],
        "concentration_version": concentration_version["version"],
        "rules_version": RULES_VERSION,
    }
