"""Binding model candidates: one-site, two-site (multi-site), dilution only.

Heats per injection follow sequential mass balance. Multiple equally good
models are decided from AIC delta against a versioned threshold; every legal
candidate is retained, with deterministic tie ordering.
"""
from __future__ import annotations

import math

import numpy as np

from .models import Diagnostic, FitResult, ModelCandidate, stable_id
from .parser import concentration_view, conditions_view, injections_view

FIT_RULES = {
    "name": "binding_rules",
    "version": 1,
    "aic_equivalence_delta": 2.0,
    "n_grid_k": 25,
    "min_usable_points": 3,
}
R = 8.314462618  # J/mol/K


def _geometry(injections, cell_volume_l, syringe_m):
    vols = np.array([inj["volume_l"] for inj in injections], dtype=float)
    V = cell_volume_l + np.cumsum(vols)
    # concentration-equivalent of ligand introduced by each shot after dilution
    ligand_conc = np.cumsum(vols / V * syringe_m)
    return vols, V, ligand_conc


def _bound_one_site(lt, mt, k):
    b = mt + lt + 1.0 / k
    disc = np.sqrt(np.clip(b * b - 4.0 * lt * mt, 0.0, None))
    return 0.5 * (b - disc)


def _free_ligand_two_site(lt, mt, k1, k2):
    lt = np.atleast_1d(np.asarray(lt, dtype=float))
    out = np.zeros_like(lt)
    for i, x in enumerate(lt):
        lo, hi = 0.0, min(x, mt)
        for _ in range(80):
            f = 0.5 * (lo + hi)
            m_free = mt / (1.0 + k1 * f + k2 * f)
            if f + m_free * (k1 * f + k2 * f) < x:
                lo = f
            else:
                hi = f
        out[i] = 0.5 * (lo + hi)
    return out


def _aic(sse, n_data, k_params):
    if sse <= 0:
        sse = 1e-30
    return n_data * math.log(sse / n_data) + 2 * k_params


def _fit_one_site(ratios, V, mt, heats, temp_k):
    n_data = len(heats)
    best = None
    for lk in np.linspace(3.0, 8.0, FIT_RULES["n_grid_k"]):
        k = 10.0 ** lk
        bound = _bound_one_site(ratios, mt, k)
        dmoles = np.diff(np.concatenate(([0.0], bound * V)))
        denom = float(np.dot(dmoles, dmoles))
        if denom <= 0:
            continue
        dh = float(np.dot(dmoles, heats) / denom)
        sse = float(np.sum((dh * dmoles - heats) ** 2))
        if best is None or sse < best[0]:
            best = (sse, k, dh, dmoles)
    sse, k, dh, dmoles = best
    pred = dh * dmoles
    entropy = -dh / temp_k + R * math.log(k)
    return {"params": {"K_M-1": k, "delta_H_J/mol": dh, "delta_S_J/mol/K": entropy},
            "param_units": {"K_M-1": "M^-1", "delta_H_J/mol": "J/mol",
                            "delta_S_J/mol/K": "J/mol/K"},
            "sse": sse, "aic": _aic(sse, n_data, 2), "pred": pred}


def _fit_two_site(ratios, V, mt, heats):
    n_data = len(heats)
    logs = np.linspace(3.0, 8.0, FIT_RULES["n_grid_k"])
    best = None
    for l1 in logs:
        for l2 in logs:
            if l2 >= l1:
                continue
            k1, k2 = 10.0 ** l1, 10.0 ** l2
            f = _free_ligand_two_site(ratios, mt, k1, k2)
            m_free = mt / (1.0 + k1 * f + k2 * f)
            x1 = np.diff(np.concatenate(([0.0], m_free * k1 * f * V)))
            x2 = np.diff(np.concatenate(([0.0], m_free * k2 * f * V)))
            coef, *_ = np.linalg.lstsq(np.column_stack((x1, x2)), heats, rcond=None)
            dh1, dh2 = float(coef[0]), float(coef[1])
            pred = dh1 * x1 + dh2 * x2
            sse = float(np.sum((pred - heats) ** 2))
            if best is None or sse < best[0]:
                best = (sse, k1, k2, dh1, dh2, pred)
    sse, k1, k2, dh1, dh2, pred = best
    return {"params": {"K1_M-1": k1, "K2_M-1": k2,
                       "delta_H1_J/mol": dh1, "delta_H2_J/mol": dh2},
            "param_units": {"K1_M-1": "M^-1", "K2_M-1": "M^-1",
                            "delta_H1_J/mol": "J/mol", "delta_H2_J/mol": "J/mol"},
            "sse": sse, "aic": _aic(sse, n_data, 4), "pred": pred}


def _fit_dilution(vols, heats):
    n_data = len(heats)
    denom = float(np.dot(vols, vols))
    slope = float(np.dot(vols, heats) / denom) if denom > 0 else 0.0
    pred = slope * vols
    sse = float(np.sum((pred - heats) ** 2))
    return {"params": {"dilution_heat_J_per_L": slope},
            "param_units": {"dilution_heat_J_per_L": "J/L"},
            "sse": sse, "aic": _aic(sse, n_data, 1), "pred": pred}


def fit_models(integration, injections_material, concentration_material,
               conditions_material, blank_material=None, rule="none",
               used_indices=None) -> FitResult:
    conc = concentration_view(concentration_material)
    cond = conditions_view(conditions_material)
    injections = injections_view(injections_material)
    vols, V, ratios = _geometry(injections, cond["cell_volume_l"], conc["syringe_m"])
    mt = conc["cell_m"]

    rows = {p["injection_index"]: p for p in integration.points}
    if used_indices is None:
        used_indices = [idx for idx, p in rows.items()
                        if p["status"] not in ("dropped", "blocked_by_gap",
                                               "endpoint_truncated")
                        and p.get("net_heat_j") is not None]
    sel = sorted(i for i in used_indices if i in rows and rows[i].get("net_heat_j") is not None)
    diagnostics: list[dict] = []
    if len(sel) < FIT_RULES["min_usable_points"]:
        diagnostics.append(Diagnostic(
            "insufficient_usable_points",
            f"{len(sel)} usable injections (< {FIT_RULES['min_usable_points']}); "
            "no binding model can be decided",
            {"usable": sel}, "semantic_unresolvable").to_dict())
        input_summary = {"integration": integration.result_id,
                         "concentration_material": concentration_material.material_id,
                         "conditions_material": conditions_material.material_id,
                         "blank_material": blank_material.material_id if blank_material else None,
                         "blank_rule": rule, "usable_indices": sel}
        return FitResult(
            result_id=stable_id("fit", [input_summary, FIT_RULES, "empty"]),
            integration_id=integration.result_id,
            blank_id=blank_material.material_id if blank_material else None,
            concentration_version=concentration_material.version,
            conditions_version=conditions_material.version,
            input_summary=input_summary, param_version=FIT_RULES["version"],
            parents=[integration.result_id], candidates=[], diagnostics=diagnostics)

    pos = np.array([i - 1 for i in sel], dtype=int)  # injection_index is 1-based
    r, Vs, vv = ratios[pos], V[pos], vols[pos]
    heats = np.array([rows[i]["net_heat_j"] for i in sel], dtype=float)

    fits = [_fit_one_site(r, Vs, mt, heats, cond["temperature_k"]),
            _fit_two_site(r, Vs, mt, heats),
            _fit_dilution(vv, heats)]
    candidates = []
    for model, rfit in zip(("one_site", "two_site", "dilution"), fits):
        candidates.append(ModelCandidate(
            model=model,
            params={k: float(v) for k, v in rfit["params"].items()},
            param_units=rfit["param_units"], sse=rfit["sse"], aic=rfit["aic"],
            equivalent_group=0, rank=0, usable_points=len(sel),
            diagnostics=[{"residuals": [float(x) for x in rfit["pred"] - heats],
                          "indices": sel}]))
    candidates.sort(key=lambda c: (round(c.aic, 9), c.model))
    # equivalence groups walk the AIC-sorted list against the current group min
    group, group_min, seen = 0, candidates[0].aic, []
    for c in candidates:
        if c.aic - group_min > FIT_RULES["aic_equivalence_delta"]:
            group += 1
            group_min = c.aic
        c.equivalent_group = group
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank
    input_summary = {
        "integration": integration.result_id,
        "concentration_material": concentration_material.material_id,
        "conditions_material": conditions_material.material_id,
        "blank_material": blank_material.material_id if blank_material else None,
        "blank_rule": rule,
        "usable_indices": sel,
        "cell_m": mt, "syringe_m": conc["syringe_m"],
        "temperature_k": cond["temperature_k"], "cell_volume_l": cond["cell_volume_l"],
    }
    parents = [integration.result_id, concentration_material.material_id,
               conditions_material.material_id]
    if blank_material:
        parents.append(blank_material.material_id)
    fit_id = stable_id("fit", [input_summary, FIT_RULES,
                               [(c.model, round(c.aic, 9), round(c.sse, 15),
                                 {k: round(v, 12) for k, v in c.params.items()})
                                for c in candidates]])
    return FitResult(result_id=fit_id, integration_id=integration.result_id,
                     blank_id=blank_material.material_id if blank_material else None,
                     concentration_version=concentration_material.version,
                     conditions_version=conditions_material.version,
                     input_summary=input_summary, param_version=FIT_RULES["version"],
                     parents=parents, candidates=[c.to_dict() for c in candidates],
                     diagnostics=diagnostics)
