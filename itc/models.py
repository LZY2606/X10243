import math

import numpy as np

from .errors import Diagnostic, SemanticUnresolvable


def _diag(severity, code, message, path, details=None):
    return Diagnostic(severity, code, message, path, details or {}).to_dict()


def ratio_series(schedule, concentration, cell_volume):
    ratios = []
    cumulative = 0.0
    ligand = concentration["ligand_syringe"]
    macro = concentration["macromolecule"]
    for injection in schedule["injections"]:
        cumulative += injection["volume"]
        ratios.append(ligand * cumulative / (macro * cell_volume))
    return ratios


def _fit_points(state, integration_version, overrides=None):
    schedule = state["materials"]["schedule"][-1]["injections"]
    concentration = state["materials"]["concentration"][-1]
    cell_volume = state["materials"]["cell"][-1]["cell_volume"]
    ratios = ratio_series(state["materials"]["schedule"][-1], concentration, cell_volume)
    points = []
    override_map = {row["injection_id"]: row for row in (overrides or [])}
    for injection, ratio, result in zip(schedule, ratios, integration_version["results"]):
        if not result.get("valid"):
            continue
        row = override_map.get(injection["id"])
        heat = row["net_heat"] if row and row.get("net_heat") is not None else result.get("net_heat")
        if heat is None:
            continue
        points.append({"injection_id": injection["id"], "ratio": ratio,
                       "heat": float(heat), "status": result["status"]})
    return points


def _sse(predicted, observed):
    return float(np.sum((np.asarray(predicted) - np.asarray(observed)) ** 2))


def _one_site(params, ratios):
    n, logk, dh = params
    k = np.exp(logk)
    x = np.asarray(ratios, dtype=float)
    macromolecule = 10.0
    ligand = x * macromolecule
    sites = n * macromolecule
    theta = 0.5 * ((sites + ligand + 1.0 / k) -
                   np.sqrt((sites + ligand + 1.0 / k) ** 2 - 4.0 * sites * ligand))
    theta_prev = np.concatenate(([0.0], theta[:-1]))
    return dh * (theta - theta_prev)


def fit_one_site(points):
    x = [p["ratio"] for p in points]
    y = np.array([p["heat"] for p in points], dtype=float)
    best = None
    for n in np.linspace(0.4, 12.0, 39):
        for logk in np.linspace(math.log(1e-4), math.log(1e5), 41):
            basis = []
            previous = 0.0
            for ratio in x:
                k = np.exp(logk)
                macro = 10.0
                ligand = ratio * macro
                sites = n * macro
                theta = 0.5 * ((sites + ligand + 1.0 / k) -
                               np.sqrt((sites + ligand + 1.0 / k) ** 2 - 4.0 * sites * ligand))
                basis.append(theta - previous)
                previous = theta
            basis = np.asarray(basis)
            denom = float(basis @ basis)
            if denom <= 1e-12:
                continue
            dh = float(basis @ y / denom)
            predicted = dh * basis
            score = _sse(predicted, y)
            if best is None or score < best[0]:
                best = (score, {"n": float(n), "K": float(np.exp(logk)), "ΔH": dh}, predicted)
    if best is None:
        raise SemanticUnresolvable("One-site model has no identifiable parameterization", "models.one_site")
    return best


def fit_dilution(points):
    x = np.array([p["ratio"] for p in points], dtype=float)
    y = np.array([p["heat"] for p in points], dtype=float)
    # Constant incremental dilution heat is the null/baseline hypothesis.
    mean = float(np.mean(y))
    predicted = np.full_like(y, mean)
    return _sse(predicted, y), {"dilution_heat_per_injection": mean}, predicted


def fit_multi_site(points):
    # Independent two-site approximation with two site-population amplitudes and
    # a shared K. A non-improved second amplitude gives the equal-fit edge case.
    x = np.array([p["ratio"] for p in points], dtype=float)
    y = np.array([p["heat"] for p in points], dtype=float)
    best = None
    for logk in np.linspace(math.log(1e-4), math.log(1e5), 41):
        k = np.exp(logk)
        strong, weak = [], []
        previous_s = previous_w = 0.0
        for ratio in x:
            root_s = 0.5 * ((ratio + 1 + 0.75 * k) - np.sqrt((ratio + 1 + 0.75 * k) ** 2 - 4 * ratio))
            root_w = 0.5 * ((ratio + 1 + 0.25 * k) - np.sqrt((ratio + 1 + 0.25 * k) ** 2 - 4 * ratio))
            strong.append(root_s - previous_s)
            weak.append(root_w - previous_w)
            previous_s, previous_w = root_s, root_w
        matrix = np.column_stack((strong, weak))
        amps, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        predicted = matrix @ amps
        score = _sse(predicted, y)
        if best is None or score < best[0]:
            best = (score, {"n1": 0.75, "n2": 0.25, "K": float(k),
                            "ΔH1": float(amps[0]), "ΔH2": float(amps[1])}, predicted)
    return best


def fit_models(state, integration_version, overrides=None):
    points = _fit_points(state, integration_version, overrides)
    diagnostics = []
    if len(points) < 3:
        raise SemanticUnresolvable(
            "At least three valid corrected injections are required for model comparison",
            "models.points", {"valid_points": len(points)})
    one = fit_one_site(points)
    dilution = fit_dilution(points)
    multi = fit_multi_site(points)
    raw = [
        ("one_site", "one independent binding site", 3, one),
        ("multi_site", "two independent site populations", 5, multi),
        ("dilution", "constant dilution-heat null model", 1, dilution),
    ]
    n = len(points)
    ranked = []
    best_sse = min(item[3][0] for item in raw)
    tolerance = max(1e-8, best_sse * 1e-3)
    for model_id, label, parameter_count, fitted in raw:
        sse, parameters, predicted = fitted
        aic = n * math.log(max(sse, 1e-12) / n) + 2 * parameter_count
        residuals = [float(v) for v in (np.asarray([p["heat"] for p in points]) - predicted)]
        ranked.append({
            "id": model_id,
            "label": label,
            "parameter_count": parameter_count,
            "sse": sse,
            "aic": aic,
            "parameters": parameters,
            "points": points,
            "predicted": [float(v) for v in predicted],
            "residuals": residuals,
            "equal_best": abs(sse - best_sse) <= tolerance,
            "units": {"ratio": "dimensionless molar ratio", "heat": "µcal",
                      "K": "µM^-1", "ΔH": "µcal per injection-equivalent"},
        })
    best_aic = min(item["aic"] for item in ranked)
    for item in ranked:
        item["delta_aic"] = item["aic"] - best_aic
    ranked.sort(key=lambda item: (item["delta_aic"], item["id"]))
    diagnostics.append(_diag(
        "warning" if sum(item["equal_best"] for item in ranked) > 1 else "info",
        "multiple_equivalent_models",
        "All legally parameterized candidates are retained; no client-side unique winner is invented",
        "models.comparison",
        {"equal_best_ids": [item["id"] for item in ranked if item["equal_best"]]}))
    return {
        "status": "candidate_set",
        "coordinate_basis": "cumulative ligand/macromolecule molar ratio vs corrected incremental heat",
        "candidates": ranked,
        "diagnostics": diagnostics,
    }
