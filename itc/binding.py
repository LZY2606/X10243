"""Binding-model candidates fitted to per-injection heats.

Implemented with NumPy only (analytic site model + a small Levenberg-Marquardt
fitter with multi-start). Candidates compared here:
* one_site      - single set of identical sites (n, K, dH)
* two_site      - two independent site populations (n1,K1,dH1,n2,K2,dH2)
* dilution_only - heat proportional to injected volume, no binding (a)

A model that exhausts the iteration budget reports ResourceExhausted, distinct
from ordinary non-convergence and from semantic impossibility.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import ResourceExhausted

R_GAS = 8.314462618  # J mol^-1 K^-1


def _site_occupancy(x_free: float, n: float, m_cell: float, k: float) -> float:
    a = x_free + n * m_cell + 1.0 / k
    root_arg = a * a - 4.0 * n * m_cell * x_free
    return 0.5 * (a - math.sqrt(max(root_arg, 0.0)))


def _site_series(observed_shots: list[int], volumes: np.ndarray, cell_volume: float,
                 m_cell: float, ligand_syringe: float, n: float, k: float, dh: float
                 ) -> np.ndarray:
    """Standard one-site MicroCal differential heats for observed shot ids.

    volumes covers every shot 1..N; heats are simulated for the full titration
    then the requested observed shot ids are returned.
    """
    q_prev = 0.0
    accum = 0.0
    out = []
    for shot in range(1, max(observed_shots) + 1):
        dv = volumes[shot - 1]
        v_after = cell_volume + accum + dv
        accum += dv
        x_total = (m_cell * cell_volume + accum * ligand_syringe) / v_after
        bound = _site_occupancy(x_total, n, m_cell, k)
        q_now = v_after * dh * bound
        # MicroCal differential heat including displaced-volume correction:
        dq = q_now + dv / v_after * q_prev - cell_volume / v_after * q_prev
        q_prev = q_now
        out.append(dq)
    return np.array([out[i - 1] for i in observed_shots])


def one_site_heats(theta: dict, **kw) -> np.ndarray:
    heats = _site_series(kw["observed_shots"], kw["volumes"], kw["cell_volume"],
                         kw["macromolecule"], kw["ligand_syringe"],
                         theta["n"], math.exp(theta["logK"]), theta["dH"])
    return heats + theta.get("q0", 0.0)


def two_site_heats(theta: dict, **kw) -> np.ndarray:
    """Two independent site populations; heats add (same ligand pool)."""
    shots, volumes = kw["observed_shots"], kw["volumes"]
    cell_volume, m_cell, syringe = (kw["cell_volume"], kw["macromolecule"],
                                    kw["ligand_syringe"])
    q_prev = 0.0
    accum = 0.0
    out = []
    n1, n2 = theta["n1"], theta["n2"]
    k1, k2 = math.exp(theta["logK1"]), math.exp(theta["logK2"])
    dh1, dh2 = theta["dH1"], theta["dH2"]
    for shot in range(1, max(shots) + 1):
        dv = volumes[shot - 1]
        v_after = cell_volume + accum + dv
        accum += dv
        x_total = (m_cell * cell_volume + accum * syringe) / v_after
        # split the shared free-ligand pool by solving total mass balance with
        # two independent site classes:
        x_free = _free_ligand_two_site(x_total, m_cell, n1, k1, n2, k2)
        b1 = _site_occupancy(x_free, n1, m_cell, k1)
        b2 = _site_occupancy(x_free, n2, m_cell, k2)
        q_now = v_after * (dh1 * b1 + dh2 * b2)
        dq = q_now + dv / v_after * q_prev - cell_volume / v_after * q_prev
        q_prev = q_now
        out.append(dq)
    heats = np.array([out[i - 1] for i in shots])
    return heats + theta.get("q0", 0.0)


def _free_ligand_two_site(x_total: float, m: float, n1: float, k1: float,
                          n2: float, k2: float) -> float:
    x_lo, x_hi = 0.0, max(x_total, 1e-9)
    for _ in range(60):
        x = 0.5 * (x_lo + x_hi)
        used = _site_occupancy(x, n1, m, k1) + _site_occupancy(x, n2, m, k2)
        if used > x_total:
            x_hi = x
        else:
            x_lo = x
    return 0.5 * (x_lo + x_hi)


def dilution_heats(theta: dict, **kw) -> np.ndarray:
    """Dilution-only hypothesis: a constant heat per injection, no binding.

    A volume-proportional variant is equivalent here because injections share
    one volume, but the constant form matches the blank-correction rule.
    """
    shots = np.array(kw["observed_shots"])
    return np.full(shots.shape, theta["q_dil"])


BOUNDS = {
    "n": (0.02, 6.0), "n1": (0.02, 6.0), "n2": (0.02, 6.0),
    "logK": (2.0, 20.0), "logK1": (2.0, 20.0), "logK2": (2.0, 20.0),
    "dH": (-5e5, -10.0), "dH1": (-5e5, -10.0), "dH2": (-5e5, -10.0),
    "q0": (-1e-3, 1e-3), "q_dil": (-1e-3, 1e-3), "alpha": (-1e6, 1e6),
}


def _clip(names, p: np.ndarray) -> np.ndarray:
    return np.array([min(max(p[j], BOUNDS[name][0]), BOUNDS[name][1])
                     for j, name in enumerate(names)])


MODELS = {
    "one_site": {
        "fn": one_site_heats,
        "params": ["n", "logK", "dH", "q0"],
        "starts": [(0.5, math.log(1e5), -2e4, 0.0), (1.0, math.log(1e7), -5e4, -8e-6),
                   (2.0, math.log(1e3), -1e4, 8e-6)],
        "scale": {"n": 1.0, "logK": 4.0, "dH": 5e4, "q0": 1e-5},
        "dof": 4,
    },
    "two_site": {
        "fn": two_site_heats,
        "params": ["n1", "logK1", "dH1", "n2", "logK2", "dH2", "q0"],
        "starts": [(0.5, math.log(1e5), -2e4, 0.5, math.log(1e4), -1e4, -8e-6),
                   (1.0, math.log(1e7), -5e4, 1.0, math.log(1e6), -2e4, 0.0)],
        "scale": {"n1": 1.0, "logK1": 4.0, "dH1": 5e4,
                  "n2": 1.0, "logK2": 4.0, "dH2": 5e4, "q0": 1e-5},
        "dof": 7,
    },
    "dilution_only": {
        "fn": dilution_heats,
        "params": ["q_dil"],
        "starts": [(-1e-5,), (-8e-6,), (1e-6,)],
        "scale": {"q_dil": 1e-5},
        "dof": 1,
    },
}


def _jacobian(fn, p: np.ndarray, names, scales, kw, y: np.ndarray, *,
              used_points: int, point_budget: int) -> np.ndarray:
    base = fn(_theta(names, p), **kw)
    cols = []
    for j, name in enumerate(names):
        h = 1e-6 * max(abs(p[j]), 1e-4) * scales[name] / scales[name]
        p2 = p.copy()
        p2[j] += h
        cols.append((fn(_theta(names, p2), **kw) - base) / h)
        used_points += 2 * y.size
        if used_points > point_budget:
            raise ResourceExhausted(
                "fit point budget exhausted",
                budget="evaluation_points", used=used_points, limit=point_budget)
    return np.column_stack(cols), used_points


def _theta(names, p: np.ndarray) -> dict:
    return dict(zip(names, p))


def levenberg_marquardt(model_name: str, start: tuple, y: np.ndarray, kw: dict,
                        *, max_iter: int = 60, point_budget: int = 4_000_000
                        ) -> dict:
    spec = MODELS[model_name]
    names, scales = spec["params"], spec["scale"]
    p = _clip(names, np.array(start, dtype=float))
    lam = 1e-3
    used_points = 0
    reason = "max_iterations"
    for it in range(max_iter):
        f = spec["fn"](_theta(names, p), **kw)
        r = f - y
        used_points += y.size
        jac, used_points = _jacobian(spec["fn"], p, names, scales, kw, y,
                                     used_points=used_points,
                                     point_budget=point_budget)
        jtj = jac.T @ jac
        g = jac.T @ r
        try:
            step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        p_new = _clip(names, p - step)
        r_new = spec["fn"](_theta(names, p_new), **kw) - y
        used_points += y.size
        if used_points > point_budget:
            raise ResourceExhausted(
                "fit point budget exhausted",
                budget="evaluation_points", used=used_points, limit=point_budget)
        if np.sum(r_new ** 2) < np.sum(r ** 2):
            p = p_new
            lam = max(lam / 3, 1e-9)
            if np.max(np.abs(step / (np.abs(p) + 1e-8))) < 1e-8:
                reason = "converged"
                break
        else:
            lam *= 4
    residual = spec["fn"](_theta(names, p), **kw) - y
    return {"params": _theta(names, p), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "residual": residual.tolist(), "iterations": it + 1,
            "reason": reason, "evaluations": used_points}


def fit_candidates(y: np.ndarray, shot_indices: list[int], volumes: np.ndarray,
                   *, cell_volume: float, macromolecule: float, ligand_syringe: float,
                   temperature_k: float, models=("one_site", "two_site", "dilution_only"),
                   max_iter: int = 60, point_budget: int = 4_000_000,
                   equal_tol: float = 5e-3) -> dict:
    kw = {"volumes": volumes, "cell_volume": cell_volume,
          "macromolecule": macromolecule, "ligand_syringe": ligand_syringe,
          "temperature_k": temperature_k, "observed_shots": shot_indices}
    results = []
    diagnostics = []
    for name in models:
        best = None
        for start in MODELS[name]["starts"]:
            try:
                got = levenberg_marquardt(
                    name, start, y, kw, max_iter=max_iter,
                    point_budget=point_budget)
            except ResourceExhausted as exc:
                diagnostics.append(exc.to_dict())
                got = None
            if got and (best is None or got["rmse"] < best["rmse"]):
                best = got
        if best is None:
            continue
        n = len(y)
        dof = max(n - MODELS[name]["dof"], 1)
        aic = n * math.log(max(best["rmse"] ** 2, 1e-30)) + 2 * MODELS[name]["dof"]
        results.append({"model": name, **best, "aic": aic, "dof": MODELS[name]["dof"]})
    results.sort(key=lambda c: (round(c["rmse"] / max(abs(float(np.median(y))), 1e-30), 8),
                                c["aic"], c["model"]))
    if not results:
        return {"candidates": [], "diagnostics": diagnostics,
                "status": "no_candidate"}
    best_rmse = results[0]["rmse"]
    scale = max(abs(float(np.max(y) - np.min(y))), abs(float(np.median(y))), 1e-30)
    for rank, c in enumerate(results, start=1):
        c["rank"] = rank
        c["equivalent_best"] = (rank > 1 and abs(c["rmse"] - best_rmse) / scale <= equal_tol)
        c["id"] = f"cand-{c['model']}"
    results[0]["equivalent_best"] = any(c["equivalent_best"] for c in results[1:])
    return {"candidates": results,
            "diagnostics": diagnostics,
            "status": "equivalent" if results[0]["equivalent_best"] else "unique"}
