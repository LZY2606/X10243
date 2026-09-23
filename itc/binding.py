"""Binding-model candidates: one-site, two-site and dilution-only hypotheses.

NumPy-only fitting (grid + coordinate descent, linear parameters solved in
closed form). All candidates within the equally-good tolerance are retained;
nothing is collapsed to a single answer when the data cannot decide. The fit
runs under an evaluation budget; exhausting it yields budget_exhausted.
"""
from __future__ import annotations

import math

import numpy as np

from .results import Result, OK, BUDGET_EXHAUSTED, SEMANTIC_UNSOLVABLE
from .units import Diagnostic

PARAM_VERSION = "fit-v1"
R_KJ_PER_MOL_K = 8.314462618e-3

PARAM_UNITS = {
    "n": "site_ratio", "n1": "site_ratio", "n2": "site_ratio",
    "K_uM": "uM^-1", "K1_uM": "uM^-1", "K2_uM": "uM^-1",
    "dH_kJ_mol": "kJ/mol", "dH1_kJ_mol": "kJ/mol", "dH2_kJ_mol": "kJ/mol",
    "offset_uJ": "uJ", "c_uJ": "uJ",
}


class _Budget(Exception):
    pass


def _complex_uM(K_uM, sites_uM, xt_uM):
    """Bound complex concentration for 1:1 binding, quadratic solution."""
    big = K_uM * (sites_uM + xt_uM) + 1.0
    disc = max(big * big - 4.0 * K_uM * K_uM * sites_uM * xt_uM, 0.0)
    return (big - math.sqrt(disc)) / (2.0 * K_uM)


def heat_contents(model_type, params, M0_uM, Xs_uM, V0_uL, volumes_uL):
    """Cumulative cell heat content Q_i (uJ) after each injection.

    Concentrations evolve by exact exponential displacement; Q = V0*c*dH with
    the uL*uM*kJ/mol -> 1e-3 uJ conversion applied explicitly.
    """
    Q = []
    Mt = M0_uM
    Xt = 0.0
    for v in volumes_uL:
        r = math.exp(-v / V0_uL)
        Mt = Mt * r
        Xt = Xt * r + Xs_uM * (1.0 - r)
        if model_type == "one_site":
            c = _complex_uM(params["K_uM"], params["n"] * Mt, Xt)
            Q.append(V0_uL * c * params["dH_kJ_mol"] * 1e-3)
        elif model_type == "two_site":
            c1 = _complex_uM(params["K1_uM"], params["n1"] * Mt, Xt)
            c2 = _complex_uM(params["K2_uM"], params["n2"] * Mt, Xt)
            Q.append(V0_uL * (c1 * params["dH1_kJ_mol"] + c2 * params["dH2_kJ_mol"]) * 1e-3)
        else:
            raise ValueError(model_type)
    return Q


def predict_heats(model_type, params, M0_uM, Xs_uM, V0_uL, volumes_uL):
    """Per-injection heats (uJ) with displacement correction."""
    Q = heat_contents(model_type, params, M0_uM, Xs_uM, V0_uL, volumes_uL)
    q = []
    prev = 0.0
    for i, v in enumerate(volumes_uL):
        q.append(Q[i] - prev * (1.0 - v / V0_uL))
        prev = Q[i]
    offset = params.get("offset_uJ", 0.0)
    return [x + offset for x in q]


def _design_rows(model_type, fixed, M0, Xs, V0, vols):
    """Design columns for the linear (dH[, dH2], offset) solve."""
    if model_type == "one_site":
        f = predict_heats("one_site",
                          {"n": fixed[0], "K_uM": 10.0 ** fixed[1], "dH_kJ_mol": 1.0},
                          M0, Xs, V0, vols)
        return np.column_stack([np.array(f), np.ones(len(f))])
    if model_type == "two_site":
        base = {"n1": fixed[0], "n2": fixed[1],
                "K1_uM": 10.0 ** fixed[2], "K2_uM": 10.0 ** fixed[3],
                "dH1_kJ_mol": 0.0, "dH2_kJ_mol": 0.0}
        f1 = predict_heats("two_site", {**base, "dH1_kJ_mol": 1.0}, M0, Xs, V0, vols)
        f2 = predict_heats("two_site", {**base, "dH2_kJ_mol": 1.0}, M0, Xs, V0, vols)
        return np.column_stack([np.array(f1), np.array(f2), np.ones(len(f1))])
    raise ValueError(model_type)


def _refine(eval_fn, x0, steps, max_iter=80):
    x = list(x0)
    best = eval_fn(x)
    st = list(steps)
    for _ in range(max_iter):
        improved = False
        for i in range(len(x)):
            for s in (st[i], -st[i]):
                y = list(x)
                y[i] += s
                r = eval_fn(y)
                if r < best - 1e-12:
                    best, x = r, y
                    improved = True
        if not improved:
            st = [s / 2.0 for s in st]
            if max(st) < 1e-3:
                break
    return x, best


def fit_models(heats_uJ, volumes_uL, M0_uM, Xs_uM, V0_uL, temp_C,
               budget=60000, rss_rel_tol=0.10, rss_abs_tol=1.5,
               model_types=("one_site", "two_site", "dilution")):
    """Returns (Result, events)."""
    q = np.asarray(list(heats_uJ), dtype=float)
    vols = list(volumes_uL)
    events = []
    if len(q) < 4:
        return Result(SEMANTIC_UNSOLVABLE, None, [Diagnostic(
            "insufficient_points",
            f"need at least 4 valid injection heats to fit, got {len(q)}",
            "fit_input", {})]), events

    state = {"used": 0}

    def charge(k=1):
        state["used"] += k
        if state["used"] > budget:
            raise _Budget()

    candidates = []
    try:
        if "one_site" in model_types:
            def rss_one(x):
                charge()
                A = _design_rows("one_site", x, M0_uM, Xs_uM, V0_uL, vols)
                coef, *_ = np.linalg.lstsq(A, q, rcond=None)
                r = q - A @ coef
                return float(r @ r)

            grid = []
            for n_sites in np.arange(0.5, 1.5001, 0.05):
                for lgk in np.arange(-2.0, 1.0001, 0.1):
                    grid.append((rss_one([n_sites, lgk]), n_sites, lgk))
            grid.sort()
            optima = []
            for _, n0, k0 in grid[:3]:
                x, rss = _refine(rss_one, [n0, k0], [0.02, 0.05])
                optima.append((rss, x))
            optima.sort()
            seen = set()
            for rss, x in optima:
                key = (round(x[0], 2), round(x[1], 2))
                if key in seen:
                    continue
                seen.add(key)
                A = _design_rows("one_site", x, M0_uM, Xs_uM, V0_uL, vols)
                coef, *_ = np.linalg.lstsq(A, q, rcond=None)
                candidates.append({
                    "model_type": "one_site",
                    "params": {"n": x[0], "K_uM": 10.0 ** x[1],
                               "dH_kJ_mol": float(coef[0]), "offset_uJ": float(coef[1])},
                    "rss": rss,
                })

        if "two_site" in model_types:
            def rss_two(x):
                charge()
                if x[3] > x[2]:
                    return float("inf")
                A = _design_rows("two_site", x, M0_uM, Xs_uM, V0_uL, vols)
                coef, *_ = np.linalg.lstsq(A, q, rcond=None)
                r = q - A @ coef
                return float(r @ r)

            grid = []
            lgks = np.arange(-1.5, 1.0001, 0.25)
            for i1, k1 in enumerate(lgks):
                for k2 in lgks[: i1 + 1]:
                    grid.append((rss_two([1.0, 1.0, k1, k2]), k1, k2))
            grid.sort()
            optima = []
            for _, k1, k2 in grid[:2]:
                x, rss = _refine(rss_two, [1.0, 1.0, k1, k2], [0.05, 0.05, 0.05, 0.05])
                optima.append((rss, x))
            optima.sort()
            seen = set()
            for rss, x in optima:
                key = (round(x[2], 2), round(x[3], 2))
                if key in seen:
                    continue
                seen.add(key)
                A = _design_rows("two_site", x, M0_uM, Xs_uM, V0_uL, vols)
                coef, *_ = np.linalg.lstsq(A, q, rcond=None)
                candidates.append({
                    "model_type": "two_site",
                    "params": {"n1": x[0], "n2": x[1],
                               "K1_uM": 10.0 ** x[2], "K2_uM": 10.0 ** x[3],
                               "dH1_kJ_mol": float(coef[0]), "dH2_kJ_mol": float(coef[1]),
                               "offset_uJ": float(coef[2])},
                    "rss": rss,
                })

        if "dilution" in model_types:
            charge()
            c = float(np.mean(q))
            rss = float(np.sum((q - c) ** 2))
            candidates.append({"model_type": "dilution", "params": {"c_uJ": c}, "rss": rss})
    except _Budget:
        return Result(BUDGET_EXHAUSTED, {"candidates": candidates, "param_version": PARAM_VERSION},
                      [Diagnostic("budget_exhausted", "fit evaluation budget exhausted",
                                  "binding.fit", {"evaluations": state["used"]})]), events

    candidates.sort(key=lambda c: (c["rss"], c["model_type"]))
    temp_K = temp_C + 273.15
    per_type_rank = {}
    for cand in candidates:
        mt = cand["model_type"]
        rank = per_type_rank.get(mt, 0)
        per_type_rank[mt] = rank + 1
        cand["candidate_id"] = f"cand-{mt}-{rank}"
        cand["param_units"] = {k: PARAM_UNITS[k] for k in cand["params"]}
        if mt == "one_site":
            cand["deltaG_kJ_mol"] = -R_KJ_PER_MOL_K * temp_K * math.log(cand["params"]["K_uM"] * 1e6)
        elif mt == "two_site":
            cand["deltaG_kJ_mol"] = [
                -R_KJ_PER_MOL_K * temp_K * math.log(cand["params"]["K1_uM"] * 1e6),
                -R_KJ_PER_MOL_K * temp_K * math.log(cand["params"]["K2_uM"] * 1e6),
            ]
    rss_min = candidates[0]["rss"]
    for cand in candidates:
        cand["equally_good"] = cand["rss"] <= rss_min * (1.0 + rss_rel_tol) + rss_abs_tol
    eg = [c["candidate_id"] for c in candidates if c["equally_good"]]
    if len(eg) > 1:
        events.append({"kind": "judgment.equally_good_models", "payload": {
            "candidate_ids": eg,
            "rss_rel_tol": rss_rel_tol,
            "rss_abs_tol": rss_abs_tol,
            "rule": "all candidates within tolerance of the best RSS are retained",
        }})
    value = {
        "candidates": candidates,
        "n_points": int(len(q)),
        "param_version": PARAM_VERSION,
        "tolerances": {"rss_rel_tol": rss_rel_tol, "rss_abs_tol": rss_abs_tol},
        "units": {"heat": "uJ", "rss": "uJ^2", "deltaG": "kJ/mol"},
        "evaluations": state["used"],
    }
    return Result(OK, value, []), events
