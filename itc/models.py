"""Binding model candidates (one-site, two independent sites, dilution-only)
with a NumPy Nelder-Mead fitter. The fitter has an explicit evaluation budget:
exhaustion is a first-class result, distinct from corrupt input or a
semantically unsolvable problem. When several minima fit equally well, all of
them are kept as equipotent candidates in a stable order."""

import hashlib
import json
import math

import numpy as np

from .integrate import SemanticUnsolvable


class BudgetExhausted(Exception):
    status = "budget_exhausted"


def simulate_cell(cond):
    m_cell, x_cell = cond["M0"], 0.0
    ms, xs = [], []
    for dv in cond["dv"]:
        frac = dv / cond["V0"]
        x_cell = x_cell * (1 - frac) + cond["Xs"] * frac
        m_cell = m_cell * (1 - frac)
        ms.append(m_cell)
        xs.append(x_cell)
    return ms, xs


def _bound_fraction(n, k, m, x):
    r = x / (n * m)
    a = 1 + r + 1.0 / (n * k * m)
    disc = a * a - 4 * r
    return 0.5 * (a - math.sqrt(max(disc, 0.0)))


def _site_heat_series(n, log_k, dh, cond):
    k = 10.0 ** log_k
    ms, xs = simulate_cell(cond)
    out, q_prev = [], 0.0
    for i, (m, x) in enumerate(zip(ms, xs)):
        q_total = n * m * cond["V0"] * dh * _bound_fraction(n, k, m, x)
        dv = cond["dv"][i]
        out.append(q_total - q_prev + (dv / (2 * cond["V0"])) * (q_total + q_prev))
        q_prev = q_total
    return out


def predict(family, params, cond):
    if family == "one_site":
        n, log_k, dh = params
        if n <= 0:
            return None
        return [q * 1e6 for q in _site_heat_series(n, log_k, dh, cond)]
    if family == "two_site":
        n1, k1, d1, n2, k2, d2 = params
        if n1 <= 0 or n2 <= 0:
            return None
        a = _site_heat_series(n1, k1, d1, cond)
        b = _site_heat_series(n2, k2, d2, cond)
        return [(x + y) * 1e6 for x, y in zip(a, b)]
    if family == "dilution":
        return [params[0]] * len(cond["dv"])
    raise ValueError(f"unknown family {family}")


def _ssr(family, params, cond, y, idx):
    pred = predict(family, params, cond)
    if pred is None or any(not math.isfinite(v) for v in pred):
        return 1e18
    return float(sum((pred[i] - y[j]) ** 2 for j, i in enumerate(idx)))


def _nelder_mead(func, x0, steps, counter, xtol=1e-7, ftol=1e-12, max_iter=600):
    def counted(x):
        if counter["used"] >= counter["budget"]:
            raise BudgetExhausted(f"evaluation budget {counter['budget']} exhausted")
        counter["used"] += 1
        return func(list(x))

    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        point[i] += steps[i]
        simplex.append(point)
    vals = [counted(p) for p in simplex]
    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        spread = max(abs(v - vals[0]) for v in vals)
        size = max(float(np.max(np.abs(simplex[i] - simplex[0]))) for i in range(1, n + 1))
        if spread < ftol and size < xtol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        f_ref = counted(reflected)
        if vals[0] <= f_ref < vals[-2]:
            simplex[-1], vals[-1] = reflected, f_ref
            continue
        if f_ref < vals[0]:
            expanded = centroid + 2 * (centroid - worst)
            f_exp = counted(expanded)
            simplex[-1], vals[-1] = (expanded, f_exp) if f_exp < f_ref else (reflected, f_ref)
            continue
        contracted = centroid + 0.5 * (worst - centroid)
        f_con = counted(contracted)
        if f_con < vals[-1]:
            simplex[-1], vals[-1] = contracted, f_con
            continue
        best = simplex[0]
        simplex = [best] + [best + 0.5 * (p - best) for p in simplex[1:]]
        vals = [vals[0]] + [counted(p) for p in simplex[1:]]
    order = np.argsort(vals)
    return list(simplex[order[0]]), vals[order[0]]


STARTS = {
    "one_site": [([1.0, 4.0, -5000.0], [0.3, 1.0, 2000.0]),
                 ([0.5, 6.0, -12000.0], [0.3, 1.0, 3000.0]),
                 ([1.5, 3.0, -2000.0], [0.3, 1.0, 1500.0])],
    "two_site": [([0.5, 5.0, -6000.0, 0.5, 3.5, -3000.0],
                  [0.2, 0.8, 2000.0, 0.2, 0.8, 2000.0]),
                 ([0.5, 3.5, -3000.0, 0.5, 5.0, -6000.0],
                  [0.2, 0.8, 2000.0, 0.2, 0.8, 2000.0])],
    "dilution": [([0.0], [1.0])],
}

PARAM_NAMES = {
    "one_site": ["n", "log10_K", "dH_cal_per_mol"],
    "two_site": ["n1", "log10_K1", "dH1_cal_per_mol", "n2", "log10_K2", "dH2_cal_per_mol"],
    "dilution": ["q_constant_ucal"],
}


def _candidate_id(family, params):
    key = json.dumps([family] + [round(p, 6) for p in params])
    return f"cand-{family}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def fit_models(heats, included, cond, families=("one_site", "two_site", "dilution"),
               budget=20000, equipotent_rel_tol=0.05, equipotent_abs_tol=1e-6):
    idx = [i for i, ok in enumerate(included) if ok]
    if cond["V0"] <= 0 or cond["M0"] <= 0 or cond["Xs"] <= 0:
        raise SemanticUnsolvable("cell volume and concentrations must be positive")
    if not idx:
        raise SemanticUnsolvable("no usable injections to fit")
    y = [heats[i] for i in idx]
    counter = {"used": 0, "budget": budget}
    candidates = []
    for family in families:
        seen = []
        for x0, steps in STARTS[family]:
            params, ssr = _nelder_mead(
                lambda p: _ssr(family, p, cond, y, idx), x0, steps, counter)
            if any(abs(ssr - s) < 1e-9 and
                   all(abs(a - b) < 1e-4 for a, b in zip(params, p))
                   for s, p in seen):
                continue
            seen.append((ssr, params))
            candidates.append({
                "id": _candidate_id(family, params),
                "family": family,
                "params": {name: round(val, 6)
                           for name, val in zip(PARAM_NAMES[family], params)},
                "ssr": ssr,
                "fitted_injections": idx,
            })
    candidates.sort(key=lambda c: (c["ssr"], c["family"], c["id"]))
    best = candidates[0]["ssr"]
    for rank, cand in enumerate(candidates):
        cand["rank"] = rank
        cand["equipotent"] = cand["ssr"] <= best * (1 + equipotent_rel_tol) + equipotent_abs_tol
        pred = predict(cand["family"], list(cand["params"].values()), cond)
        cand["residuals"] = ({str(i): round(pred[i] - heats[i], 6) for i in idx}
                             if pred is not None else {})
    return {"candidates": candidates, "evaluations": counter["used"],
            "budget": budget, "best_ssr": best}
