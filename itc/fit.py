"""Binding-model candidates: one-site, two-site (multi-site) and dilution
hypothesis. Pure NumPy; no scipy.

- Exhausting the evaluation budget yields status ``budget_exhausted``.
- Having no usable heats yields status ``semantic_unsolvable``.
- Candidates within tolerance of the best chi2 are all retained as
  ``equally_good``; ordering is stable: (chi2, model name).
"""

import itertools

import numpy as np


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, max_evals):
        self.left = int(max_evals)

    def use(self, n=1):
        self.left -= n
        if self.left < 0:
            raise BudgetExhausted()


def cell_trajectory(dv, V0, Mt0, Cs):
    """Cell concentrations after each injection (displacement model)."""
    n = len(dv)
    Mt = np.zeros(n)
    Xt = np.zeros(n)
    mt, xt = Mt0, 0.0
    for i in range(n):
        f = dv[i] / V0
        xt = xt * (1 - f) + Cs * f
        mt = mt * (1 - f)
        Mt[i] = mt
        Xt[i] = xt
    return Mt, Xt


def bound_ligand(Mt, Xt, K, n):
    s = n * Mt + Xt + 1.0 / K
    disc = np.maximum(s * s - 4.0 * n * Mt * Xt, 0.0)
    return (s - np.sqrt(disc)) / 2.0


def heat_signal(Mt, Xt, dv, V0, K, n):
    """S_i such that Q_i = dH * S_i for a one-site class."""
    Lb = bound_ligand(Mt, Xt, K, n)
    Lb_prev = np.concatenate([[0.0], Lb[:-1]])
    dil = 1.0 - dv / V0
    return V0 * (Lb - Lb_prev * dil)


def _grid_search(eval_fn, axes, budget, refine_rounds=4):
    best = None
    grids = [np.linspace(a, b, c) for a, b, c in axes]
    for point in itertools.product(*grids):
        chi2, prm = eval_fn(*point)
        if best is None or chi2 < best[0]:
            best = (chi2, list(point), prm)
    steps = [(b - a) / max(c - 1, 1) for a, b, c in axes]
    for _ in range(refine_rounds):
        improved = False
        for d in range(len(axes)):
            for sign in (1.0, -1.0):
                cand = list(best[1])
                cand[d] += sign * steps[d]
                chi2, prm = eval_fn(*cand)
                if chi2 < best[0]:
                    best = (chi2, cand, prm)
                    improved = True
        if not improved:
            steps = [s / 2.0 for s in steps]
    return best


def fit_models(heats, dv_all, conc_cell_M, conc_syringe_M, V0_L, params=None):
    """Fit all candidate model families.

    heats: [{"index": i, "heat": value_in_J}, ...] for included injections.
    dv_all: injection volumes in L for every injection (trajectory needs all).
    """
    params = dict(params or {})
    budget = Budget(params.get("max_evals", 20000))
    rel_tol = params.get("equal_rel_tol", 0.05)
    abs_tol = params.get("equal_abs_tol", 1e-9)

    diagnostics = []
    if not heats:
        return {"status": "semantic_unsolvable",
                "diagnostics": [{"code": "no_usable_heats", "path": "heats"}],
                "candidates": []}
    if any(v is None for v in dv_all):
        return {"status": "semantic_unsolvable",
                "diagnostics": [{"code": "volume_unconvertible",
                                 "path": "injection_schedule.injections.volume"}],
                "candidates": []}

    dv = np.asarray(dv_all, dtype=float)
    mask = np.zeros(len(dv), dtype=bool)
    Qfull = np.zeros(len(dv))
    for h in heats:
        mask[h["index"]] = True
        Qfull[h["index"]] = h["heat"]
    Qm = Qfull[mask]
    idxs = [h["index"] for h in heats]
    Mt, Xt = cell_trajectory(dv, V0_L, conc_cell_M, conc_syringe_M)

    def eval_one_site(logK, n):
        budget.use()
        S = heat_signal(Mt, Xt, dv, V0_L, 10.0 ** logK, n)[mask]
        denom = float(S @ S)
        dH = float(S @ Qm) / denom if denom > 0 else 0.0
        r = Qm - dH * S
        return float(r @ r), {"K": 10.0 ** logK, "n": n, "dH": dH}

    def eval_two_site(logK1, logK2):
        budget.use()
        S1 = heat_signal(Mt, Xt, dv, V0_L, 10.0 ** logK1, 1.0)[mask]
        S2 = heat_signal(Mt, Xt, dv, V0_L, 10.0 ** logK2, 1.0)[mask]
        A = np.column_stack([S1, S2])
        sol, *_ = np.linalg.lstsq(A, Qm, rcond=None)
        r = Qm - A @ sol
        return float(r @ r), {"K1": 10.0 ** logK1, "K2": 10.0 ** logK2,
                              "dH1": float(sol[0]), "dH2": float(sol[1]),
                              "n1": 1.0, "n2": 1.0}

    def eval_dilution():
        budget.use()
        A = np.column_stack([np.ones(int(mask.sum())), dv[mask]])
        sol, *_ = np.linalg.lstsq(A, Qm, rcond=None)
        r = Qm - A @ sol
        return float(r @ r), {"c0": float(sol[0]), "c1": float(sol[1])}

    def predicted(model, prm):
        if model == "one_site":
            S = heat_signal(Mt, Xt, dv, V0_L, prm["K"], prm["n"])[mask]
            return prm["dH"] * S
        if model == "two_site":
            S1 = heat_signal(Mt, Xt, dv, V0_L, prm["K1"], 1.0)[mask]
            S2 = heat_signal(Mt, Xt, dv, V0_L, prm["K2"], 1.0)[mask]
            return prm["dH1"] * S1 + prm["dH2"] * S2
        return prm["c0"] + prm["c1"] * dv[mask]

    candidates = []
    try:
        chi2, _, prm = _grid_search(eval_one_site, [(2.0, 8.0, 25), (0.5, 1.6, 12)], budget)
        candidates.append({"model": "one_site", "params": prm, "chi2": chi2})
        chi2, _, prm = _grid_search(eval_two_site, [(2.0, 8.0, 13), (2.0, 8.0, 13)], budget)
        candidates.append({"model": "two_site", "params": prm, "chi2": chi2})
        chi2, prm = eval_dilution()
        candidates.append({"model": "dilution", "params": prm, "chi2": chi2})
        status = "ok"
    except BudgetExhausted:
        diagnostics.append({"code": "budget_exhausted",
                            "detail": "evaluation budget exhausted during fit"})
        status = "budget_exhausted"

    if candidates:
        candidates.sort(key=lambda c: (c["chi2"], c["model"]))
        best = candidates[0]["chi2"]
        tol = max(abs_tol, rel_tol * abs(best))
        for rank, c in enumerate(candidates):
            c["rank"] = rank
            c["equally_good"] = (c["chi2"] - best) <= tol
            pred = predicted(c["model"], c["params"])
            c["residuals"] = [
                {"index": i, "value": float(q - p), "unit": "J"}
                for i, q, p in zip(idxs, Qm, pred)
            ]
            c["n_points"] = int(mask.sum())
    return {"status": status, "candidates": candidates, "diagnostics": diagnostics}
