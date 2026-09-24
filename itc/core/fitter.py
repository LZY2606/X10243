"""绑定模型拟合与候选选择。

- Levenberg-Marquardt（数值雅可比），预算 = 最大迭代 / 最大求值；
  预算耗尽是 resource_budget_exhausted，与不收敛/无解区分；
- 多起点网格（two_site 固定较少起点），参数规范化后去重；
- BIC = n*ln(RSS/n) + k*ln n；delta-BIC <= tie_tol 的全部候选保留为等优集合，
  等优集合顺序按 (BIC, 模型稳定序, 规范化参数) 稳定排列。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .constants import ERR_BUDGET
from .errors import ResourceBudgetExhaustedError
from .hashing import canonical
from .models import (MODELS, PARAM_SPECS, canonical_theta, delta_g,
                     simulate)


class Budget:
    def __init__(self, max_evals: int = 4000, max_iters: int = 80):
        self.max_evals = max_evals
        self.max_iters = max_iters
        self.evals = 0
        self.iters = 0

    def eval(self) -> None:
        self.evals += 1
        if self.evals > self.max_evals:
            raise ResourceBudgetExhaustedError(
                "拟合求值次数超过预算",
                details={"used_evals": self.evals,
                         "max_evals": self.max_evals})


def fit_dataset(points: list[dict[str, Any]], v0: float, c_cell: float,
                c_syringe: float, temperature_k: float,
                tie_tol: float = 2.0, models: tuple[str, ...] = MODELS,
                injected: bool = False,
                budget: Budget | None = None) -> dict[str, Any]:
    budget = budget or Budget()
    fit_points = [p for p in points if p.get("participates")
                  and p.get("q_kcal_mol") is not None
                  and not p.get("blank_missing")]
    if len(fit_points) < 3:
        return {
            "node_type": "fit", "status": "semantic_unsolvable",
            "reason": "fewer_than_three_usable_injections",
            "candidates": [], "usable_ordinals":
                [p["ordinal"] for p in fit_points],
        }
    aux = [{**p, "_c_syringe": c_syringe} for p in fit_points]
    y = np.array([p["q_kcal_mol"] for p in fit_points], dtype=float)

    candidates: list[dict[str, Any]] = []
    for model in models:
        for theta0 in _starts(model, injected):
            try:
                theta, rss, converged, used = _lm(
                    model, theta0, aux, y, v0, c_cell, c_syringe, budget)
            except ResourceBudgetExhaustedError:
                return {"node_type": "fit", "status": ERR_BUDGET,
                        "candidates": [],
                        "usable_ordinals": [p["ordinal"] for p in fit_points]}
            if not converged or not math.isfinite(rss):
                continue
            k = len(PARAM_SPECS[model])
            n = len(y)
            bic = n * math.log(max(rss / n, 1e-30)) + k * math.log(n)
            theta = canonical_theta(model, theta)
            candidates.append({
                "model": model, "theta": theta, "rss": rss, "bic": bic,
                "k": k, "converged": True,
                "delta_g": _thermo(model, theta, temperature_k),
                "fingerprint": canonical(
                    [model, {p: round(v, 8) for p, v in theta.items()}]),
            })
    candidates = _dedupe(candidates)
    if not candidates:
        return {"node_type": "fit", "status": "semantic_unsolvable",
                "reason": "no_converged_candidate", "candidates": [],
                "usable_ordinals": [p["ordinal"] for p in fit_points]}
    candidates.sort(key=lambda c: (round(c["bic"], 6),
                                   MODELS.index(c["model"]),
                                   c["fingerprint"]))
    best_bic = candidates[0]["bic"]
    tied = [c for c in candidates if c["bic"] - best_bic <= tie_tol]
    for rank, c in enumerate(candidates):
        c["rank"] = rank
        c["is_equal_optimal"] = c in tied
    chosen = tied[0]
    return {
        "node_type": "fit", "status": "ok",
        "tie_tol": tie_tol,
        "chosen_model": chosen["model"],
        "unique_selection": len(tied) == 1,
        "equal_optimal": [c["model"] for c in tied],
        "candidates": candidates,
        "usable_ordinals": [p["ordinal"] for p in fit_points],
        "rule_version": "fit-lm-bic-v1",
    }
def predictions(candidate: dict[str, Any],
                points: list[dict[str, Any]], v0: float, c_cell: float,
                c_syringe: float) -> list[dict[str, Any]]:
    pred = simulate(candidate["model"], candidate["theta"], points, v0,
                    c_cell, c_syringe)
    out = []
    for p, phat in zip(points, pred):
        y = p.get("q_kcal_mol")
        out.append({"ordinal": p["ordinal"], "observed": y,
                    "predicted": phat,
                    "residual": (y - phat) if (y is not None and phat
                                                is not None) else None})
    return out


def _thermo(model: str, theta: dict[str, float],
            temperature_k: float) -> dict[str, float]:
    out = {}
    if model == "one_site":
        out["dG"] = delta_g(theta["logK"], temperature_k)
    if model == "two_site":
        out["dG1"] = delta_g(theta["logK1"], temperature_k)
        out["dG2"] = delta_g(theta["logK2"], temperature_k)
    return out


def _starts(model: str, injected: bool) -> list[dict[str, float]]:
    if model == "dilution":
        return [{"q0": v} for v in (-2.0, 0.0, 2.0)]
    if model == "one_site":
        grid = []
        for n in (0.8, 1.2, 2.0):
            for logk in (4.0, 5.5, 7.0):
                for dh in (-10.0, -5.0):
                    grid.append({"n": n, "logK": logk, "dH": dh})
        return grid
    return [
        {"n1": 0.6, "logK1": 5.0, "dH1": -6.0,
         "n2": 0.6, "logK2": 7.0, "dH2": -3.0},
        {"n1": 1.0, "logK1": 6.0, "dH1": -8.0,
         "n2": 1.0, "logK2": 4.5, "dH2": -2.0},
    ]


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for c in candidates:
        key = f"{c['model']}:{c['fingerprint']}"
        if key not in seen or c["bic"] < seen[key]["bic"]:
            seen[key] = c
    return list(seen.values())


def _residuals(model: str, theta: dict[str, float],
               points: list[dict[str, Any]], y: np.ndarray,
               v0: float, c_cell: float, c_syringe: float,
               budget: Budget) -> np.ndarray:
    budget.eval()
    pred = simulate(model, theta, points, v0, c_cell, c_syringe)
    return y - np.array([p if p is not None else 0.0 for p in pred])


def _lm(model: str, theta0: dict[str, float], points: list[dict[str, Any]],
        y: np.ndarray, v0: float, c_cell: float, c_syringe: float,
        budget: Budget, max_iter: int = 60
        ) -> tuple[dict[str, float], float, bool, int]:
    names = [s[0] for s in PARAM_SPECS[model]]
    lo = np.array([s[2] for s in PARAM_SPECS[model]])
    hi = np.array([s[3] for s in PARAM_SPECS[model]])
    x = np.array([theta0[n] for n in names], dtype=float)
    x = np.minimum(np.maximum(x, lo + 1e-6), hi - 1e-6)

    def pack(xv: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(names, xv)}

    r = _residuals(model, pack(x), points, y, v0, c_cell, c_syringe, budget)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(max_iter):
        budget.iters += 1
        if budget.iters > budget.max_iters:
            raise ResourceBudgetExhaustedError(
                "拟合迭代次数超过预算",
                details={"used_iters": budget.iters})
        j = np.zeros((len(y), len(x)))
        base = pack(x)
        for jcol in range(len(x)):
            step = max(1e-6, 1e-5 * abs(x[jcol]))
            xp = x.copy()
            xp[jcol] = min(x[jcol] + step, hi[jcol] - 1e-8)
            rp = _residuals(model, pack(xp), points, y, v0, c_cell,
                            c_syringe, budget)
            j[:, jcol] = (rp - r) / max(xp[jcol] - x[jcol], 1e-12)
        jtj = j.T @ j
        g = j.T @ r
        accepted = False
        for _try in range(12):
            a = jtj + lam * np.diag(np.diag(jtj) + 1e-12)
            try:
                dx = np.linalg.solve(a, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = np.minimum(np.maximum(x + dx, lo + 1e-8), hi - 1e-8)
            rn = _residuals(model, pack(xn), points, y, v0, c_cell,
                            c_syringe, budget)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                break
            lam *= 3.0
        if accepted and np.linalg.norm(g, ord=np.inf) < 1e-7:
            return pack(x), cost, True, budget.evals
    converged = cost < float(np.sum((y - np.mean(y)) ** 2)) and cost > 0
    return pack(x), cost, converged, budget.evals
