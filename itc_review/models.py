"""Binding-model candidates: one-site Wiseman, two-site, dilution-only.

Fitting uses NumPy only.  A deterministic seeded grid of starts feeds a
Nelder-Mead simplex; the evaluation budget is explicit and exhaustion raises
ResourceBudgetExhausted (a distinct outcome from corruption / impossibility).
Equally optimal models are all retained and ranked deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .errors import ResourceBudgetExhausted

R = 1.987204258  # cal/(mol*K)
TIE_BIC_DELTA = 2.0
DEFAULT_BUDGET = 40000


def cumulative_syringe_concentration(
    volumes_l: np.ndarray, syringe_m: float, cell_v0_l: float
) -> np.ndarray:
    """Ligand concentration in cell after each injection (displaced volume)."""
    total = np.cumsum(volumes_l)
    return syringe_m * (cell_v0_l / total) * (1.0 - np.exp(-total / cell_v0_l))


def _theta_root(X: np.ndarray, n: float, K: float) -> np.ndarray:
    M_term = X + 1.0 / K + n
    disc = M_term**2 - 4.0 * X * n
    disc = np.maximum(disc, 0.0)
    return (M_term - np.sqrt(disc)) / 2.0


def one_site_heats(volumes_l: np.ndarray, *, syringe_m: float,
                   macromolecule_m: float, cell_v0_l: float,
                   K: float, dH: float, n: float) -> np.ndarray:
    """Per-injection heat (microcalorie) for one set of sites."""
    X = cumulative_syringe_concentration(volumes_l, syringe_m, cell_v0_l)
    theta = _theta_root(X, n, K) / n
    Qt = macromolecule_m * cell_v0_l * dH * theta  # cal
    Qprev = np.concatenate(([0.0], Qt[:-1]))
    factor = cell_v0_l / (cell_v0_l - np.concatenate(([0.0], volumes_l[:-1])) / 2.0)
    qi = Qt + factor * np.concatenate(([0.0], volumes_l[:-1])) / cell_v0_l * Qprev - Qprev
    # simpler, stable Wiseman form:
    qi = np.empty_like(Qt)
    qi[0] = Qt[0]
    for k in range(1, len(Qt)):
        qi[k] = Qt[k] - Qt[k - 1] + (volumes_l[k - 1] / cell_v0_l) * (
            Qt[k] + Qt[k - 1]) / 2.0
    return qi * 1e6


def two_site_heats(volumes_l: np.ndarray, *, syringe_m: float,
                   macromolecule_m: float, cell_v0_l: float,
                   K1: float, dH1: float, K2: float,
                   dH2: float) -> np.ndarray:
    X = cumulative_syringe_concentration(volumes_l, syringe_m, cell_v0_l)
    th1 = _theta_root(X, 1.0, K1)
    th2 = _theta_root(X, 1.0, K2)
    Qt = macromolecule_m * cell_v0_l * (dH1 * th1 + dH2 * th2)
    qi = np.empty_like(Qt)
    qi[0] = Qt[0]
    for k in range(1, len(Qt)):
        qi[k] = Qt[k] - Qt[k - 1] + (volumes_l[k - 1] / cell_v0_l) * (
            Qt[k] + Qt[k - 1]) / 2.0
    return qi * 1e6


def dilution_heats(volumes_l: np.ndarray, *, syringe_m: float,
                   cell_v0_l: float, dH_dil: float,
                   offset: float = 0.0) -> np.ndarray:
    """Dilution-only: heat proportional to injected ligand amount."""
    moles = syringe_m * volumes_l  # mol
    return moles * dH_dil * 1e6 + offset


@dataclass
class ModelSpec:
    name: str
    label: str
    nparams: int
    starts: list[tuple[float, ...]]
    predict: Callable[..., np.ndarray]
    keys: list[str]
    units: dict[str, str]


def _specs() -> dict[str, ModelSpec]:
    os_starts = [(lk, ln, dh)
                 for lk in np.linspace(3.0, 9.0, 7)
                 for ln in np.linspace(-0.55, 0.05, 3)
                 for dh in np.linspace(-20000.0, 5000.0, 3)]
    ts_starts = [(lk1, dh1, lk2, dh2)
                 for lk1 in np.linspace(3.0, 8.0, 4)
                 for lk2 in np.linspace(3.0, 8.0, 4)
                 for dh1 in (-10000.0, -3000.0)
                 for dh2 in (-3000.0, 0.0)]
    dl_starts = [(a, b) for a in np.linspace(-500.0, 500.0, 5)
                 for b in np.linspace(-50.0, 50.0, 3)]
    return {
        "one_site": ModelSpec(
            "one_site", "One-site Wiseman", 3, os_starts,
            one_site_heats, ["K", "n", "dH"],
            {"K": "M^-1", "n": "sites", "dH": "cal/mol"}),
        "two_site": ModelSpec(
            "two_site", "Two independent sites", 4, ts_starts,
            two_site_heats, ["K1", "dH1", "K2", "dH2"],
            {"K1": "M^-1", "dH1": "cal/mol", "K2": "M^-1",
             "dH2": "cal/mol"}),
        "dilution": ModelSpec(
            "dilution", "Dilution-only hypothesis", 2, dl_starts,
            dilution_heats, ["dH_dil", "offset"],
            {"dH_dil": "cal/mol", "offset": "µcal"}),
    }


def _to_native(name: str, z: np.ndarray) -> dict[str, float]:
    if name == "one_site":
        return {"K": float(10 ** z[0]), "n": float(10 ** z[1]),
                "dH": float(z[2])}
    if name == "two_site":
        return {"K1": float(10 ** z[0]), "dH1": float(z[1]),
                "K2": float(10 ** z[2]), "dH2": float(z[3])}
    return {"dH_dil": float(z[0]), "offset": float(z[1])}


def _residual(name: str, z: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    v = ctx["volumes_l"]
    common = dict(syringe_m=ctx["syringe_m"], cell_v0_l=ctx["cell_v0_l"])
    if name == "one_site":
        p = _to_native(name, z)
        pred = one_site_heats(v, macromolecule_m=ctx["macromolecule_m"],
                              K=p["K"], n=p["n"], dH=p["dH"], **common)
    elif name == "two_site":
        p = _to_native(name, z)
        pred = two_site_heats(v, macromolecule_m=ctx["macromolecule_m"],
                              K1=p["K1"], dH1=p["dH1"], K2=p["K2"],
                              dH2=p["dH2"], **common)
    else:
        p = _to_native(name, z)
        pred = dilution_heats(v, dH_dil=p["dH_dil"], offset=p["offset"],
                              **common)
    return pred - ctx["heats_ucal"]


def budgeted_nelder_mead(name: str, x0: np.ndarray, ctx: dict[str, Any],
                         budget: int, counter: list[int]) -> tuple[np.ndarray, float, bool]:
    """Deterministic simplex optimizer; raises when the budget is exhausted."""
    def f(z: np.ndarray) -> float:
        counter[0] += 1
        if counter[0] > budget:
            raise ResourceBudgetExhausted(
                f"fit evaluation budget exhausted for {name}",
                budget=budget, used=counter[0])
        r = _residual(name, z, ctx)
        return float(np.dot(r, r))

    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        x[i] += 0.35 if abs(x[i]) > 1e-9 else 0.25
        simplex.append(x)
    vals = [f(x) for x in simplex]
    order = np.argsort(vals)
    simplex = [simplex[i] for i in order]
    vals = [vals[i] for i in order]

    for _ in range(2000):
        if max(np.linalg.norm(simplex[j] - simplex[0]) for j in range(1, n + 1)) < 1e-7:
            return simplex[0], vals[0], True
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = f(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        elif fr < vals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid - 0.5 * (centroid - simplex[-1])
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
    return simplex[0], vals[0], False


def fit_models(*, volumes_l: np.ndarray, heats_ucal: np.ndarray,
               syringe_m: float, macromolecule_m: float, cell_v0_l: float,
               temperature_k: float, budget: int = DEFAULT_BUDGET,
               include: tuple[str, ...] = ("one_site", "two_site", "dilution")
               ) -> list[dict[str, Any]]:
    """Fit all candidates; every candidate and its rank is preserved."""
    specs = _specs()
    ctx = dict(volumes_l=volumes_l, heats_ucal=heats_ucal,
               syringe_m=syringe_m, macromolecule_m=macromolecule_m,
               cell_v0_l=cell_v0_l, temperature_k=temperature_k)
    n = int(len(volumes_l))
    results: list[dict[str, Any]] = []
    counter = [0]
    for name in include:
        spec = specs[name]
        if n < spec.nparams + 1 or (name == "two_site" and n < 8):
            results.append({"model": name, "label": spec.label,
                            "status": "insufficient-data", "keys": spec.keys,
                            "units": spec.units})
            continue
        best = None
        converged = False
        evals = 0
        for start in spec.starts:
            z0 = np.array(start, dtype=float)
            z, sse, conv = budgeted_nelder_mead(
                name, z0, ctx, budget, counter)
            evals = counter[0]
            if best is None or sse < best[1]:
                best = (z, sse)
            converged = converged or conv
        assert best is not None
        z, sse = best
        params = _to_native(name, z)
        pred_ctx = dict(ctx)
        pred = (_residual(name, z, ctx) + heats_ucal)
        residuals = (pred - heats_ucal).tolist()
        bic = n * float(np.log(max(sse, 1e-12) / n)) + spec.nparams * np.log(n)
        if name == "one_site":
            K, dH = params["K"], params["dH"]
            params["dS"] = float(dH / temperature_k - R * np.log(max(K, 1e-30)))
        results.append({
            "model": name, "label": spec.label, "status": "ok",
            "keys": spec.keys + (["dS"] if name == "one_site" else []),
            "units": {**spec.units, **({"dS": "cal/(mol*K)"} if
                                       name == "one_site" else {})},
            "params": params, "sse": float(sse), "bic": float(bic),
            "rmse_ucal": float(np.sqrt(sse / n)),
            "predicted_ucal": [float(x) for x in pred],
            "residuals_ucal": [float(x) for x in residuals],
            "evaluations": evals, "converged": bool(converged),
            "npoints": n,
        })

    ok = [r for r in results if r.get("status") == "ok"]
    best_bic = min(r["bic"] for r in ok)
    ranked = sorted(ok, key=lambda r: (round(r["bic"], 9), r["model"]))
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank
        r["delta_bic"] = float(r["bic"] - best_bic)
        r["is_best_group"] = bool(r["delta_bic"] <= TIE_BIC_DELTA)
        r["is_unique_best"] = bool(rank == 1 and
                                   not any(x["delta_bic"] <= TIE_BIC_DELTA
                                           for x in ranked[1:]))
    best_group = [r for r in ranked if r["is_best_group"]]
    tied = len(best_group) > 1
    for r in ranked:
        r["equally_optimal"] = tied and r["is_best_group"]
        r["decision_required"] = tied and r["is_best_group"]
    return sorted(results, key=lambda r: (r.get("rank", 99), r["model"]))
