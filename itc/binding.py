"""Binding-model candidates and a deterministic NumPy-only fitter.

Models compared on the review page:

* ``one_site``  - single-class Wiseman saturation (n, K, dH, offset)
* ``two_site``  - additive two-class model (n1,K1,dH1,n2,K2,dH2,offset)
* ``dilution``  - constant heat of dilution (offset only)

The fitter is deterministic: fixed grids, fixed simplex seeds and a bounded
iteration budget. Exhausting that budget is a distinct outcome
(:class:`BudgetExhausted`), never a silent default. All co-optimal models are
retained when |delta BIC| < the rule-set tie threshold.

Reference form (Wiseman, 1989):

    Q_i  = n M0 dH V0 theta(Xt_i)
    dQ_i = Q_i - Q_{i-1} + dV_i/V0 * (Q_i + Q_{i-1})/2

with ligand mass balance under sequential displacement.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import BudgetExhausted, SemanticImpasse
from .rules import RuleSet

N_MIN, N_MAX = 0.02, 30.0
LOGK_MIN, LOGK_MAX = 2.0, 9.0  # K in 1/M


def cumulative_ligand(injected_volumes_l: np.ndarray, cell_volume_l: float,
                      syr_conc_m: float) -> np.ndarray:
    dv = np.asarray(injected_volumes_l, dtype=float)
    v0 = float(cell_volume_l)
    disp = 1.0 - np.cumprod(1.0 - dv / v0)
    return float(syr_conc_m) * disp


def _theta(xt: np.ndarray, n: float, mt: float, logk: float) -> np.ndarray:
    k = 10.0 ** logk
    nm = max(n * mt, 1e-18)
    r = np.maximum(xt / nm, 0.0)
    a = 1.0 + 1.0 / (k * nm)
    s = np.sqrt(np.maximum((r + a) ** 2 - 4.0 * r, 0.0))
    return 0.5 * (r + a - s)


def _q_one(xt: np.ndarray, n: float, mt: float, logk: float, dh: float,
           v0: float) -> np.ndarray:
    return n * mt * dh * v0 * _theta(xt, n, mt, logk)


def one_site_heat(xt: np.ndarray, dv: np.ndarray, mt: float, v0: float,
                  n: float, logk: float, dh: float, offset: float) -> np.ndarray:
    q = _q_one(xt, n, mt, logk, dh, v0)
    dq = np.empty_like(q)
    dq[0] = q[0]
    dq[1:] = q[1:] - q[:-1] + dv[1:] / v0 * 0.5 * (q[1:] + q[:-1])
    return dq + offset


def two_site_heat(xt: np.ndarray, dv: np.ndarray, mt: float, v0: float,
                  p: np.ndarray) -> np.ndarray:
    n1, lk1, dh1, n2, lk2, dh2, offset = p
    q1 = _q_one(xt, n1, mt, lk1, dh1, v0)
    q2 = _q_one(xt, n2, mt, lk2, dh2, v0)
    q = q1 + q2
    dq = np.empty_like(q)
    dq[0] = q[0]
    dq[1:] = q[1:] - q[:-1] + dv[1:] / v0 * 0.5 * (q[1:] + q[:-1])
    return dq + offset


def dilution_heat(n: int, offset: float) -> np.ndarray:
    return np.full(n, offset, dtype=float)


# --------------------------------------------------------------------------
# deterministic bounded Nelder-Mead in *non-linear* coordinates; dH/offset are
# profiled out by least squares at every trial, which keeps the simplex small.
# --------------------------------------------------------------------------


def _linear_fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return coef, float(resid @ resid)


def _profile_one(z: np.ndarray, xt: np.ndarray, dv: np.ndarray, mt: float,
                 v0: float, y: np.ndarray) -> tuple[float, np.ndarray]:
    n, logk = _unbounded(z, (N_MIN, N_MAX), (LOGK_MIN, LOGK_MAX))
    shape = one_site_heat(xt, dv, mt, v0, n, logk, 1.0, 0.0)
    design = np.column_stack([shape, np.ones_like(y)])
    coef, sse = _linear_fit(design, y)
    return sse, coef


def _profile_two(z: np.ndarray, xt: np.ndarray, dv: np.ndarray, mt: float,
                 v0: float, y: np.ndarray) -> tuple[float, np.ndarray]:
    n1, lk1, n2, lk2 = (
        _scale(z[0], N_MIN, N_MAX), _scale(z[1], LOGK_MIN, LOGK_MAX),
        _scale(z[2], N_MIN, N_MAX), _scale(z[3], LOGK_MIN, LOGK_MAX))
    s1 = one_site_heat(xt, dv, mt, v0, n1, lk1, 1.0, 0.0)
    s2 = one_site_heat(xt, dv, mt, v0, n2, lk2, 1.0, 0.0)
    design = np.column_stack([s1, s2, np.ones_like(y)])
    coef, sse = _linear_fit(design, y)
    return sse, np.array([n1, lk1, coef[0], n2, lk2, coef[1], coef[2]])


def _scale(z: float, lo: float, hi: float) -> float:
    # logistic map onto (lo, hi)
    u = 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
    return lo + (hi - lo) * u


def _unbounded(z: np.ndarray, *bounds: tuple[float, float]):
    return tuple(_scale(z[i], lo, hi) for i, (lo, hi) in enumerate(bounds))


def nelder_mead(fun, x0: np.ndarray, *, budget: int,
                step: float = 0.12, tag: str = "") -> tuple[np.ndarray, float, int, bool]:
    """Deterministic NM. Returns (x, f, evaluations, converged).

    Raises BudgetExhausted if the evaluation budget runs out first.
    """
    dim = x0.size
    simplex = [x0.copy()]
    for i in range(dim):
        v = x0.copy()
        v[i] += step
        simplex.append(v)
    pts = np.asarray(simplex, dtype=float)
    vals = np.array([fun(p)[0] for p in pts], dtype=float)
    used = pts.shape[0]

    def shrink(cent: np.ndarray, worst: np.ndarray, gamma: float) -> np.ndarray:
        return cent + gamma * (worst - cent)

    for _ in range(budget):
        order = np.argsort(vals, kind="mergesort")
        pts, vals = pts[order], vals[order]
        if np.max(np.abs(pts - pts[0])) < 1e-7 and (
                vals[-1] - vals[0]) < 1e-11 * (1.0 + abs(vals[0])):
            return pts[0], float(vals[0]), used, True
        centroid = pts[:-1].mean(axis=0)
        xr = shrink(centroid, pts[-1], -1.0)
        fr = fun(xr)[0]
        used += 1
        if used > budget:
            raise BudgetExhausted(
                f"binding fitter budget exhausted ({tag})", budget=budget, used=used)
        if vals[0] <= fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
            continue
        if fr < vals[0]:
            xe = shrink(centroid, pts[-1], -2.0)
            fe = fun(xe)[0]
            used += 1
            if used > budget:
                raise BudgetExhausted(
                    f"binding fitter budget exhausted ({tag})",
                    budget=budget, used=used)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
            continue
        xc = shrink(centroid, pts[-1], 0.5)
        fc = fun(xc)[0]
        used += 1
        if used > budget:
            raise BudgetExhausted(
                f"binding fitter budget exhausted ({tag})", budget=budget, used=used)
        if fc < vals[-1]:
            pts[-1], vals[-1] = xc, fc
            continue
        # contraction
        best = pts[0]
        for j in range(1, dim + 1):
            pts[j] = best + 0.5 * (pts[j] - best)
            vals[j] = fun(pts[j])[0]
            used += 1
            if used > budget:
                raise BudgetExhausted(
                    f"binding fitter budget exhausted ({tag})",
                    budget=budget, used=used)
    order = np.argsort(vals, kind="mergesort")
    return pts[order[0]], float(vals[order[0]]), used, False


def _bic(sse: float, k: int, n: int, rules: RuleSet) -> float:
    sse_eff = max(float(sse), rules.sse_floor)
    return float(n * math.log(sse_eff / max(n, 1)) + k * math.log(max(n, 2)))


def fit_one_site(xv: list[float], yv: list[float], *, macromol_m: float,
                 syr_conc_m: float, cell_volume_l: float, rules: RuleSet
                 ) -> dict[str, Any]:
    n_pts = len(yv)
    if n_pts < 4:
        raise SemanticImpasse(
            "one-site fit needs at least 4 corrected injections",
            locations=[{"n_points": n_pts, "required": 4}])
    dv = np.asarray(xv, dtype=float)
    y = np.asarray(yv, dtype=float)
    xt = cumulative_ligand(dv, cell_volume_l, syr_conc_m)
    mt = max(float(macromol_m), 1e-18)

    grid_n = [0.05, 0.15, 0.3, 0.6, 1.0, 1.6, 2.5, 4.0, 8.0, 15.0]
    grid_k = [3.0, 4.0, 5.0, 5.75, 6.5, 7.5, 8.5]

    def to_z(n: float, logk: float) -> np.ndarray:
        return np.array([_invscale(n, N_MIN, N_MAX),
                         _invscale(logk, LOGK_MIN, LOGK_MAX)])

    best = None
    for n0 in grid_n:
        for k0 in grid_k:
            z = to_z(n0, k0)
            sse, coef = _profile_one(z, xt, dv, mt, cell_volume_l, y)
            if best is None or sse < best[0]:
                best = (sse, z, coef)
    assert best is not None
    zbest, _, _, _ = nelder_mead(
        lambda z: _profile_one(z, xt, dv, mt, cell_volume_l, y),
        best[1], budget=rules.one_site_budget, tag="one_site")
    sse, coef = _profile_one(zbest, xt, dv, mt, cell_volume_l, y)
    n_fit, logk = _unbounded(zbest, (N_MIN, N_MAX), (LOGK_MIN, LOGK_MAX))
    dh, offset = float(coef[0]), float(coef[1])
    pred = one_site_heat(xt, dv, mt, cell_volume_l, n_fit, logk, dh, offset)
    params = {"n": n_fit, "K": 10.0 ** logk, "logK": logk,
              "dH_cal_per_mol": dh, "offset_cal": offset}
    return _result("one_site", k=4, sse=sse, n_pts=n_pts, params=params,
                   pred=pred, y=y, rules=rules)


def fit_two_site(xv: list[float], yv: list[float], *, macromol_m: float,
                 syr_conc_m: float, cell_volume_l: float, rules: RuleSet
                 ) -> dict[str, Any]:
    n_pts = len(yv)
    if n_pts < 8:
        raise SemanticImpasse(
            "two-site fit needs at least 8 corrected injections",
            locations=[{"n_points": n_pts, "required": 8}])
    dv = np.asarray(xv, dtype=float)
    y = np.asarray(yv, dtype=float)
    xt = cumulative_ligand(dv, cell_volume_l, syr_conc_m)
    mt = max(float(macromol_m), 1e-18)
    starts = [
        (0.2, 4.5, 0.8, 6.5),
        (0.5, 5.5, 0.5, 7.5),
        (1.0, 5.0, 1.0, 7.0),
        (0.1, 3.5, 0.3, 6.0),
        (0.8, 6.0, 0.2, 4.0),
    ]

    def to_z(t):
        n1, l1, n2, l2 = t
        return np.array([_invscale(n1, N_MIN, N_MAX), _invscale(l1, LOGK_MIN, LOGK_MAX),
                         _invscale(n2, N_MIN, N_MAX), _invscale(l2, LOGK_MIN, LOGK_MAX)])

    best = None
    for tup in starts:
        z = to_z(tup)
        sse, coef = _profile_two(z, xt, dv, mt, cell_volume_l, y)
        if best is None or sse < best[0]:
            best = (sse, z, coef)
    zbest, _, _, _ = nelder_mead(
        lambda z: _profile_two(z, xt, dv, mt, cell_volume_l, y),
        best[1], budget=rules.two_site_budget, tag="two_site")
    sse, p = _profile_two(zbest, xt, dv, mt, cell_volume_l, y)
    n1, lk1, dh1, n2, lk2, dh2, offset = (float(v) for v in p)
    pred = two_site_heat(xt, dv, mt, cell_volume_l, p)
    params = {"n1": n1, "K1": 10.0 ** lk1, "logK1": lk1, "dH1_cal_per_mol": dh1,
              "n2": n2, "K2": 10.0 ** lk2, "logK2": lk2, "dH2_cal_per_mol": dh2,
              "offset_cal": offset}
    return _result("two_site", k=7, sse=sse, n_pts=n_pts, params=params,
                   pred=pred, y=y, rules=rules)


def fit_dilution(xv: list[float], yv: list[float], rules: RuleSet
                 ) -> dict[str, Any]:
    y = np.asarray(yv, dtype=float)
    n_pts = y.size
    if n_pts < 2:
        raise SemanticImpasse(
            "dilution fit needs at least 2 corrected injections",
            locations=[{"n_points": n_pts, "required": 2}])
    offset = float(np.mean(y))
    pred = np.full(n_pts, offset)
    sse = float(np.sum((y - pred) ** 2))
    return _result("dilution", k=1, sse=sse, n_pts=n_pts,
                   params={"q_dilution_cal": offset}, pred=pred, y=y,
                   rules=rules)


def _invscale(v: float, lo: float, hi: float) -> float:
    u = (v - lo) / (hi - lo)
    u = min(max(u, 1e-9), 1 - 1e-9)
    return math.log(u / (1 - u))


def _result(name: str, *, k: int, sse: float, n_pts: int,
            params: dict[str, float], pred: np.ndarray, y: np.ndarray,
            rules: RuleSet) -> dict[str, Any]:
    bic = _bic(sse, k, n_pts, rules)
    resid = (y - pred).astype(float)
    return {
        "kind": "fit",
        "model": name,
        "n_params": k,
        "n_points": int(n_pts),
        "sse": float(sse),
        "rmse": float(math.sqrt(max(sse, 0.0) / max(n_pts, 1))),
        "bic": bic,
        "parameters": params,
        "parameter_units": {
            "K": "1/M", "logK": "log10(1/M)",
            "dH_cal_per_mol": "cal/mol", "dH1_cal_per_mol": "cal/mol",
            "dH2_cal_per_mol": "cal/mol", "offset_cal": "cal",
            "q_dilution_cal": "cal", "n": "sites", "n1": "sites",
            "n2": "sites"},
        "predicted_cal": [float(v) for v in pred],
        "observed_cal": [float(v) for v in y],
        "residuals_cal": [float(v) for v in resid],
        "rule_version": rules.version,
    }


def compare(fits: list[dict[str, Any]], rules: RuleSet) -> dict[str, Any]:
    """Rank fits and retain every co-optimal model (ties are kept, not broken)."""
    ordered = sorted(fits, key=lambda f: (f["bic"], f["n_params"], f["model"]))
    best_bic = ordered[0]["bic"]
    ranked = []
    for f in ordered:
        d = f["bic"] - best_bic
        ranked.append({
            "model": f["model"],
            "bic": f["bic"],
            "delta_bic": d,
            "rmse": f["rmse"],
            "co_optimal": d <= rules.tie_bic_delta,
            "fit": f,
        })
    co = [r for r in ranked if r["co_optimal"]]
    return {
        "kind": "model_comparison",
        "tie_bic_delta": rules.tie_bic_delta,
        "ranked": ranked,
        "co_optimal": [r["model"] for r in co],
        "is_tie": len(co) > 1,
        "deterministic_order": [r["model"] for r in ranked],
        "rule_version": rules.version,
    }
