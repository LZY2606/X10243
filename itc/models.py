"""Binding-model candidates: one-site, multi-site, dilution-only.

Everything here is pure and server-side testable.  Fitting is deterministic
NumPy only: for a given binding parameter set the per-injection heats are
linear in the enthalpies (linear least squares), and binding parameters are
searched on fixed grids with coordinate refinement up to a declared
evaluation budget.  Exceeding the budget is BUDGET_EXHAUSTED, a distinct
outcome from semantic failure.
"""

from __future__ import annotations

import math

import numpy as np

from .diagnostics import BUDGET_EXHAUSTED, SEMANTIC_INVALID, diagnostic
from .hashing import digest


# --------------------------------------------------------------------------
# concentration recurrence (Wiseman)
# --------------------------------------------------------------------------

def concentrations(schedule_l, v0_l, c0_m, x0_m):
    """Post-injection cell (c_i macromolecule, x_i ligand) concentrations."""
    dv = np.asarray(schedule_l, dtype=float)
    k = np.arange(1, len(dv) + 1, dtype=float)
    v_i = v0_l + np.cumsum(dv)
    c_i = c0_m * v0_l / v_i
    x_i = x0_m * np.cumsum(dv) / v_i
    return c_i, x_i, v_i


def _fraction_bound(c, x, ka):
    """θ for one site type given total c, x (per-cell-site scaled)."""
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    ka_c = np.maximum(ka * c, 1e-30)
    r = x / np.maximum(c, 1e-30)
    a = 1.0 + r + 1.0 / ka_c
    disc = a * a - 4.0 * r
    disc = np.maximum(disc, 0.0)
    return 0.5 * (a - np.sqrt(disc))


def total_heat_one_site(c_i, x_i, v_i, dv_i, n, ka, dh):
    """Incremental heat q_i (J) for independent one-site binding."""
    theta = _fraction_bound(c_i * n, x_i, ka)
    q_total = c_i * n * dh * v_i * theta
    q_prev = np.concatenate(([0.0], q_total[:-1]))
    # displaced volume carries half the prior titrant out
    v_prev = np.concatenate(([v_i[0] - dv_i[0]], v_i[:-1]))
    q = q_total - (1.0 - 0.5 * dv_i / v_i) * q_prev
    return q


def total_heat_two_site(c_i, x_i, v_i, dv_i, n1, ka1, n2, ka2, dh1, dh2):
    t1 = c_i * n1 * dh1 * v_i * _fraction_bound(c_i * n1, x_i, ka1)
    t2 = c_i * n2 * dh2 * v_i * _fraction_bound(c_i * n2, x_i, ka2)
    q_total = t1 + t2
    q_prev = np.concatenate(([0.0], q_total[:-1]))
    q = q_total - (1.0 - 0.5 * dv_i / v_i) * q_prev
    return q


def dilution_heat(dv_i, v_i, d_dil):
    """Dilution-only: constant heat per unit injected volume (J/L)."""
    return dv_i * d_dil


# --------------------------------------------------------------------------
# fit helpers
# --------------------------------------------------------------------------

def _bic(sse, n_obs, n_params):
    if sse <= 0:
        return -math.inf
    return n_obs * math.log(sse / n_obs) + n_params * math.log(n_obs)


def _linear_fit(A, y):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return coef, float(resid @ resid)


def _stable_candidate(family, params, sse, n_obs, n_params, notes=None):
    params_clean = {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v)
                    for k, v in params.items()}
    return {
        "family": family,
        "sse": sse,
        "bic": _bic(sse, n_obs, n_params),
        "n_params": n_params,
        "params": params_clean,
        "notes": notes or [],
    }


def _tie_groups(cands, tol):
    best = min(c["sse"] for c in cands)
    group = [c for c in cands if c["sse"] <= best * (1.0 + tol) + 1e-30]
    return group, best


def fit_models(normalized, evaluation_budget=4000, tie_tolerance=5e-4):
    """Fit all candidate families to per-injection corrected heats.

    ``normalized`` = list of dicts with keys:
        index, volume_l, cell_c, ligand_c, v_i, q_j (corrected heat, J),
        usable (bool)
    Returns dict with candidates (stable order), tie information and
    diagnostics.  Budget exhaustion is reported as a distinct outcome while
    still returning the best candidates found within budget.
    """
    diags = []
    rows = [r for r in normalized if r.get("usable")]
    if len(rows) < 3:
        diags.append(diagnostic(SEMANTIC_INVALID,
                                f"need >= 3 usable injections for model fit, got {len(rows)}"))
        return {"candidates": [], "tie": False, "tied_families": [],
                "diagnostics": diags, "outcome": SEMANTIC_INVALID}

    dv = np.array([r["volume_l"] for r in rows])
    v0_ref = rows[0]["v_i"] - dv[0]
    c0_ref = rows[0]["cell_c"]
    x0_ref = rows[0].get("syringe_c_ref")
    c_i, x_i, v_i = concentrations(dv, v0_ref, c0_ref,
                                   x0_ref if x0_ref else _infer_x0(rows))
    q = np.array([r["q_j"] for r in rows])
    n_obs = len(q)

    evaluations = 0
    exhausted = False

    # --- dilution-only (linear, exact) ----------------------------------
    d_dil, sse_dil = _linear_fit(dv.reshape(-1, 1), q)
    dilution_candidate = _stable_candidate(
        "dilution", {"d_dil_J_per_L": float(d_dil[0])}, sse_dil, n_obs, 1)
    candidates = [dilution_candidate]
    evaluations += 1

    # --- one site --------------------------------------------------------
    best_one = None
    n_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    log_ka_grid = np.linspace(2.0, 9.0, 29)
    for n in n_grid:
        for log_ka in log_ka_grid:
            ka = 10.0 ** log_ka
            q_shapes = total_heat_one_site(c_i, x_i, v_i, dv, n, ka, 1.0)
            scale = np.array([q_shapes])
            coef, sse = _linear_fit(scale.T, q)
            evaluations += 1
            if best_one is None or sse < best_one[0]:
                best_one = (sse, n, ka, float(coef[0]))
    # coordinate refinement
    sse, n, ka, dh = best_one
    for step in (0.2, 0.08, 0.03, 0.01):
        improved = True
        while improved:
            improved = False
            for n2 in (n - step, n + step):
                if n2 <= 0:
                    continue
                q_shapes = total_heat_one_site(c_i, x_i, v_i, dv, n2, ka, 1.0)
                coef, sse2 = _linear_fit(q_shapes.reshape(-1, 1), q)
                evaluations += 1
                if sse2 < sse:
                    sse, n, dh = sse2, n2, float(coef[0])
                    improved = True
            for ka2 in (ka * 10.0 ** (-step), ka * 10.0 ** step):
                q_shapes = total_heat_one_site(c_i, x_i, v_i, dv, n, ka2, 1.0)
                coef, sse2 = _linear_fit(q_shapes.reshape(-1, 1), q)
                evaluations += 1
                if sse2 < sse:
                    sse, ka, dh = sse2, ka2, float(coef[0])
                    improved = True
            if evaluations > evaluation_budget:
                exhausted = True
                break
        if exhausted:
            break
    q_pred = total_heat_one_site(c_i, x_i, v_i, dv, n, ka, dh)
    candidates.append(_stable_candidate(
        "one_site", {"n": n, "Ka_M-1": ka, "dH_J_per_mol": dh, "dG_J_per_mol": _dg(ka)},
        sse, n_obs, 3, notes=(["fit_truncated_at_budget"] if exhausted else []),
    ))
    one_pred = q_pred

    # --- two independent sites (only with enough data) -------------------
    two_pred = None
    if n_obs >= 6 and not exhausted:
        best_two = None
        for n1 in (0.5, 1.0):
            for log_ka1 in np.linspace(2.5, 7.0, 10):
                for n2 in (0.5, 1.0):
                    for log_ka2 in np.linspace(3.0, 8.0, 11):
                        ka1, ka2 = 10 ** log_ka1, 10 ** log_ka2
                        s1 = total_heat_two_site(c_i, x_i, v_i, dv,
                                                 n1, ka1, n2, ka2, 1.0, 1.0)
                        z1 = total_heat_two_site(c_i, x_i, v_i, dv,
                                                 n1, ka1, 0.0, 1.0, 1.0, 0.0)
                        A = np.column_stack([s1 - z1, z1])
                        coef, sse = _linear_fit(A, q)
                        evaluations += 2
                        if best_two is None or sse < best_two[0]:
                            best_two = (sse, n1, ka1, n2, ka2,
                                        float(coef[0]), float(coef[1]))
                        if evaluations > evaluation_budget:
                            exhausted = True
                            break
                    if exhausted:
                        break
                if exhausted:
                    break
            if exhausted:
                break
        if best_two:
            sse, n1, ka1, n2, ka2, dh1, dh2 = best_two
            two_pred = total_heat_two_site(c_i, x_i, v_i, dv,
                                           n1, ka1, n2, ka2, dh1, dh2)
            notes = ["fit_truncated_at_budget"] if exhausted else []
            if abs(ka1 - ka2) / max(ka1, ka2) < 0.02:
                notes.append("sites_poorly_identified")
            candidates.append(_stable_candidate(
                "two_site",
                {"n1": n1, "Ka1_M-1": ka1, "dH1_J_per_mol": dh1,
                 "n2": n2, "Ka2_M-1": ka2, "dH2_J_per_mol": dh2,
                 "dG1_J_per_mol": _dg(ka1), "dG2_J_per_mol": _dg(ka2)},
                sse, n_obs, 6, notes=notes,
            ))
    elif n_obs < 6:
        diags.append(diagnostic(SEMANTIC_INVALID,
                                "two_site candidate skipped: fewer than 6 usable injections"))

    if exhausted:
        diags.append(diagnostic(BUDGET_EXHAUSTED,
                                f"evaluation budget {evaluation_budget} exhausted; "
                                "returning best candidates found within budget",
                                evaluations=evaluations))

    # stable ordering: SSE then family then canonical params
    candidates.sort(key=lambda c: (round_sse(c["sse"]), c["family"],
                                   _param_key(c["params"])))
    tied, best = _tie_groups(candidates, tie_tolerance)
    tied_families = sorted({c["family"] for c in tied})
    return {
        "candidates": candidates,
        "predicted": {
            "index": [r["index"] for r in rows],
            "one_site": list(map(float, one_pred)),
            "two_site": (list(map(float, two_pred)) if two_pred is not None else None),
            "dilution": list(map(float, dv * float(d_dil[0]))),
            "observed": list(map(float, q)),
        },
        "tie": len(tied_families) > 1,
        "tied_families": tied_families,
        "best_sse": best,
        "evaluations": evaluations,
        "outcome": BUDGET_EXHAUSTED if exhausted else "OK",
        "diagnostics": diags,
    }


def round_sse(x):
    return f"{x:.10g}"


def _param_key(params):
    from .hashing import canonical
    return canonical(params)


def _dg(ka, temperature_k=298.15):
    return float(-8.314462618 * temperature_k * math.log(max(ka, 1e-30)))


def _infer_x0(rows):
    # back out nominal syringe concentration from first two injections
    dv = [r["volume_l"] for r in rows]
    v_i = [r["v_i"] for r in rows]
    return rows[0]["ligand_c"] * v_i[0] / dv[0]


def fit_identity(normalized, rule_id_str, rule_body):
    """Stable identity of a fit result: inputs + rule version + parent heats."""
    summary = {
        "inputs": [
            {"index": r["index"], "volume_l": r["volume_l"],
             "cell_c": r["cell_c"], "ligand_c": r["ligand_c"],
             "v_i": r["v_i"], "q_j": _r(r["q_j"]), "usable": r["usable"]}
            for r in normalized
        ],
        "rule_id": rule_id_str,
        "rule_body": rule_body,
    }
    return digest("fit", summary)


def _r(x, nd=12):
    return None if x is None else round(float(x), nd)
