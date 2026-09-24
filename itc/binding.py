"""Binding-model candidates: one-site, two-site (multi-site), dilution-only.

Fits are deterministic (fixed grids and seeds) so equivalent inputs always
produce the same candidates in the same stable order. Distinct failure
outcomes: 'budget_exhausted' (resource limit), 'unsatisfiable' (semantically
impossible inputs). Format corruption is detected at import, not here.
"""
import numpy as np

PARAM_VERSION = "binding/1.0"
EQUAL_RSS_TOL = 0.05  # models within 5% RSS of the best are "equally good"


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def tick(self, n=1):
        self.used += n
        return self.limit is None or self.used <= self.limit


def _complex_conc(Ka, n, M, X):
    """Bound-ligand concentration (M) for one independent site set."""
    a = n * M + X + 1.0 / Ka
    disc = np.maximum(a * a - 4.0 * n * M * X, 0.0)
    return 0.5 * (a - np.sqrt(disc))


def _cell_state(M0, Xs, V0, dV):
    """Macromolecule/ligand totals in the cell after each injection."""
    f = np.cumprod(1.0 - np.asarray(dV) / V0)
    M = M0 * f
    X = Xs * (1.0 - f)
    return M, X, f


def _per_injection(Q, f):
    """Cumulative heats -> per-injection heats with displacement correction."""
    q = np.empty_like(Q)
    prev = 0.0
    for i in range(len(Q)):
        prev_diluted = prev * (f[i - 1] / f[i]) if i > 0 else 0.0
        # f[i-1]/f[i] = 1/(1-dv_i/V0); heat remaining after displacement:
        prev_remaining = prev * (f[i] / f[i - 1]) if i > 0 else 0.0
        q[i] = Q[i] - prev_remaining
        prev = Q[i]
    return q


def heats_one_site(Ka, dH_J, n, M0, Xs, V0, dV):
    M, X, f = _cell_state(M0, Xs, V0, dV)
    Q = V0 * dH_J * _complex_conc(Ka, n, M, X)
    return _per_injection(Q, f)


def heats_two_site(Ka1, dH1_J, n1, Ka2, dH2_J, n2, M0, Xs, V0, dV):
    M, X, f = _cell_state(M0, Xs, V0, dV)
    Q = V0 * (dH1_J * _complex_conc(Ka1, n1, M, X)
              + dH2_J * _complex_conc(Ka2, n2, M, X))
    return _per_injection(Q, f)


def _to_kJ_per_mol(q_J, Xs, dV):
    return q_J / (np.asarray(dV) * Xs) / 1000.0


def _rss(y, yhat):
    r = np.asarray(y) - np.asarray(yhat)
    return float(r @ r)


def _linear_dH(q_per_dH, y):
    """Least-squares scale for enthalpy (heat is linear in dH)."""
    A = np.asarray(q_per_dH)
    denom = float(A @ A)
    if denom == 0.0:
        return 0.0
    return float(A @ np.asarray(y)) / denom


def fit_one_site(M0, Xs, V0, dV, y, budget=None):
    b = Budget(budget)
    best = None
    for logK in np.linspace(2.0, 9.0, 141):
        for n in np.linspace(0.3, 2.0, 35):
            if not b.tick():
                return _budget_result(best, b)
            Ka = 10.0 ** logK
            q_unit = _to_kJ_per_mol(heats_one_site(Ka, 1.0, n, M0, Xs, V0, dV), Xs, dV)
            dH = _linear_dH(q_unit, y)
            rss = _rss(y, dH * q_unit)
            if best is None or rss < best["rss"]:
                best = {"logK": float(logK), "n": float(n), "dH": dH, "rss": rss}
    # deterministic coordinate refinement
    step = {"logK": 0.05, "n": 0.02}
    for _ in range(60):
        improved = False
        for key in ("logK", "n"):
            for sign in (1.0, -1.0):
                if not b.tick():
                    return _budget_result(best, b)
                cand = dict(best)
                cand[key] = best[key] + sign * step[key]
                Ka = 10.0 ** cand["logK"]
                q_unit = _to_kJ_per_mol(
                    heats_one_site(Ka, 1.0, cand["n"], M0, Xs, V0, dV), Xs, dV)
                dH = _linear_dH(q_unit, y)
                rss = _rss(y, dH * q_unit)
                if rss < best["rss"]:
                    best = {"logK": cand["logK"], "n": cand["n"], "dH": dH, "rss": rss}
                    improved = True
        if not improved:
            step["logK"] *= 0.5
            step["n"] *= 0.5
            if step["logK"] < 1e-4:
                break
    return {"status": "ok", "model": "one_site",
            "params": {"Ka": 10.0 ** best["logK"], "dH_kJ_mol": best["dH"] / 1000.0,
                       "n": best["n"]},
            "rss": best["rss"], "evals": b.used}


def _two_site_rss(logK1, n1, logK2, n2, M0, Xs, V0, dV, y):
    q1 = _to_kJ_per_mol(heats_one_site(10.0 ** logK1, 1.0, n1, M0, Xs, V0, dV), Xs, dV)
    q2 = _to_kJ_per_mol(heats_one_site(10.0 ** logK2, 1.0, n2, M0, Xs, V0, dV), Xs, dV)
    A = np.stack([q1, q2])
    denom = A @ A.T
    rhs = A @ np.asarray(y)
    try:
        dH = np.linalg.solve(denom + 1e-12 * np.eye(2), rhs)
    except np.linalg.LinAlgError:
        return None
    rss = _rss(y, dH @ A)
    return rss, float(dH[0]), float(dH[1])


def fit_two_site(M0, Xs, V0, dV, y, budget=None, seed=None):
    b = Budget(budget)
    seed = seed or {"logK1": 5.5, "n1": 1.0, "logK2": 3.5, "n2": 1.0}
    cur = dict(seed)
    r = _two_site_rss(cur["logK1"], cur["n1"], cur["logK2"], cur["n2"],
                      M0, Xs, V0, dV, y)
    if r is None:
        return {"status": "unsatisfiable", "model": "two_site",
                "diagnostics": [{"code": "singular_fit", "path": "two_site",
                                 "message": "linear system singular",
                                 "severity": "error"}]}
    best = {"rss": r[0], "dH1": r[1], "dH2": r[2], **cur}
    steps = {"logK1": 0.3, "n1": 0.1, "logK2": 0.3, "n2": 0.1}
    for _ in range(200):
        improved = False
        for key in ("logK1", "n1", "logK2", "n2"):
            for sign in (1.0, -1.0):
                if not b.tick():
                    return _budget_result(best, b, model="two_site")
                cand = dict(best)
                cand[key] = cand[key] + sign * steps[key]
                if cand["n1"] <= 0.05 or cand["n2"] <= 0.05:
                    continue
                r = _two_site_rss(cand["logK1"], cand["n1"], cand["logK2"],
                                  cand["n2"], M0, Xs, V0, dV, y)
                if r is not None and r[0] < best["rss"]:
                    best = {"rss": r[0], "dH1": r[1], "dH2": r[2], **cand}
                    improved = True
        if not improved:
            for key in steps:
                steps[key] *= 0.5
            if steps["logK1"] < 1e-3:
                break
    return {"status": "ok", "model": "two_site",
            "params": {"Ka1": 10.0 ** best["logK1"], "dH1_kJ_mol": best["dH1"] / 1000.0,
                       "n1": best["n1"], "Ka2": 10.0 ** best["logK2"],
                       "dH2_kJ_mol": best["dH2"] / 1000.0, "n2": best["n2"]},
            "rss": best["rss"], "evals": b.used}


def fit_dilution(y, budget=None):
    b = Budget(budget)
    if not b.tick():
        return _budget_result(None, b, model="dilution")
    c = float(np.mean(y))
    return {"status": "ok", "model": "dilution",
            "params": {"constant_kJ_mol": c},
            "rss": _rss(y, np.full_like(np.asarray(y), c)), "evals": b.used}


def _budget_result(best, b, model="one_site"):
    return {"status": "budget_exhausted", "model": model,
            "partial": best, "evals": b.used,
            "diagnostics": [{"code": "budget_exhausted", "path": model,
                             "message": f"evaluation budget exhausted after "
                                        f"{b.used} evaluations",
                             "severity": "error"}]}


def model_curve(model, params, M0, Xs, V0, dV):
    """Per-injection fitted heats (kJ/mol of injectant)."""
    if model == "one_site":
        q = heats_one_site(params["Ka"], params["dH_kJ_mol"] * 1000.0,
                           params["n"], M0, Xs, V0, dV)
    elif model == "two_site":
        q = heats_two_site(params["Ka1"], params["dH1_kJ_mol"] * 1000.0,
                           params["n1"], params["Ka2"],
                           params["dH2_kJ_mol"] * 1000.0, params["n2"],
                           M0, Xs, V0, dV)
    elif model == "dilution":
        q = np.full(len(dV), params["constant_kJ_mol"] * 1000.0)
    else:
        raise ValueError(model)
    return _to_kJ_per_mol(q, Xs, dV)


def fit_all(M0, Xs, V0, dV, y, budget=None):
    """Fit all candidate models; keep every equally-good candidate.

    Returns (candidates, diagnostics, status). Candidates are sorted by
    (rss, model name, rounded params) so equivalent inputs give a stable
    order. status is 'ok' unless every model failed.
    """
    y = np.asarray(y, dtype=float)
    diags = []
    if len(y) == 0:
        return [], [{"code": "no_fittable_injections", "path": "injections",
                     "message": "no injections available for fitting",
                     "severity": "error"}], "unsatisfiable"
    if not np.isfinite(V0) or V0 <= 0:
        return [], [{"code": "cell_volume_invalid", "path": "cell_volume",
                     "message": f"cell volume {V0} is not positive; cannot "
                                f"compute molar ratios", "severity": "error"},
                    ], "unsatisfiable"
    if not np.isfinite(Xs) or Xs <= 0 or not np.isfinite(M0) or M0 <= 0:
        return [], [{"code": "concentration_invalid", "path": "concentrations",
                     "message": "concentrations must be positive",
                     "severity": "error"}], "unsatisfiable"

    one = fit_one_site(M0, Xs, V0, dV, y, budget=budget)
    seeds = [None]
    if one["status"] == "ok":
        # deterministic seed from the one-site solution: lets a degenerate
        # two-site model converge to (nearly) the same minimum
        import math
        seeds.append({"logK1": math.log10(one["params"]["Ka"]),
                      "n1": one["params"]["n"],
                      "logK2": math.log10(one["params"]["Ka"]) - 1.0,
                      "n2": 0.3})
    two_fits = [fit_two_site(M0, Xs, V0, dV, y, budget=budget, seed=sd)
                for sd in seeds]
    two_ok = [f for f in two_fits if f["status"] == "ok"]
    two = min(two_ok, key=lambda f: f["rss"]) if two_ok else two_fits[0]
    fits = [one, two, fit_dilution(y, budget=budget)]
    ok = [f for f in fits if f["status"] == "ok"]
    for f in fits:
        diags.extend(f.get("diagnostics", []))
    if not ok:
        status = ("budget_exhausted" if any(f["status"] == "budget_exhausted"
                                            for f in fits) else "unsatisfiable")
        return [], diags, status

    best_rss = min(f["rss"] for f in ok)
    candidates = []
    for f in ok:
        yhat = model_curve(f["model"], f["params"], M0, Xs, V0, dV)
        equally = f["rss"] <= best_rss * (1.0 + EQUAL_RSS_TOL)
        candidates.append({
            "model": f["model"], "params": f["params"], "rss": f["rss"],
            "equally_good": equally,
            "yhat_kJ_mol": [float(v) for v in yhat],
            "residuals_kJ_mol": [float(a - b) for a, b in zip(y, yhat)],
        })
    candidates.sort(key=lambda c: (
        c["rss"], c["model"],
        tuple(round(float(v), 6) for v in c["params"].values())))
    if sum(1 for c in candidates if c["equally_good"]) > 1:
        diags.append({
            "code": "multiple_equally_good_models",
            "path": "models",
            "message": "more than one model within RSS tolerance; all "
                       "candidates retained for human decision",
            "severity": "info"})
    return candidates, diags, "ok"
