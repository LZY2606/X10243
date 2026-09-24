"""Binding model candidates: dilution-only, one-site, two-site.

Pure NumPy/math implementation (Nelder-Mead, no scipy) so the service and
storage layers stay directly testable.  ``fit_candidates`` keeps ALL legal
candidates when the fit is not uniquely decidable: every model whose SSE is
within tolerance of the best is flagged ``equally_good`` and retained.
"""

import math

MODELS = ("dilution", "one_site", "two_site")


def _bound(mt, xt, n, K):
    """Bound ligand concentration (mol/L) for one set of sites."""
    a = n * mt + xt + 1.0 / K
    disc = a * a - 4.0 * n * mt * xt
    return 0.5 * (a - math.sqrt(max(disc, 0.0)))


def per_injection_J(model, params, volumes_uL, mt0_mM, xs_mM, v0_mL):
    """Per-injection heat in J for a model, with cell-volume displacement."""
    V0 = v0_mL * 1e-3
    xs = xs_mM * 1e-3
    mt = mt0_mM * 1e-3
    xt = 0.0
    qc_prev = 0.0
    out = []
    for dv_uL in volumes_uL:
        dv = dv_uL * 1e-6
        mt *= (1 - dv / V0)
        xt = xt * (1 - dv / V0) + xs * (dv / V0)
        if model == "dilution":
            qc = qc_prev + params[0]
        elif model == "one_site":
            K, dH, n = params
            qc = _bound(mt, xt, n, K) * dH * V0
        elif model == "two_site":
            K1, dH1, n1, K2, dH2, n2 = params
            qc = (_bound(mt, xt, n1, K1) * dH1 + _bound(mt, xt, n2, K2) * dH2) * V0
        else:
            raise ValueError(f"unknown model {model!r}")
        dq = qc - qc_prev + (dv / (2 * V0)) * (qc + qc_prev)
        out.append(dq)
        qc_prev = qc
    return out


def to_kJ_per_mol(dq_J, volumes_uL, xs_mM):
    return [q * 1e6 / (v * xs_mM) for q, v in zip(dq_J, volumes_uL)]


def _nelder_mead(f, x0, steps, max_evals):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        x[i] += steps[i]
        simplex.append(x)
    vals = [f(x) for x in simplex]
    evals = len(simplex)
    exhausted = False
    for _ in range(100000):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= 1e-14 * (1 + abs(vals[0])):
            break
        if evals >= max_evals:
            exhausted = True
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        xr = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = f(xr)
        evals += 1
        if fr < vals[0]:
            xe = [centroid[j] + 2 * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            evals += 1
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = [centroid[j] + 0.5 * (xr[j] - centroid[j]) for j in range(n)]
            else:
                xc = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            evals += 1
            if fc < min(fr, vals[-1]):
                simplex[-1], vals[-1] = xc, fc
            else:
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j]) for j in range(n)]
                    vals[i] = f(simplex[i])
                    evals += 1
    bi = min(range(n + 1), key=lambda i: vals[i])
    return simplex[bi], vals[bi], evals, exhausted


def _fit_one_site(obs, volumes, mt0, xs, v0, budget, evals):
    def f(x):
        n = x[2]
        if n <= 0.01 or n > 20:
            return 1e30
        dq = per_injection_J("one_site", (10 ** x[0], x[1] * 1e4, n), volumes, mt0, xs, v0)
        m = to_kJ_per_mol(dq, volumes, xs)
        return sum((o - mi) ** 2 for o, mi in zip(obs, m))

    best, exhausted = None, False
    starts = [(lk, dh, 1.0) for lk in (3.0, 4.0, 5.0, 6.0, 7.0)
              for dh in (-4.0, -1.0, 1.0, 4.0)]
    for s in starts:
        remaining = budget - evals
        if remaining <= 0:
            exhausted = True
            break
        x, fx, ev, ex = _nelder_mead(f, list(s), [0.5, 1.0, 0.2], min(150, remaining))
        evals += ev
        exhausted = exhausted or ex
        if best is None or fx < best[1]:
            best = (x, fx)
    if best is None:
        return None, evals, True
    x, fx = best
    params = {"K": 10 ** x[0], "dH_J_mol": x[1] * 1e4, "n": x[2]}
    dq = per_injection_J("one_site", (params["K"], params["dH_J_mol"], params["n"]),
                         volumes, mt0, xs, v0)
    m = to_kJ_per_mol(dq, volumes, xs)
    resid = [o - mi for o, mi in zip(obs, m)]
    return {"model": "one_site", "params": params, "sse": fx,
            "residuals": resid, "evals": evals}, evals, exhausted


def _fit_two_site(obs, volumes, mt0, xs, v0, budget, evals):
    def f(x):
        n1, n2 = x[2], x[5]
        if n1 <= 0.01 or n2 <= 0.01 or n1 > 20 or n2 > 20:
            return 1e30
        dq = per_injection_J("two_site",
                             (10 ** x[0], x[1] * 1e4, n1, 10 ** x[3], x[4] * 1e4, n2),
                             volumes, mt0, xs, v0)
        m = to_kJ_per_mol(dq, volumes, xs)
        return sum((o - mi) ** 2 for o, mi in zip(obs, m))

    best, exhausted = None, False
    starts = [(4.0, -3.0, 1.0, 6.0, -1.0, 1.0),
              (5.0, -2.0, 1.0, 6.0, -2.0, 1.0),
              (4.0, -1.0, 1.0, 7.0, -3.0, 1.0),
              (6.0, -3.0, 1.0, 3.0, -1.0, 1.0)]
    for s in starts:
        remaining = budget - evals
        if remaining <= 0:
            exhausted = True
            break
        x, fx, ev, ex = _nelder_mead(f, list(s), [0.5, 1.0, 0.2, 0.5, 1.0, 0.2],
                                     min(300, remaining))
        evals += ev
        exhausted = exhausted or ex
        if best is None or fx < best[1]:
            best = (x, fx)
    if best is None:
        return None, evals, True
    x, fx = best
    params = {"K1": 10 ** x[0], "dH1_J_mol": x[1] * 1e4, "n1": x[2],
              "K2": 10 ** x[3], "dH2_J_mol": x[4] * 1e4, "n2": x[5]}
    dq = per_injection_J("two_site",
                         (params["K1"], params["dH1_J_mol"], params["n1"],
                          params["K2"], params["dH2_J_mol"], params["n2"]),
                         volumes, mt0, xs, v0)
    m = to_kJ_per_mol(dq, volumes, xs)
    resid = [o - mi for o, mi in zip(obs, m)]
    return {"model": "two_site", "params": params, "sse": fx,
            "residuals": resid, "evals": evals}, evals, exhausted


def fit_candidates(obs, volumes, mt0, xs, v0, models=MODELS,
                   budget=6000, equal_rel=0.05, equal_abs=1e-4):
    """Fit all candidate models.  Distinct failure statuses:
    ``unsatisfiable`` (no usable observations), ``budget_exhausted``
    (resource budget ran out), ``ok``.
    """
    obs = [float(o) for o in obs]
    if len(obs) == 0:
        return {"status": "unsatisfiable", "candidates": [], "evals": 0,
                "diagnostics": [{"path": "observations", "code": "no_usable_injections",
                                 "message": "no usable injections after exclusions"}]}
    evals = 0
    cands = []
    exhausted = False
    for model in models:
        if model == "dilution":
            w = [1e6 / (v * xs) for v in volumes]
            c = sum(o * wi for o, wi in zip(obs, w)) / sum(wi * wi for wi in w)
            resid = [o - c * wi for o, wi in zip(obs, w)]
            cands.append({"model": "dilution", "params": {"q_inj_J": c},
                          "sse": sum(r * r for r in resid), "residuals": resid, "evals": 0})
        elif model == "one_site":
            cand, evals, ex = _fit_one_site(obs, volumes, mt0, xs, v0, budget, evals)
            exhausted = exhausted or ex
            if cand is not None:
                cands.append(cand)
        elif model == "two_site":
            cand, evals, ex = _fit_two_site(obs, volumes, mt0, xs, v0, budget, evals)
            exhausted = exhausted or ex
            if cand is not None:
                cands.append(cand)
    if not cands:
        return {"status": "budget_exhausted", "candidates": [], "evals": evals,
                "diagnostics": [{"path": "budget", "code": "budget_exhausted",
                                 "message": f"evaluation budget {budget} exhausted"}]}
    cands.sort(key=lambda c: (c["sse"], c["model"]))  # stable order for equivalent inputs
    best_sse = cands[0]["sse"]
    for c in cands:
        c["equally_good"] = c["sse"] <= max(best_sse * (1 + equal_rel), best_sse + equal_abs)
    return {"status": "budget_exhausted" if exhausted else "ok",
            "candidates": cands, "evals": evals, "diagnostics": []}
