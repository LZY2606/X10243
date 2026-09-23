"""Binding-model candidates: one-site, two-site (multi-site) and a
dilution-only hypothesis, fitted with a NumPy-free-budget Nelder-Mead.

Equilibria are solved exactly per injection (free ligand by bisection on
the mass balance), with cell-content displacement between injections.
Fitting is budgeted: exceeding the evaluation budget yields the distinct
status BUDGET_EXHAUSTED, never a silently truncated "success".
"""

import math

from .errors import OK, BUDGET_EXHAUSTED

MODEL_KINDS = ("one_site", "two_site", "dilution")

DEFAULT_X0 = {
    "one_site": [5.0, -30.0, 1.0],          # log10(K/mM^-1), dH kJ/mol, n
    "two_site": [2.0, -20.0, 1.0, 4.0, -5.0],  # log10K1, dH1, n1, log10K2, dH2
    "dilution": [0.0, -1.0],                # a uJ, b uJ/uL
}
DEFAULT_STEP = {
    "one_site": [0.5, 5.0, 0.1],
    "two_site": [0.5, 5.0, 0.1, 0.5, 5.0],
    "dilution": [0.5, 0.5],
}


def _free_ligand(sites, xt):
    """Solve x + sum(S*K*x/(1+K*x)) = xt for free ligand x (bisection)."""
    if xt <= 0.0:
        return 0.0
    lo, hi = 0.0, xt
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f = mid + sum(S * K * mid / (1.0 + K * mid) for K, S in sites) - xt
        if f > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def injection_heats(kind, theta, cell_mM, syr_mM, V0_uL, volumes_uL):
    """Simulated per-injection heats in uJ."""
    if kind == "dilution":
        return [theta[0] + theta[1] * v for v in volumes_uL]
    if kind == "one_site":
        Ks = [10.0 ** theta[0]]
        dH = [theta[1]]
        ns = [theta[2]]
    elif kind == "two_site":
        Ks = [10.0 ** theta[0], 10.0 ** theta[3]]
        dH = [theta[1], theta[4]]
        ns = [theta[2], 1.0]
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    V0 = V0_uL / 1000.0  # mL
    mt = cell_mM * V0    # umol macromolecule in cell
    xt = 0.0             # umol ligand in cell
    prev = [0.0] * len(Ks)
    heats = []
    for v in volumes_uL:
        vmL = v / 1000.0
        f = (V0 - vmL) / V0  # displaced fraction
        mt *= f
        xt = xt * f + syr_mM * vmL
        sites = [(K, n * mt) for K, n in zip(Ks, ns)]
        x = _free_ligand(sites, xt)
        bound = [S * K * x / (1.0 + K * x) for (K, S) in sites]
        q = sum(d * (b - pb * f) for d, b, pb in zip(dH, bound, prev)) * 1000.0
        heats.append(q)
        prev = bound
    return heats


class _BudgetHit(Exception):
    pass


def nelder_mead(f, x0, step):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        x[i] += step[i]
        simplex.append(x)
    vals = [f(x) for x in simplex]
    for _ in range(600):
        order = sorted(range(n + 1), key=lambda i: (vals[i], i))
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        spread = max(abs(vals[i] - vals[0]) for i in range(1, n + 1))
        size = max(abs(simplex[i][j] - simplex[0][j])
                   for i in range(1, n + 1) for j in range(n))
        if spread < 1e-10 and size < 1e-7:
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        refl = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [centroid[j] + 2.0 * (centroid[j] - worst[j]) for j in range(n)]
            fe = f(exp)
            simplex[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = refl, fr
        else:
            con = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            fc = f(con)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = con, fc
            else:
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j])
                                  for j in range(n)]
                    vals[i] = f(simplex[i])
    best_i = min(range(n + 1), key=lambda i: (vals[i], i))
    return simplex[best_i], vals[best_i]


def fit_model(kind, obs_heats, volumes, cell_mM, syr_mM, V0_uL,
              budget=20000, x0=None, step=None):
    """Fit one candidate. Budget exhaustion is a distinct result status."""
    evals = [0]

    def chi2(theta):
        evals[0] += 1
        if evals[0] > budget:
            raise _BudgetHit()
        try:
            sim = injection_heats(kind, theta, cell_mM, syr_mM, V0_uL, volumes)
        except (OverflowError, ValueError):
            return 1e30
        val = sum((o - s) ** 2 for o, s in zip(obs_heats, sim))
        return val if math.isfinite(val) else 1e30

    try:
        theta, best = nelder_mead(chi2, x0 or DEFAULT_X0[kind],
                                  step or DEFAULT_STEP[kind])
    except _BudgetHit:
        return {"kind": kind, "status": BUDGET_EXHAUSTED,
                "params": None, "chi2": None, "evals": evals[0]}
    sim = injection_heats(kind, theta, cell_mM, syr_mM, V0_uL, volumes)
    residuals = [o - s for o, s in zip(obs_heats, sim)]
    return {"kind": kind, "status": OK, "params": list(theta),
            "chi2": best, "residuals": residuals, "evals": evals[0]}


EQUAL_TOL = 0.02  # relative chi2 tolerance for "equally good" candidates


def fit_all(obs_heats, volumes, cell_mM, syr_mM, V0_uL, budget=20000):
    """Fit every candidate kind; flag the equally-good set (all retained)."""
    per_kind_budget = max(50, budget // len(MODEL_KINDS))
    candidates = []
    one = fit_model("one_site", obs_heats, volumes, cell_mM, syr_mM, V0_uL,
                    budget=per_kind_budget)
    candidates.append(one)
    # two-site is seeded both generically and from the one-site solution,
    # the way an analyst would; the better of the two starts is kept.
    starts = [None]
    if one["status"] == OK:
        starts.append([one["params"][0], one["params"][1], one["params"][2],
                       -2.0, 0.0])  # inert second site: starts at one-site optimum
    two_best = None
    for st in starts:
        c = fit_model("two_site", obs_heats, volumes, cell_mM, syr_mM, V0_uL,
                      budget=per_kind_budget // max(1, len(starts)), x0=st)
        if two_best is None or (c["status"] == OK and (
                two_best["status"] != OK or c["chi2"] < two_best["chi2"])):
            two_best = c
    candidates.append(two_best)
    candidates.append(fit_model("dilution", obs_heats, volumes, cell_mM,
                                syr_mM, V0_uL, budget=per_kind_budget))
    ok = [c for c in candidates if c["status"] == OK]
    best = min((c["chi2"] for c in ok), default=None)
    for c in candidates:
        c["equally_good"] = bool(
            best is not None and c["status"] == OK
            and c["chi2"] <= best * (1.0 + EQUAL_TOL) + 1e-12)
    # stable ordering: chi2 (None last), then kind name
    candidates.sort(key=lambda c: (c["chi2"] is None, c["chi2"] or 0.0, c["kind"]))
    return candidates
