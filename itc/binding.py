"""Binding-model candidates and bounded-budget fitting (pure NumPy).

Models compared by the reviewer:
- one_site:  single set of identical sites (Wiseman-style isotherm)
- two_site:  two independent site classes
- dilution:  constant heat per injection (dilution-only hypothesis)

Fit outcomes are distinct: ok / budget_exhausted / semantic_unsolvable.
"""
import numpy as np

FIT_PARAM_VERSION = "itc-fit/1.0"
EQUAL_SSE_RTOL = 0.05


def _bound_site(Mt, Xt, n, K):
    S = n * Mt
    b = K * (S + Xt) + 1.0
    disc = np.maximum(b * b - 4.0 * K * K * S * Xt, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * K)


def _site_heats(vols, M0, Xs, V0, n, K, dH_J):
    M = M0
    X = 0.0
    prev = 0.0
    out = []
    for v in vols:
        f = 1.0 - v / V0
        M *= f
        X = X * f + Xs * v / V0
        xb = _bound_site(M, X, n, K)
        out.append((xb - prev * f) * 1e-6 * V0 * 1e-6 * dH_J * 1e6)  # uJ
        prev = xb
    return np.array(out)


def model_heats(name, params, vols, M0, Xs, V0):
    if name == "one_site":
        return _site_heats(vols, M0, Xs, V0, params["n"], params["K"], params["dH"])
    if name == "two_site":
        return (_site_heats(vols, M0, Xs, V0, params["n1"], params["K1"], params["dH1"])
                + _site_heats(vols, M0, Xs, V0, params["n2"], params["K2"], params["dH2"]))
    if name == "dilution":
        return np.full(len(vols), params["q0"])
    raise ValueError(f"unknown model {name!r}")


def molar_ratios(vols, M0, Xs, V0):
    """Total ligand / macromolecule ratio after each injection (for plots)."""
    M = M0
    X = 0.0
    out = []
    for v in vols:
        f = 1.0 - v / V0
        M *= f
        X = X * f + Xs * v / V0
        out.append(X / M if M > 0 else 0.0)
    return out


def nelder_mead(func, x0, steps, max_evals, tol=1e-9):
    x0 = np.asarray(x0, float)
    n = x0.size
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] = x[i] + steps[i]
        simplex.append(x)
    simplex = np.array(simplex)
    f = np.array([func(x) for x in simplex])
    evals = n + 1
    converged = False
    while evals < max_evals:
        order = np.argsort(f)
        simplex = simplex[order]
        f = f[order]
        if np.max(np.abs(simplex[1:] - simplex[0])) < tol:
            converged = True
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = func(xr)
        evals += 1
        if fr < f[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = func(xe)
            evals += 1
            if fe < fr:
                simplex[-1], f[-1] = xe, fe
            else:
                simplex[-1], f[-1] = xr, fr
        elif fr < f[-2]:
            simplex[-1], f[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            evals += 1
            if fc < f[-1]:
                simplex[-1], f[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f[i] = func(simplex[i])
                    evals += 1
                    if evals >= max_evals:
                        break
    order = np.argsort(f)
    simplex = simplex[order]
    f = f[order]
    return simplex[0], float(f[0]), evals, converged


_X0 = {
    "one_site": [1.0, 4.0, -4.0],
    "two_site": [4.3, -4.0, 3.0, -0.5],
    "dilution": [0.0],
}
_STEPS = {
    "one_site": [0.2, 0.5, 1.0],
    "two_site": [0.4, 1.0, 0.4, 1.0],
    "dilution": [1.0],
}
_MIN_POINTS = {"one_site": 3, "two_site": 6, "dilution": 1}


def _unpack(name, x):
    if name == "one_site":
        return {"n": float(x[0]), "K": float(10.0 ** x[1]), "dH": float(x[2]) * 1e4}
    if name == "two_site":
        return {"n1": 1.0, "K1": float(10.0 ** x[0]), "dH1": float(x[1]) * 1e4,
                "n2": 1.0, "K2": float(10.0 ** x[2]), "dH2": float(x[3]) * 1e4}
    return {"q0": float(x[0])}


def fit_model(name, obs, vols, M0, Xs, V0, max_evals=4000):
    """Fit one model. Returns a result dict with a distinct status."""
    obs = np.asarray(obs, float)
    if len(obs) < _MIN_POINTS[name]:
        return {"model": name, "status": "semantic_unsolvable", "params": None,
                "sse": None, "evals": 0,
                "diagnostics": [{"code": "too_few_points",
                                 "message": f"{name} needs >= {_MIN_POINTS[name]} "
                                            f"points, got {len(obs)}",
                                 "location": f"model[{name}]"}]}

    def sse(x):
        if name == "one_site" and x[0] <= 0.01:
            return 1e18
        pred = model_heats(name, _unpack(name, x), vols, M0, Xs, V0)
        if not np.all(np.isfinite(pred)):
            return 1e18
        return float(np.sum((pred - obs) ** 2))

    x, f, evals, converged = nelder_mead(sse, _X0[name], _STEPS[name], max_evals)
    if not converged:
        return {"model": name, "status": "budget_exhausted", "params": None,
                "sse": None, "evals": evals,
                "diagnostics": [{"code": "budget_exhausted",
                                 "message": f"fit did not converge within "
                                            f"{max_evals} evaluations",
                                 "location": f"model[{name}]"}]}
    return {"model": name, "status": "ok", "params": _unpack(name, x),
            "sse": f, "evals": evals, "diagnostics": []}
