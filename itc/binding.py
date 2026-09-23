import numpy as np


class BudgetExhausted(Exception):
    pass


def cell_concs(M0, Xs, V0_mL, vols_uL):
    vols = np.asarray(vols_uL, dtype=float)
    f = vols / 1000.0 / V0_mL
    mt = np.empty(len(vols))
    xt = np.empty(len(vols))
    m = M0
    x = 0.0
    for i in range(len(vols)):
        m = m * (1.0 - f[i])
        x = x * (1.0 - f[i]) + Xs * f[i]
        mt[i] = m
        xt[i] = x
    return mt, xt


def injection_heats_uJ(sites, M0, Xs, V0_mL, vols_uL):
    vols = np.asarray(vols_uL, dtype=float)
    f = vols / 1000.0 / V0_mL
    mt, xt = cell_concs(M0, Xs, V0_mL, vols)
    qcum = np.zeros(len(vols))
    for (n, K, dH) in sites:
        a = 1.0 + K * (n * mt + xt)
        disc = np.maximum(a * a - 4.0 * K * K * n * mt * xt, 0.0)
        y = (a - np.sqrt(disc)) / (2.0 * K)
        qcum = qcum + 1000.0 * V0_mL * dH * y
    dq = np.empty(len(vols))
    prev = 0.0
    for i in range(len(vols)):
        dq[i] = qcum[i] - prev * (1.0 - f[i])
        prev = qcum[i]
    return dq


def model_norm_heats(kind, params, M0, Xs, V0_mL, vols_uL):
    vols = np.asarray(vols_uL, dtype=float)
    if kind == "dilution":
        return np.full(len(vols), float(params["q0"]))
    dq = injection_heats_uJ(params["sites"], M0, Xs, V0_mL, vols)
    n_umol = vols * Xs * 1e-3
    return dq / (n_umol * 1000.0)


def _unpack(kind, theta):
    theta = [float(v) for v in theta]
    if kind == "one-site":
        K = 10.0 ** theta[0]
        n = theta[1]
        dH = theta[2]
        return {"sites": [[n, K, dH]], "n": n, "K": K, "dH": dH}
    if kind == "two-site":
        s1 = [theta[1], 10.0 ** theta[0], theta[2]]
        s2 = [theta[4], 10.0 ** theta[3], theta[5]]
        return {"sites": [s1, s2], "n1": s1[0], "K1": s1[1], "dH1": s1[2],
                "n2": s2[0], "K2": s2[1], "dH2": s2[2]}
    if kind == "dilution":
        return {"sites": [], "q0": theta[0]}
    raise ValueError("unknown model kind %s" % kind)


def _sse(kind, theta, idx, y, M0, Xs, V0, vols):
    params = _unpack(kind, theta)
    pred = model_norm_heats(kind, params, M0, Xs, V0, vols)
    r = pred[idx] - y
    return float(r @ r)


def nelder_mead(f, x0, step, budget, used=0):
    n = len(x0)

    def ev(x):
        nonlocal used
        if used >= budget:
            raise BudgetExhausted()
        used += 1
        return float(f(np.asarray(x, dtype=float)))

    simplex = [np.asarray(x0, dtype=float)]
    for i in range(n):
        x = np.asarray(x0, dtype=float).copy()
        x[i] += step[i]
        simplex.append(x)
    vals = [ev(x) for x in simplex]
    for _ in range(600):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        spread_v = max(abs(vals[0] - v) for v in vals[1:])
        spread_x = max(float(np.abs(simplex[0] - s).max()) for s in simplex[1:])
        if spread_v < 1e-10 and spread_x < 1e-8:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = ev(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
            continue
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = ev(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
            continue
        if fr >= vals[-1]:
            xc = centroid + 0.5 * (worst - centroid)
        else:
            xc = centroid + 0.5 * (xr - centroid)
        fc = ev(xc)
        if fc < min(fr, vals[-1]):
            simplex[-1] = xc
            vals[-1] = fc
            continue
        best = simplex[0]
        simplex = [best] + [best + 0.5 * (s - best) for s in simplex[1:]]
        vals = [vals[0]] + [ev(x) for x in simplex[1:]]
    order = np.argsort(vals)
    i = int(order[0])
    return simplex[i], vals[i], used


STARTS = {
    "one-site": [([2.0, 1.0, -30.0], [0.5, 0.2, 10.0]),
                 ([1.0, 1.2, -60.0], [0.5, 0.3, 15.0])],
    "two-site": [([2.0, 0.6, -20.0, 1.0, 0.6, -20.0], [0.5, 0.3, 10.0, 0.5, 0.3, 10.0]),
                 ([2.3, 1.0, -35.0, 0.3, 0.5, -10.0], [0.4, 0.3, 10.0, 0.4, 0.3, 10.0])],
    "dilution": None,
}


def fit_kind(kind, idx, y, M0, Xs, V0, vols, budget, used=0):
    if kind == "dilution":
        starts = [([float(np.median(y))], [5.0])]
    else:
        starts = STARTS[kind]
    best = None
    for x0, step in starts:
        x, fx, used = nelder_mead(
            lambda th: _sse(kind, th, idx, y, M0, Xs, V0, vols),
            x0, step, budget, used)
        if best is None or fx < best[1]:
            best = (x, fx)
    x, fx = best
    params = _unpack(kind, x)
    cand = {"kind": kind, "params": params, "sse": fx, "evals": used}
    if len(x) >= len(idx):
        cand["notes"] = ["underdetermined: parameter count >= fit point count"]
    else:
        cand["notes"] = []
    return cand, used


def fit_all(kinds, idx, y, M0, Xs, V0, vols, budget, tol_tss=0.05):
    idx = np.asarray(idx, dtype=int)
    y = np.asarray(y, dtype=float)
    tss = float(((y - y.mean()) ** 2).sum()) or 1.0
    cands = []
    used = 0
    for kind in kinds:
        cand, used = fit_kind(kind, idx, y, M0, Xs, V0, vols, budget, used)
        cands.append(cand)
    cands.sort(key=lambda c: (round(c["sse"], 9), c["kind"]))
    best_sse = cands[0]["sse"]
    for c in cands:
        c["equally_good"] = bool(c["sse"] <= best_sse + tol_tss * tss)
    return cands
