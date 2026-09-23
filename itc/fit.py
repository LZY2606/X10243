"""绑定模型拟合: 一位点 / 多位点 / 稀释假设。

- 摩尔热在派生层计算: q[kJ/mol] = heat[µJ] / (V[µL] * C[mM])。
- 浓度修订使模型过期(视图层判定), 但不改原积分(积分不依赖浓度)。
- 多个等优模型: RSS 在容差内的全部候选都保留, 按 (rss, 参数摘要) 稳定排序。
- 有效点不足 -> semantically_unsolvable, 与 budget_exhausted / format_corrupt 区分。
"""
from __future__ import annotations

import numpy as np

from .units import normalize, diag
from .store import digest_of

FIT_VERSION = "fit/1.0"
EQUAL_TOL = 0.05  # RSS 相对容差: 容差内视为等优


# ---------- 摩尔热派生 ----------
def molar_points(integrals_payload, blank_payload, concentrations_payload,
                 discarded=()):
    """从积分(可选空白扣除)与浓度派生拟合序列。

    序列包含全部可积注射(被舍弃/缺空白的针仍占据稀释体积, 模型的稀释
    因子必须按完整序列计算); 只有可观测的针进入拟合。无法转换的值留
    诊断, 不伪装。
    """
    csyr, d1 = normalize(concentrations_payload["syringe"], "concentration",
                         position={"field": "syringe"})
    ccell, d2 = normalize(concentrations_payload["cell"], "concentration",
                          position={"field": "cell"})
    diags = [d for d in (d1, d2) if d]
    if diags:
        return None, None, None, diags

    if blank_payload is not None:
        corr = {e["index"]: e for e in blank_payload["series"]["blank_corrected"]}
    else:
        corr = {}
    series, cum_vol = [], 0.0
    for inj in integrals_payload.get("injections", []):
        pos = {"injection": inj["index"]}
        if inj["status"] != "ok":
            continue
        vol = inj["vol_uL"] if inj["vol_ok"] else None
        if not inj["vol_ok"]:
            diags.append(diag("unknown_unit",
                              f"注射 {inj['injection_id']} 体积单位不可转换, "
                              "摩尔热不可计算; 其排代体积按 0 处理",
                              position=pos, severity="warning"))
        cum_vol += vol or 0.0
        include = inj["injection_id"] not in discarded and inj["vol_ok"]
        heat = None
        if blank_payload is not None:
            entry = corr.get(inj["index"])
            if entry is None or entry["status"] != "ok":
                if include:
                    diags.append(diag(
                        "blank_missing",
                        f"注射 {inj['injection_id']} 无有效空白对应, "
                        "不进入扣除拟合", position=pos, severity="warning"))
                include = False
            else:
                heat = entry["heat_uJ"]
        else:
            heat = inj["heat_uJ"]
        series.append({
            "injection_id": inj["injection_id"], "index": inj["index"],
            "vol_uL": vol, "cum_vol_uL": cum_vol,
            "q_kJmol": (heat / (vol * csyr)) if (heat is not None and vol) else None,
            "heat_uJ": heat if include else None,
            "include": include,
        })
    return series, csyr, ccell, diags


# ---------- 模型 ----------
def _dilution_factors(vols, v0):
    factors, f = [], 1.0
    for v in vols:
        f *= (1.0 - v / v0)
        factors.append(f)
    return np.array(factors)


def _theta(ka, n, m, x):
    a = 1.0 + x / (n * m) + 1.0 / (ka * n * m)
    return 0.5 * (a - np.sqrt(np.maximum(a * a - 4.0 * x / (n * m), 0.0)))


def _per_injection(q_cum, vols, v0):
    out, prev = [], 0.0
    for i, q in enumerate(q_cum):
        dq = q - prev + (vols[i] / v0) * (q + prev) / 2.0
        out.append(dq)
        prev = q
    return np.array(out)


def predict_uJ(model, params, points, ccell, csyr, v0):
    """模型预测每次注射热量 (µJ)。浓度单位 mM, 体积 µL, ΔH kJ/mol。"""
    vols = np.array([p["vol_uL"] or 0.0 for p in points])
    f = _dilution_factors(vols, v0)
    m = ccell * f
    x = csyr * (1.0 - f)
    if model == "one_site":
        n, log_ka, dh = params
        ka = 10.0 ** log_ka
        q_cum = n * m * v0 * dh * _theta(ka, n, m, x)
        return _per_injection(q_cum, vols, v0)
    if model == "two_site":
        log_ka1, dh1, log_ka2, dh2 = params
        q_cum = v0 * m * (_theta(10.0 ** log_ka1, 1.0, m, x) * dh1
                          + _theta(10.0 ** log_ka2, 1.0, m, x) * dh2)
        return _per_injection(q_cum, vols, v0)
    if model == "dilution_constant":
        (c_uJ,) = params
        return np.full(len(points), c_uJ)
    raise ValueError(f"未知模型 {model}")


def _rss(model, params, points, ccell, csyr, v0):
    try:
        pred = predict_uJ(model, params, points, ccell, csyr, v0)
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        return np.inf
    if not np.all(np.isfinite(pred)):
        return np.inf
    mask = np.array([p.get("include", True) for p in points])
    obs = np.array([p["heat_uJ"] for p, m in zip(points, mask) if m])
    pred = pred[mask]
    return float(np.sum((obs - pred) ** 2))


def _nelder_mead(f, x0, steps, max_iter=800, tol=1e-9):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        x[i] += steps[i]
        simplex.append(x)
    vals = [f(x) for x in simplex]
    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * max(1.0, abs(vals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]]


_STARTS = {
    "one_site": [(n, lk, dh)
                 for n in (0.5, 1.0, 1.5)
                 for lk in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
                 for dh in (-60.0, -20.0, -5.0, 5.0, 20.0)],
    "two_site": [(lk1, dh1, lk2, dh2)
                 for lk1 in (-3.0, -1.0, 1.0)
                 for dh1 in (-40.0, -10.0)
                 for lk2 in (-2.0, 0.0, 2.0)
                 for dh2 in (-20.0, 0.0, 5.0)],
    "dilution_constant": [(0.0,)],
}
_STEPS = {
    "one_site": (0.2, 0.3, 5.0),
    "two_site": (0.3, 5.0, 0.3, 5.0),
    "dilution_constant": (1.0,),
}
_PARAM_NAMES = {
    "one_site": ("n", "log10_Ka_mM", "dH_kJmol"),
    "two_site": ("log10_Ka1_mM", "dH1_kJmol", "log10_Ka2_mM", "dH2_kJmol"),
    "dilution_constant": ("c_uJ",),
}
MIN_POINTS = {"one_site": 3, "two_site": 5, "dilution_constant": 1}


def _canonical_params(model, params):
    p = [float(v) for v in params]
    if model == "two_site":  # 位点交换等价: 按 Ka 排序, 保证等价输入身份稳定
        pair = sorted([(p[0], p[1]), (p[2], p[3])])
        p = [pair[0][0], pair[0][1], pair[1][0], pair[1][1]]
    return p


def fit_model(model, points, ccell, csyr, v0, equal_tol=EQUAL_TOL):
    """拟合一个模型, 保留全部等优候选。"""
    base = {"kind": "model_fit", "model": model, "param_version": FIT_VERSION,
            "units": {"heat": "µJ", "molar_heat": "kJ/mol",
                      "concentration": "mM", "volume": "µL"}}
    n_obs = sum(1 for p in points if p.get("include", True))
    if n_obs < MIN_POINTS[model]:
        return {**base, "status": "semantically_unsolvable",
                "diagnostics": [diag(
                    "insufficient_points",
                    f"模型 {model} 至少需要 {MIN_POINTS[model]} 个有效点, "
                    f"实际 {n_obs}")],
                "candidates": []}
    f = lambda p: _rss(model, p, points, ccell, csyr, v0)  # noqa: E731
    found = {}
    for start in _STARTS[model]:
        p, rss = _nelder_mead(f, np.array(start, dtype=float), _STEPS[model])
        if not np.isfinite(rss):
            continue
        cp = _canonical_params(model, p)
        key = tuple(round(v, 6) for v in cp)
        if key not in found or rss < found[key]:
            found[key] = rss
    if not found:
        return {**base, "status": "semantically_unsolvable",
                "diagnostics": [diag("no_finite_fit", "所有起点均未收敛到有限 RSS")],
                "candidates": []}
    best = min(found.values())
    kept = [(rss, key) for key, rss in found.items()
            if rss <= best * (1.0 + equal_tol) + 1e-12]
    # 稳定顺序: (rss, 参数摘要)
    kept.sort(key=lambda rk: (rk[0], digest_of(list(rk[1]))))
    candidates = []
    for rss, key in kept:
        params = dict(zip(_PARAM_NAMES[model], key))
        pred = predict_uJ(model, list(key), points, ccell, csyr, v0)
        residuals = [round(float(p["heat_uJ"] - pr), 9)
                     for p, pr in zip(points, pred) if p.get("include", True)]
        candidates.append({
            "candidate_id": "cand/" + digest_of(
                {"model": model, "params": key})[7:19],
            "params": params, "rss": rss,
            "residuals_uJ": residuals,
            "injection_ids": [p["injection_id"] for p in points
                              if p.get("include", True)],
        })
    return {**base, "status": "ok", "diagnostics": [],
            "equal_tol": equal_tol, "candidates": candidates}


def compare_models(fit_payloads, equal_tol=EQUAL_TOL):
    """跨模型比较: RSS 最优候选之间, 容差内全部标记为等优, 全部保留。"""
    entries = []
    for fp in fit_payloads:
        if fp["status"] != "ok" or not fp["candidates"]:
            entries.append({"model": fp["model"], "status": fp["status"],
                            "best_rss": None})
            continue
        best = fp["candidates"][0]
        entries.append({"model": fp["model"], "status": "ok",
                        "best_rss": best["rss"],
                        "best_candidate": best["candidate_id"],
                        "n_candidates": len(fp["candidates"])})
    finite = [e["best_rss"] for e in entries if e["best_rss"] is not None]
    equally_good = []
    if finite:
        best = min(finite)
        equally_good = [e["model"] for e in entries
                        if e["best_rss"] is not None
                        and e["best_rss"] <= best * (1.0 + equal_tol) + 1e-12]
    return {"kind": "model_comparison", "param_version": FIT_VERSION,
            "equal_tol": equal_tol, "models": entries,
            "equally_good": sorted(equally_good)}
