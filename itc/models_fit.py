"""绑定模型候选拟合（纯 NumPy，服务端可直接测试）。

候选：
- one_site:   一位点 Wiseman 等温式（Ka, dH, q_dil）
- multi_site: 两位点独立结合（Ka1, Ka2, dH1, dH2, q_dil）
- dilution:   纯稀释/机械热假设（每注恒定 q_dil；等价 Ka -> 0 的一位点）

无法唯一决定时保留全部合法候选。比较使用每注归一化 SSE；差异在
TIE_REL_TOL 内判为等优，等优集合以 (SSE, 模型名, 参数元组) 稳定排序。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ResourceExhausted

MODEL_VERSION = "binding-1.0.0"
TIE_REL_TOL = 1e-9

DEFAULT_BUDGETS = {
    "logk_grid": 601,
    "max_refine_iter": 60,
    "max_two_site_grid": 121,
}


def _theta(Ka, Xt, Mt):
    root_arg = (Xt / Mt + 1.0 + 1.0 / (Ka * Mt)) ** 2 - 4.0 * Xt / Mt
    root_arg = np.maximum(root_arg, 0.0)
    return 0.5 * (Xt / Mt + 1.0 + 1.0 / (Ka * Mt) - np.sqrt(root_arg))


def one_site_q(Ka, dH, q_dil, v0, Vinj, Xt, Mt):
    """标准 Wiseman 逐注热量（J）。

    dQ_i = dH V0 M_t (theta_i - theta_{i-1})
           + dH V_i M_t (theta_i + theta_{i-1}) / 2
           + q_dil V_i / V_std
    （末项为每注稀释/机械热，V_std 取各注体积的基准 1e-6 L=1 uL）
    """
    th = _theta(Ka, Xt, Mt)
    th_prev = np.concatenate(([0.0], th[:-1]))
    V_std = 1e-6
    return (dH * v0 * Mt * (th - th_prev)
            + 0.5 * dH * Vinj * Mt * (th + th_prev)
            + q_dil * Vinj / V_std)


def two_site_q(Ka1, Ka2, dH1, dH2, q_dil, v0, Vinj, Xt, Mt):
    th1 = _theta(Ka1, Xt, Mt)
    th2 = _theta(Ka2, Xt, Mt)
    tp1 = np.concatenate(([0.0], th1[:-1]))
    tp2 = np.concatenate(([0.0], th2[:-1]))
    V_std = 1e-6
    return (dH1 * v0 * Mt * (th1 - tp1)
            + 0.5 * dH1 * Vinj * Mt * (th1 + tp1)
            + dH2 * v0 * Mt * (th2 - tp2)
            + 0.5 * dH2 * Vinj * Mt * (th2 + tp2)
            + q_dil * Vinj / V_std)


def prepare_arrays(integral: dict, syringe_conc_m: float, cell_conc_m: float,
                   cell_volume_l: float) -> dict | None:
    """从积分结果构造拟合输入；无可用品阶注入返回 None。

    输入不完整时由调用方产出带位置诊断；这里只消费规范值。
    """
    rows = integral["injections"]
    keep = [r for r in rows if r["status"] not in ("invalid", "discarded_first")
            and r.get("blank_corrected_j") is not None]
    if len(keep) < 4:
        return None
    Vinj = np.array([r["volume_l"] for r in keep])
    q = np.array([r["blank_corrected_j"] for r in keep])
    v0 = cell_volume_l
    # 池内配体总浓度：随累计注入体积稀释
    cumV = np.cumsum(Vinj)
    Xt = syringe_conc_m * cumV / (v0 + cumV)
    Mt = cell_conc_m * v0 / (v0 + cumV)
    return {"Vinj": Vinj, "q": q, "v0": v0, "Xt": Xt, "Mt": Mt,
            "indices": [r["index"] for r in keep]}


def _linear_dh(Ka, Xt, Mt, Vinj, v0, q):
    """给定 Ka，[dH, q_dil] 线性最小二乘。"""
    th = _theta(Ka, Xt, Mt)
    thp = np.concatenate(([0.0], th[:-1]))
    a = v0 * Mt * (th - thp) + 0.5 * Vinj * Mt * (th + thp)
    b = Vinj / 1e-6
    A = np.column_stack([a, b])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    pred = A @ coef
    return coef[0], coef[1], pred


def fit_one_site(d, *, budgets=None):
    budgets = budgets or DEFAULT_BUDGETS
    Xt, Mt, Vinj, v0, q = d["Xt"], d["Mt"], d["Vinj"], d["v0"], d["q"]
    n_grid = budgets.get("logk_grid", DEFAULT_BUDGETS["logk_grid"])
    if n_grid > 100_000:
        raise ResourceExhausted(f"logk 网格预算过大: {n_grid}",
                                locations=["budgets.logk_grid"])
    logk = np.linspace(2.0, 12.0, n_grid)
    best = None
    for lk in logk:
        Ka = 10.0 ** lk
        dH, qd, pred = _linear_dh(Ka, Xt, Mt, Vinj, v0, q)
        sse = float(np.sum((pred - q) ** 2))
        if best is None or sse < best[0]:
            best = (sse, Ka, dH, qd)
    sse, Ka, dH, qd = best
    # Gauss-Newton 精修 logK
    it = budgets.get("max_refine_iter", DEFAULT_BUDGETS["max_refine_iter"])
    if it > 10_000:
        raise ResourceExhausted(f"精修迭代预算过大: {it}",
                                locations=["budgets.max_refine_iter"])
    x = np.log10(Ka)
    for _ in range(it):
        Ka = 10.0 ** x
        th = _theta(Ka, Xt, Mt)
        thp = np.concatenate(([0.0], th[:-1]))
        a = v0 * Mt * (th - thp) + 0.5 * Vinj * Mt * (th + thp)
        b = Vinj / 1e-6
        A = np.column_stack([a, b])
        coef, *_ = np.linalg.lstsq(A, q, rcond=None)
        pred = A @ coef
        r = pred - q
        eps = 1e-6
        Ka2 = 10.0 ** (x + eps)
        th2 = _theta(Ka2, Xt, Mt)
        tp2 = np.concatenate(([0.0], th2[:-1]))
        a2 = v0 * Mt * (th2 - tp2) + 0.5 * Vinj * Mt * (th2 + tp2)
        pred2 = coef[0] * a2 + coef[1] * b
        J = ((pred2 - pred) / (eps * np.log(10.0))).reshape(-1, 1)
        JtJ = float(J.T @ J)
        if JtJ <= 0:
            break
        step = -float(J.T @ r) / JtJ
        step = float(np.clip(step, -1.0, 1.0))
        x_new = x + step
        Ka_new = 10.0 ** x_new
        dHn, qdn, pn = _linear_dh(Ka_new, Xt, Mt, Vinj, v0, q)
        sse_new = float(np.sum((pn - q) ** 2))
        if abs(step) < 1e-10:
            sse, Ka, dH, qd = sse_new, Ka_new, dHn, qdn
            break
        if sse_new <= sse:
            x, sse, Ka, dH, qd = x_new, sse_new, Ka_new, dHn, qdn
        else:
            break
    pred = one_site_q(Ka, dH, qd, v0, Vinj, Xt, Mt)
    return {"model": "one_site", "sse": float(np.sum((pred - q) ** 2)),
            "params": {"Ka_M-1": Ka, "dH_J/mol": dH, "q_dil_J": qd},
            "pred": pred, "residual": q - pred}


def fit_dilution(d):
    """纯稀释：q_i = q_dil * V_i/V_std（单参数，经原点直线）。"""
    Vinj, q = d["Vinj"], d["q"]
    b = Vinj / 1e-6
    qd = float(np.dot(b, q) / np.dot(b, b))
    pred = qd * b
    return {"model": "dilution",
            "sse": float(np.sum((pred - q) ** 2)),
            "params": {"q_dil_J": qd}, "pred": pred,
            "residual": q - pred}


def fit_two_site(d, *, budgets=None):
    """两位点：在 (logK1, logK2) 网格上对 (dH1,dH2,q_dil) 线性求解。"""
    budgets = budgets or DEFAULT_BUDGETS
    Xt, Mt, Vinj, v0, q = d["Xt"], d["Mt"], d["Vinj"], d["v0"], d["q"]
    n = budgets.get("max_two_site_grid",
                    DEFAULT_BUDGETS["max_two_site_grid"])
    grid = int(np.sqrt(n))
    if grid * grid > 100_000:
        raise ResourceExhausted(f"两位点网格预算过大: {grid}x{grid}",
                                locations=["budgets.max_two_site_grid"])
    logks = np.linspace(2.0, 12.0, grid)
    cols = []
    sse_mat = np.empty((grid, grid))
    store = {}
    for i, l1 in enumerate(logks):
        K1 = 10.0 ** l1
        th1 = _theta(K1, Xt, Mt)
        tp1 = np.concatenate(([0.0], th1[:-1]))
        c1 = v0 * Mt * (th1 - tp1) + 0.5 * Vinj * Mt * (th1 + tp1)
        for j, l2 in enumerate(logks):
            K2 = 10.0 ** l2
            th2 = _theta(K2, Xt, Mt)
            tp2 = np.concatenate(([0.0], th2[:-1]))
            c2 = v0 * Mt * (th2 - tp2) + 0.5 * Vinj * Mt * (th2 + tp2)
            A = np.column_stack([c1, c2, Vinj / 1e-6])
            coef, *_ = np.linalg.lstsq(A, q, rcond=None)
            pred = A @ coef
            sse_mat[i, j] = np.sum((pred - q) ** 2)
            store[(i, j)] = (K1, K2, coef[0], coef[1], coef[2])
    i, j = np.unravel_index(np.argmin(sse_mat), sse_mat.shape)
    K1, K2, dH1, dH2, qd = store[(int(i), int(j))]
    pred = two_site_q(K1, K2, dH1, dH2, qd, v0, Vinj, Xt, Mt)
    return {"model": "multi_site",
            "sse": float(np.sum((pred - q) ** 2)),
            "params": {"Ka1_M-1": K1, "Ka2_M-1": K2,
                       "dH1_J/mol": dH1, "dH2_J/mol": dH2,
                       "q_dil_J": qd},
            "pred": pred, "residual": q - pred}


def _rank_key(c):
    return (round(c["sse_norm"], 12), c["model"],
            tuple(round(v, 9) for v in c["params"].values()))


def fit_binding(integral: dict, syringe_conc_m: float, cell_conc_m: float,
                cell_volume_l: float, *, parent_versions: dict,
                budgets: dict | None = None) -> dict:
    """全部候选都计算并保留；标记等优集合，绝不只留一个。"""
    d = prepare_arrays(integral, syringe_conc_m, cell_conc_m, cell_volume_l)
    if d is None:
        return {"status": "insufficient_points",
                "diagnostics": [{"message": "可用于拟合的有效注射少于 4 注"
                                             "（首注舍弃、缺口无效注入已排除）",
                                 "location": "integral.injections"}],
                "candidates": [], "tied": []}
    cands = [fit_one_site(d, budgets=budgets),
             fit_two_site(d, budgets=budgets),
             fit_dilution(d)]
    scale = float(np.dot(d["q"], d["q"])) or 1.0
    for c in cands:
        c["sse_norm"] = c["sse"] / scale
    cands.sort(key=_rank_key)
    best_norm = cands[0]["sse_norm"]
    tied = [c["model"] for c in cands
            if c["sse_norm"] <= best_norm + max(TIE_REL_TOL,
                                                TIE_REL_TOL * max(1.0, best_norm))]
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "used_injection_indices": d["indices"],
        "syringe_version_id": parent_versions["syringe"],
        "cell_conc_version_id": parent_versions["cell_conc"],
        "cell_volume_version_id": parent_versions["cell_volume"],
        "integral_digest": integral["input_digest"],
        "integral_result_id": integral.get("result_id"),
        "budgets": budgets or DEFAULT_BUDGETS,
        "tie_tolerance": TIE_REL_TOL,
        "candidates": [{k: v for k, v in c.items()
                        if k not in ("pred", "residual")} for c in cands],
        "tied": tied,
        "pred_by_model": {c["model"]: c["pred"].tolist() for c in cands},
        "residual_by_model": {c["model"]: c["residual"].tolist()
                              for c in cands},
        "q": d["q"].tolist(), "x_ratio": (d["Xt"] / d["Mt"]).tolist(),
    }
