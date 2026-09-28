"""绑定模型候选与拟合（纯 NumPy，无浏览器逻辑）。

候选：
* one_site   一位点独立结合（Wiseman 累积形式）
* two_site   两位点顺序独立结合（自由能/焓可加）
* dilution   稀释假设：每注射摩尔的热为常数

等优（equipotential）模型 *全部保留*：|ΔAICc| <= tolerance 即视为不可唯一决定，
不会只输出“最优”一个。每个候选带参数版本、输入摘要、父级依赖。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

AICC_TIE_TOLERANCE = 2.0


def _accumulated(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x)


def one_site_q(X: np.ndarray, M0: float, K: float, DH: float, D0: float, offset: float) -> np.ndarray:
    # X: 逐注射配体摩尔；M0: 池内主体摩尔；返回逐注射 q (cal/mol 配体)
    L = _accumulated(X)
    Q = np.empty_like(L, dtype=float)

    def bound(lt: float) -> float:
        s = M0 + lt + 1.0 / K
        return 0.5 * (s - math.sqrt(max(s * s - 4.0 * M0 * lt, 0.0)))

    for i, lt in enumerate(L):
        b_now = bound(lt)
        b_prev = bound(L[i - 1]) if i else 0.0
        Q[i] = DH * (b_now - b_prev) / max(X[i], 1e-30) + D0
    return Q + offset


def two_site_q(X: np.ndarray, M0: float, K1: float, DH1: float,
               K2: float, DH2: float, D0: float, offset: float) -> np.ndarray:
    L = _accumulated(X)

    def bound_pair(lt: float) -> Tuple[float, float]:
        # 序贯两独立位点：各自按一位点结合，共享池体积，但位点2主体量也为 M0
        def b(K: float) -> float:
            s = M0 + lt + 1.0 / K
            return 0.5 * (s - math.sqrt(max(s * s - 4.0 * M0 * lt, 0.0)))
        return b(K1), b(K2)

    q = np.empty_like(L, dtype=float)
    for i, lt in enumerate(L):
        n1, n2 = bound_pair(lt)
        p1, p2 = bound_pair(L[i - 1]) if i else (0.0, 0.0)
        q[i] = (DH1 * (n1 - p1) + DH2 * (n2 - p2) + D0 * X[i]) / max(X[i], 1e-30) + offset
    return q


def dilution_q(X: np.ndarray, D0: float) -> np.ndarray:
    return np.full_like(X, D0, dtype=float)


def _aicc(ssr: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return float("inf")
    return n * math.log(max(ssr, 1e-30) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _grid_refine(predict, p_lo: np.ndarray, p_hi: np.ndarray, x: np.ndarray, y: np.ndarray,
                 log_axes: Sequence[int], rounds: int = 4, grid: int = 9) -> Tuple[np.ndarray, float, np.ndarray]:
    """确定性网格收缩 + 线性最小二乘精炼（对 log 参数用对数网格）。"""
    lo, hi = p_lo.astype(float).copy(), p_hi.astype(float).copy()
    best_p = None
    best_ssr = float("inf")
    rng = np.random.default_rng(0)
    axes = list(range(len(lo)))
    for rnd in range(rounds):
        spaces = []
        for ax in axes:
            if ax in log_axes:
                spaces.append(np.geomspace(max(lo[ax], 1e-30), hi[ax], grid))
            else:
                spaces.append(np.linspace(lo[ax], hi[ax], grid))
        meshes = np.meshgrid(*spaces, indexing="ij")
        pts = np.stack([m.ravel() for m in meshes], axis=1)
        for p in pts:
            try:
                pred = predict(p, x)
                if np.any(~np.isfinite(pred)):
                    continue
                ssr = float(np.sum((pred - y) ** 2))
            except (FloatingPointError, ValueError, OverflowError):
                continue
            if ssr < best_ssr:
                best_ssr, best_p = ssr, p
        if best_p is None:
            break
        for ax in axes:
            span = hi[ax] - lo[ax]
            frac = 0.25 / (2 ** rnd)
            lo[ax] = max(lo[ax], best_p[ax] - span * frac)
            hi[ax] = min(hi[ax], best_p[ax] + span * frac)
    return best_p, best_ssr, x


def _jac_residual(predict, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = p.astype(float)
    base = predict(p, x)
    r = base - y
    J = np.empty((len(x), len(p)))
    for j in range(len(p)):
        h = max(abs(p[j]) * 1e-6, 1e-12)
        pp = p.copy(); pp[j] += h
        J[:, j] = (predict(pp, x) - base) / h
    return J, r


def fit_models(points: List[Dict[str, Any]], moles_total_mol: float,
               injected_moles: Sequence[float],
               model_names: Sequence[str] = ("one_site", "two_site", "dilution"),
               tie_tolerance: float = AICC_TIE_TOLERANCE) -> Dict[str, Any]:
    """对同一份归一化热拟合全部候选，返回含等优集合的结构化结果。"""
    used = [pt for pt in points if pt.get("used_for_fit") and pt.get("q_cal_mol") is not None]
    if len(used) < 3:
        return {"ok": False, "reason": "insufficient_points", "n_points": len(used),
                "candidates": [], "equipotential": []}
    ord_idx = np.array([pt["ordinal"] for pt in used])
    order = np.argsort(ord_idx)
    used = [used[i] for i in order]
    Xall = np.array(injected_moles, dtype=float)
    ord_to_i = {pt["ordinal"]: i for i, pt in enumerate(points)}
    X = np.array([Xall[ord_to_i[pt["ordinal"]]] for pt in used], dtype=float)
    y = np.array([pt["q_cal_mol"] for pt in used], dtype=float)
    n = len(y)
    span = max(float(np.ptp(y)), 1e-12)
    q_scale = max(abs(float(np.mean(y))), span, 1.0)
    M0 = moles_total_mol

    candidates: List[Dict[str, Any]] = []

    if "one_site" in model_names:
        def pred1(p, xx):
            K, DH, offset = p
            return one_site_q(xx, M0, math.exp(K), math.exp(DH), 0.0, offset)
        p1, ssr1, _ = _grid_refine(
            pred1,
            np.array([math.log(1e3), math.log(q_scale * 0.05), -q_scale]),
            np.array([math.log(1e10), math.log(q_scale * 20.0), q_scale]),
            X, y, log_axes=[0, 1], rounds=4, grid=11,
        )
        if p1 is not None:
            p1, ssr1 = _lsq_polish(pred1, p1, X, y)
            candidates.append(_candidate("one_site", pred1, p1, ssr1, n,
                                         ["K_M-1", "dH_cal/mol", "offset_cal/mol"],
                                         [math.exp(p1[0]), math.exp(p1[1]), p1[2]],
                                         {"K_is_log": True, "dH_is_log_sign": True}))

    if "two_site" in model_names:
        def pred2(p, xx):
            K1, DH1, K2, DH2, offset = p
            return two_site_q(xx, M0, math.exp(K1), math.exp(DH1),
                              math.exp(K2), math.exp(DH2), 0.0, offset)
        p2, ssr2, _ = _grid_refine(
            pred2,
            np.array([math.log(1e3), math.log(q_scale * 0.05), math.log(1e3), math.log(-q_scale * 10 + 1e-12), -q_scale]),
            np.array([math.log(1e10), math.log(q_scale * 20.0), math.log(1e10), math.log(q_scale * 20.0), q_scale]),
            X, y, log_axes=[0, 1, 2, 3], rounds=3, grid=7,
        )
        if p2 is not None:
            p2, ssr2 = _lsq_polish(pred2, p2, X, y)
            candidates.append(_candidate("two_site", pred2, p2, ssr2, n,
                                         ["K1_M-1", "dH1_cal/mol", "K2_M-1", "dH2_cal/mol", "offset_cal/mol"],
                                         [math.exp(p2[0]), math.exp(p2[1]), math.exp(p2[2]), math.exp(p2[3]), p2[4]],
                                         {"note": "2 位点各按 M0 主体量"}))

    if "dilution" in model_names:
        p0 = float(np.mean(y))
        pred = lambda p, xx: dilution_q(xx, p[0])
        ssr = float(np.sum((pred(np.array([p0]), X) - y) ** 2))
        candidates.append(_candidate("dilution", pred, np.array([p0]), ssr, n,
                                     ["D0_cal/mol"], [p0], {}))

    if not candidates:
        return {"ok": False, "reason": "no_candidate_converged", "n_points": n,
                "candidates": [], "equipotential": []}

    best = min(c["aic_c"] for c in candidates)
    for c in candidates:
        c["delta_aic_c"] = c["aic_c"] - best
        c["is_best"] = abs(c["delta_aic_c"]) <= 1e-9
    equip = sorted(
        [c["model"] for c in candidates if c["delta_aic_c"] <= tie_tolerance],
        key=lambda m: ({"one_site": 0, "dilution": 1, "two_site": 2}.get(m, 9), m),
    )
    return {"ok": True, "n_points": n, "candidates": candidates,
            "equipotential": equip, "tie_tolerance": tie_tolerance,
            "unique_model": equip[0] if len(equip) == 1 else None,
            "ordinals": [pt["ordinal"] for pt in used],
            "x_moles": X.tolist(), "y_obs": y.tolist()}


def _lsq_polish(predict, p: np.ndarray, x: np.ndarray, y: np.ndarray, iters: int = 12):
    cur = p.astype(float).copy()
    ssr = None
    for _ in range(iters):
        J, r = _jac_residual(predict, cur, x, y)
        try:
            dp, *_ = np.linalg.lstsq(J, -r, rcond=None)
        except np.linalg.LinAlgError:
            break
        cand = cur + dp
        try:
            ssr_new = float(np.sum((predict(cand, x) - y) ** 2))
        except (FloatingPointError, ValueError, OverflowError):
            break
        if ssr is not None and abs(ssr - ssr_new) < 1e-10 * max(ssr, 1e-30):
            cur, ssr = cand, ssr_new
            break
        cur, ssr = cand, ssr_new
    if ssr is None:
        ssr = float(np.sum((predict(cur, x) - y) ** 2))
    return cur, ssr


def _candidate(name, predict, p, ssr, n, param_names, physical_params, extra) -> Dict[str, Any]:
    # predict 闭包保留在内存拟合期间；序列化时只保留模型名与参数
    aic = _aicc(ssr, n, len(p))
    return {
        "model": name,
        "param_names": list(param_names),
        "params_log_space": [float(v) for v in p],
        "params_physical": [float(v) for v in physical_params],
        "ssr": float(ssr),
        "aic_c": float(aic),
        "n_params": int(len(p)),
        **extra,
    }


def predict_candidate(candidate: Dict[str, Any], moles_total_mol: float,
                      x_moles: Sequence[float]) -> np.ndarray:
    x = np.asarray(x_moles, dtype=float)
    name = candidate["model"]
    pp = candidate["params_log_space"]
    if name == "one_site":
        return one_site_q(x, moles_total_mol, math.exp(pp[0]), math.exp(pp[1]), 0.0, pp[2])
    if name == "two_site":
        return two_site_q(x, moles_total_mol, math.exp(pp[0]), math.exp(pp[1]),
                          math.exp(pp[2]), math.exp(pp[3]), 0.0, pp[4])
    if name == "dilution":
        return dilution_q(x, pp[0])
    raise ValueError(name)
