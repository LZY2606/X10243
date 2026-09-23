"""绑定模型候选：稀释假设 / 一位点 1:1 / 两位点独立。

拟合只用 NumPy（Levenberg–Marquardt + 数值雅可比）。
- 每个候选记录参数版本、输入摘要、父级依赖；
- 无法唯一决定时返回全部合法候选，等优判定交由 service（阈值 TIE_REL_TOL）；
- 点数不足/参数不可辨识属于“语义无解”，返回诊断而非伪造结果；
- 全部派生量携带单位，浓度使用规范化后的 mM（原浓度在材料层保留）。
"""
from __future__ import annotations

import math
import numpy as np

from .models import (
    MODEL_DILUTION, MODEL_ONE_SITE, MODEL_TWO_SITE, MODEL_ORDER,
    MODEL_LABELS, TIE_REL_TOL, Diagnostic, ITCError,
)

BINDING_RULE_VERSION = "binding-1.0.0"
MIN_POINTS = {MODEL_DILUTION: 1, MODEL_ONE_SITE: 3, MODEL_TWO_SITE: 7}
# 物理量保护：浓度非正、K 非正等属于语义无解
_MIN_POS = 1e-12
# kcal per mol 与 µJ per µmol 的换算（1 µcal = 4.184 µJ）
KCAL_PER_MOL_TO_UJ_PER_UMOL = 4184.0


def _theta_one(X, K):
    # θ = ( (1+X+K M0) - sqrt((1+X+K M0)^2 - 4 K M0 X) ) / (2 K M0)
    KM = K * 1.0
    A = 1.0 + X + KM
    disc = A * A - 4.0 * KM * X
    disc = np.maximum(disc, 0.0)
    return (A - np.sqrt(disc)) / (2.0 * KM)


def model_values(model: str, X, V0, dH, K=None, n2=None, dH2=None, K2=None):
    """返回每个注射的 q_i（µJ）。X,V0 为数组，浓度 mM、体积 µL。"""
    X = np.asarray(X, dtype=float)
    if model == MODEL_DILUTION:
        return np.full_like(X, float(dH), dtype=float)
    if model == MODEL_ONE_SITE:
        th = _theta_one(X, K)
        return dH * V0 * th
    if model == MODEL_TWO_SITE:
        th1 = _theta_one(X, K)
        th2 = _theta_one(X, K2)
        return V0 * (dH * th1 + dH2 * n2 * th2)
    raise ITCError(f"未知模型 {model!r}",
                   [Diagnostic("unknown_model", f"未知模型 {model!r}", model)])


def _residuals(model, p, X, V0, q):
    if model == MODEL_DILUTION:
        pred = model_values(model, X, V0, p[0])
    elif model == MODEL_ONE_SITE:
        dH, logK = p
        pred = model_values(model, X, V0, dH, K=math.exp(logK))
    else:
        dH, logK, dH2, logK2, n2 = p[0], p[1], p[2], p[3], p[4]
        pred = model_values(model, X, V0, dH, K=math.exp(logK),
                            n2=n2, dH2=dH2, K2=math.exp(logK2))
    return pred - q


def _jacobian(model, p, X, V0, q, eps=1e-7):
    base = _residuals(model, p, X, V0, q)
    J = np.zeros((len(q), len(p)))
    for j in range(len(p)):
        step = eps * max(1.0, abs(p[j]))
        pp = p.copy()
        pp[j] += step
        J[:, j] = (_residuals(model, pp, X, V0, q) - base) / step
    return J, base


def _lm(model, p0, X, V0, q, n_iter=80) -> tuple[np.ndarray, float, bool]:
    """正参数通过 log/softplus 变换；返回参数、SSE、收敛标志。"""
    p = np.array(p0, dtype=float)
    lam = 1e-3
    J, r = _jacobian(model, p, X, V0, q)
    sse = float(r @ r)
    for _ in range(n_iter):
        JTJ = J.T @ J
        Jr = J.T @ r
        diag = np.diag(np.maximum(np.diag(JTJ), 1e-12))
        try:
            step = -np.linalg.solve(JTJ + lam * diag, Jr)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        pn = p + step
        if model != MODEL_DILUTION:
            # n2 必须为正
            if model == MODEL_TWO_SITE and pn[4] <= 1e-6:
                pn[4] = 1e-6
        rn = _residuals(model, pn, X, V0, q)
        sse_n = float(rn @ rn)
        if sse_n < sse:
            if abs(sse - sse_n) <= 1e-11 * (1.0 + sse):
                p, sse = pn, sse_n
                return p, sse, True
            p, sse, lam = pn, sse_n, max(lam / 3.0, 1e-12)
            J, r = _jacobian(model, p, X, V0, q)
        else:
            lam *= 5.0
            if lam > 1e12:
                return p, sse, False
    return p, sse, False


def build_xy(dilution_doc: dict, conditions: dict, concentration: dict):
    """从稀释产物构造模型输入 X_i（mM，配体/大分子摩尔比）与 V0（µL，池体积）。

    只纳入：未舍弃、有净值、空白对应成功、非致命 的注射。其余以诊断返回。
    """
    cell = concentration["cell_macromolecule"]["value_mM"]
    if cell is None or cell <= 0:
        raise ITCError(
            "池内大分子浓度不可用，无法定义绑定模型",
            [Diagnostic("cell_concentration_unusable",
                        "池内浓度缺失或不可换算", "concentration.cell")],
        )
    V0 = float(conditions["cell_volume"]["value_uL"])
    X = []
    q = []
    used = []
    excluded = []
    for e in dilution_doc["entries"]:
        reason = None
        if e.get("discarded"):
            reason = "discarded"
        elif e.get("fatal"):
            reason = "fatal_integration"
        elif e.get("q_net_uJ") is None:
            reason = e.get("blank_status", "no_net")
        if reason:
            excluded.append({"ordinal": e["ordinal"], "reason": reason})
            continue
        Xi = _xi_for(e["ordinal"], e, concentration)
        if Xi is None:
            excluded.append({"ordinal": e["ordinal"],
                             "reason": "concentration_unusable"})
            continue
        X.append(Xi)
        q.append(float(e["q_net_uJ"]))
        used.append(e["ordinal"])
    return (np.array(X, dtype=float), np.array(q, dtype=float), V0, cell,
            used, excluded)


def _xi_for(ordinal, entry, concentration):
    """第 i 注后配体/大分子摩尔比（累积注射体积 / 池体积 × 浓度比）。"""
    cell = concentration["cell_macromolecule"]["value_mM"]
    syr = (concentration.get("syringe_ligand") or {}).get("value_mM")
    per = concentration.get("per_injection") or {}
    # 单注浓度修订优先（per_injection 为该注注射时配体浓度）
    node = per.get(str(ordinal))
    if node and node.get("ok"):
        lig = node["value_mM"]
    else:
        lig = syr
    if lig is None:
        return None
    # X_i = (Σ_{k<=i} V_k * L_k)/V0 / M0；这里 entry 不带累积量，
    # 由 service 在构造时把 cumulative_volume_uL 写入（见 prepare_fit_inputs）。
    cum = entry.get("cumulative_volume_uL")
    if cum is None:
        return None
    return (cum * lig / entry["_V0_for_ratio"]) / cell


def prepare_fit_inputs(dilution_doc, conditions, concentration):
    """补齐累积体积后构造模型输入（service 调用）。"""
    cum = 0.0
    V0 = float(conditions["cell_volume"]["value_uL"])
    for e in dilution_doc["entries"]:
        # 即使被排除也保持累积一致（按注射时序物理累积）
        cum += float(e["injection_volume_uL"])
        e["cumulative_volume_uL"] = cum
        e["_V0_for_ratio"] = V0
    X, q, V0, M0, used, excluded = build_xy(dilution_doc, conditions, concentration)
    return X, q, V0, M0, used, excluded


def _decode(model, p):
    if model == MODEL_DILUTION:
        return {"q_dilution_uJ": float(p[0]),
                "dH_dilution_kcal_per_mol": None,
                "K_mM": None, "n_sites": None, "K2_mM": None,
                "dH2_kcal_per_mol": None}
    dH, K = p[0], math.exp(p[1])
    dec = {"dH_uJ_per_umol": float(dH),
           "dH_kcal_per_mol": float(dH) / KCAL_PER_MOL_TO_UJ_PER_UMOL,
           "K_mM": float(K), "K_per_M": float(K * 1000.0),
           "n_sites": None, "K2_mM": None, "dH2_kcal_per_mol": None}
    if model == MODEL_TWO_SITE:
        dec.update({
            "dH2_uJ_per_umol": float(p[2]),
            "dH2_kcal_per_mol": float(p[2]) / KCAL_PER_MOL_TO_UJ_PER_UMOL,
            "K2_mM": float(math.exp(p[3])),
            "K2_per_M": float(math.exp(p[3]) * 1000.0),
            "n_sites": float(p[4]),
        })
    return dec


def _starts(model, X, q, V0):
    if model == MODEL_DILUTION:
        return [np.array([float(np.median(q) if len(q) else 0.0)]),
                np.array([0.0]), np.array([float(np.mean(q) if len(q) else 0.0)])]
    dH0_lo, dH0_hi = sorted([float(np.min(q)) / max(V0, 1e-9),
                             float(np.max(q)) / max(V0, 1e-9)])
    dHs = [dH0_hi, dH0_lo, (dH0_hi + dH0_lo) / 2.0 or -1.0, 1.0, -1.0]
    Ks = [0.01, 1.0, 100.0]  # mM，log 空间起点
    if model == MODEL_ONE_SITE:
        return [np.array([dH, math.log(K)]) for dH in dHs for K in Ks]
    # 两位点：两组位点不同 K 与焓，n2 起点
    out = []
    for dH in dHs[:3]:
        for K in (0.05, 5.0):
            for K2 in (0.5, 50.0):
                out.append(np.array([dH, math.log(K), dH * 0.5,
                                     math.log(K2), 1.0]))
    return out


def fit_models(dilution_doc, conditions, concentration, *,
               models=MODEL_ORDER) -> dict:
    """拟合全部候选；每个模型给 best SSE 与参数。等优不在此裁决。"""
    X, q, V0, M0, used, excluded = prepare_fit_inputs(
        dilution_doc, conditions, concentration)
    candidates = []
    diag: list[Diagnostic] = []
    if len(used) == 0:
        raise ITCError(
            "没有可用于建模的注射（全部舍弃/缺口/空白缺失）",
            [Diagnostic("no_fit_points", "无有效注射进入模型",
                        "entries")] +
            [Diagnostic("excluded", f"第 {e['ordinal']} 注被排除: {e['reason']}",
                        f"injection #{e['ordinal']}", severity="info")
             for e in excluded],
        )
    for model in models:
        if len(used) < MIN_POINTS[model]:
            diag.append(Diagnostic(
                "under_determined",
                f"{MODEL_LABELS[model]} 需要 >= {MIN_POINTS[model]} 个有效注射，"
                f"当前 {len(used)} 个（语义无解：参数不可辨识）",
                f"model:{model}", severity="error"))
            candidates.append({
                "model": model, "label": MODEL_LABELS[model],
                "status": "under_determined",
                "n_points": len(used), "sse_uJ2": None,
                "aic": None, "params": None, "converged": False,
                "diagnostics": [diag[-1].to_dict()],
            })
            continue
        best = None
        for p0 in _starts(model, X, q, V0):
            try:
                p, sse, conv = _lm(model, p0, X, V0, q)
            except (OverflowError, FloatingPointError):
                continue
            if not np.isfinite(sse):
                continue
            if best is None or sse < best[1]:
                best = (p, sse, conv)
        if best is None:
            candidates.append({
                "model": model, "label": MODEL_LABELS[model],
                "status": "failed", "n_points": len(used),
                "sse_uJ2": None, "aic": None, "params": None,
                "converged": False,
                "diagnostics": [Diagnostic(
                    "fit_failed", "拟合未收敛到有限解", f"model:{model}",
                    severity="error").to_dict()],
            })
            continue
        p, sse, conv = best
        k = {MODEL_DILUTION: 1, MODEL_ONE_SITE: 2, MODEL_TWO_SITE: 5}[model]
        n = len(used)
        # 小样本保护的 AIC（仅用于展示与并列提示）
        aic = n * math.log(max(sse / max(n, 1), 1e-300)) + 2 * k
        pred = (_residuals(model, p, X, V0, q) + q)
        cand = {
            "model": model, "label": MODEL_LABELS[model],
            "status": "converged" if conv else "converged_loose",
            "n_points": n, "sse_uJ2": float(sse),
            "rmse_uJ": float(math.sqrt(sse / n)),
            "aic": float(aic),
            "params": _decode(model, p),
            "param_count": k,
            "converged": bool(conv),
            "fit_ordinals": used,
            "predicted_q_uJ": [float(x) for x in pred],
            "observed_q_uJ": [float(x) for x in q],
            "X_ratio": [float(x) for x in X],
            "diagnostics": [] if conv else [Diagnostic(
                "loose_convergence", "拟合收敛较松，请人工复核残差",
                f"model:{model}", severity="warning").to_dict()],
        }
        candidates.append(cand)

    _mark_ties(candidates)
    return {
        "kind": "binding",
        "rule_version": BINDING_RULE_VERSION,
        "energy_unit": "uJ",
        "concentration_unit": "mM",
        "cell_concentration_mM": M0,
        "cell_volume_uL": V0,
        "candidates": candidates,
        "used_ordinals": used,
        "excluded": excluded,
        "diagnostics": [d.to_dict() for d in diag],
    }


def _mark_ties(candidates):
    """等优：相对差 <= TIE_REL_TOL 的候选全部标 tied，并列集合全部保留。"""
    ok = [c for c in candidates if c.get("sse_uJ2") is not None]
    if not ok:
        return
    best = min(c["sse_uJ2"] for c in ok)
    winners = [c for c in ok
               if abs(c["sse_uJ2"] - best) <= TIE_REL_TOL * (1.0 + abs(best))]
    # 稳定顺序：sse 再按 MODEL_ORDER
    order_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    winners.sort(key=lambda c: (c["sse_uJ2"], order_rank[c["model"]]))
    tied_set = {c["model"] for c in winners}
    for c in candidates:
        c["is_winner"] = c["model"] in tied_set
        c["tie_size"] = len(tied_set) if c["model"] in tied_set else 0
