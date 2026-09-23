"""绑定模型候选与拟合（只用 NumPy）。

模型：
  one_site   单位点 Wiseman 等温式（浓度经池稀释递推）
  two_site   两个独立位点之和（位点互换时天然给出等优候选）
  dilution   仅稀释热：dQ = q0 + q1*[X]

所有候选都保留；χ² 在等价容差内的全部标记 equivalent，不自动挑选。
浓度等条件版本由调用方写进 deps；浓度修订后旧拟合标 stale（见 project）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .errors import Budget, Diagnostic, Outcome

PARAM_VERSION = "binding-params/v1"
EQUIV_REL = 0.02
EQUIV_ABS = 1e-6


@dataclass
class Conditions:
    cell_mM: float
    syringe_mM: float
    V0_uL: float
    T_K: float
    volumes_uL: list[float]   # 完整注射序列体积（含被舍弃/裁窗注射）

    def dilution_series(self) -> tuple[np.ndarray, np.ndarray]:
        """返回每次注射后的池内蛋白浓度 M_i 与配体浓度 X_i（mM）。"""
        d = np.ones(len(self.volumes_uL) + 1)
        for i, v in enumerate(self.volumes_uL):
            d[i + 1] = d[i] * (1.0 - v / self.V0_uL)
        d = d[1:]
        M = self.cell_mM * d
        X = self.syringe_mM * (1.0 - d)
        return M, X


def _bound(X: np.ndarray, S: float, K: float) -> np.ndarray:
    """1:1 结合平衡：配合物浓度（mM），S=位点总浓度，K 单位 1/mM。

    用有理化形式 B = 2SX/(a+sqrt(a^2-4SX))，K 很大时不产生 inf-inf。
    """
    a = S + X + 1.0 / K
    disc = np.sqrt(np.maximum(a ** 2 - 4.0 * S * X, 0.0))
    return (2.0 * S * X) / (a + disc)


def _safe_exp(x: float | np.ndarray) -> float | np.ndarray:
    return np.exp(np.clip(x, -50.0, 700.0) / 1.0)


_LOGK_MIN, _LOGK_MAX = -10.0, 20.0


def _clip(p: np.ndarray, kind: str) -> np.ndarray:
    q = p.copy()
    if kind in ("one_site", "two_site"):
        idxs = [0] if kind == "one_site" else [0, 3]
        for j in idxs:
            q[j] = float(np.clip(q[j], _LOGK_MIN, _LOGK_MAX))
        nidx = [2] if kind == "one_site" else [2, 5]
        for j in nidx:
            q[j] = float(np.clip(q[j], 1e-3, 20.0))
    return q


def _injection_heats(Q: np.ndarray, volumes_uL: list[float], V0_uL: float
                     ) -> np.ndarray:
    """累积热 -> 每次注射热，含体积位移修正。"""
    Qp = np.concatenate(([0.0], Q[:-1]))
    v = np.asarray(volumes_uL, dtype=float)
    # µL*mM*ΔH(J/mol) -> µJ : V[L]*[mol/L]*J/mol = J ; 1 µL*1 mM = 1e-9 mol
    return (Q - Qp) + (v / V0_uL) * 0.5 * (Q + Qp)


# Q_i 单位 µJ：V0_uL * 1e-9 * B_mM * ΔH
def one_site_Q(p: np.ndarray, M: np.ndarray, X: np.ndarray, V0_uL: float
               ) -> np.ndarray:
    logK, dH, n = p
    K = _safe_exp(logK)
    # µL*mM*J/mol -> µJ : 1e-6 L * 1e-3 mol/L * J/mol * 1e6 µJ/J = 1e-3
    return V0_uL * 1e-3 * dH * _bound(X, n * M, K)


def two_site_Q(p: np.ndarray, M: np.ndarray, X: np.ndarray, V0_uL: float
               ) -> np.ndarray:
    logK1, dH1, n1, logK2, dH2, n2 = p
    return V0_uL * 1e-3 * (
        dH1 * _bound(X, n1 * M, _safe_exp(logK1))
        + dH2 * _bound(X, n2 * M, _safe_exp(logK2)))


def dilution_Q(p: np.ndarray, M: np.ndarray, X: np.ndarray, V0_uL: float
               ) -> np.ndarray:
    # 直接返回累积量的形状由 _injection_heats 配合：累积稀释热累加线性项
    q0, q1 = p
    d_inj = q0 + q1 * X
    return np.cumsum(d_inj)


MODELS: dict[str, dict[str, Any]] = {
    "one_site": {
        "Q": one_site_Q,
        "n_params": 3,
        "p0": [np.log(5e2), -2.0e4, 1.0],
        "scale": [2.0, 2.0e4, 0.5],
    },
    "two_site": {
        "Q": two_site_Q,
        "n_params": 6,
        "p0": [np.log(2e2), -1.5e4, 0.6, np.log(2e3), -2.5e4, 0.6],
        "scale": [2.0, 2.0e4, 0.5, 2.0, 2.0e4, 0.5],
    },
    "dilution": {
        "Q": dilution_Q,
        "n_params": 2,
        "p0": [-20.0, -30.0],
        "scale": [50.0, 100.0],
    },
}


def model_heats(kind: str, p: np.ndarray, cond: Conditions
                ) -> np.ndarray:
    M, X = cond.dilution_series()
    Q = MODELS[kind]["Q"](p, M, X, cond.V0_uL)
    return _injection_heats(Q, cond.volumes_uL, cond.V0_uL)


def usable_subset(integration_payload: dict[str, Any]
                  ) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    """返回可用注射的 id、下标、观测热及行；稀释递推仍用完整序列。"""
    ids, idx, heats, rows = [], [], [], []
    for j, r in enumerate(integration_payload["injections"]):
        if r["usable_for_fit"]:
            ids.append(r["injection_id"])
            idx.append(j)
            heats.append(r["heat_net_uJ"])
            rows.append(r)
    return ids, np.asarray(idx, dtype=int), np.asarray(heats, dtype=float), rows


MAX_ITER_PER_FIT = 1000


def _lm(kind: str, obs: np.ndarray, cond: Conditions, budget: Budget,
        used: np.ndarray, p0: np.ndarray | None = None) -> dict[str, Any]:
    spec = MODELS[kind]
    p = _clip(np.asarray(p0 if p0 is not None else spec["p0"], dtype=float),
              kind)
    scale = np.asarray(spec["scale"], dtype=float)
    lam = 1e-3

    def full_heats(pp: np.ndarray) -> np.ndarray:
        return model_heats(kind, pp, cond)

    resid = full_heats(p)[used] - obs
    ssr = float(np.sum(resid ** 2))
    it = 0
    while True:
        budget.spend(1, f"{kind}.lm.iter{it}")
        J = np.zeros((len(obs), spec["n_params"]))
        for j in range(spec["n_params"]):
            dp = 1e-6 * max(abs(p[j]), scale[j] * 1e-4)
            pj = p.copy(); pj[j] += dp
            J[:, j] = (full_heats(pj)[used] - full_heats(p)[used]) / dp
        JtJ = J.T @ J
        Jtr = J.T @ resid
        step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), -Jtr)
        pn = p + step
        rn = full_heats(pn)[used] - obs
        ssrn = float(np.sum(rn ** 2))
        it += 1
        if ssrn < ssr:
            if abs(ssr - ssrn) < 1e-10 * max(1.0, ssr):
                p, ssr, resid = _clip(pn, kind), ssrn, rn
                break
            p, ssr, resid = _clip(pn, kind), ssrn, rn
            lam = max(lam / 5.0, 1e-12)
        else:
            lam *= 10.0
            if lam > 1e12:
                break
        if it >= MAX_ITER_PER_FIT:
            # 预算耗尽：显式失败，绝不返回伪装收敛的解
            from .errors import BudgetExhausted
            raise BudgetExhausted(
                f"{kind}: LM 达到 {it} 次迭代预算（每候选上限）")
    return {"params": p, "chi2": ssr, "iterations": it,
            "residuals": resid}


def fit_one(kind: str, integration_payload: dict[str, Any],
            cond: Conditions, budget: Budget,
            p0: np.ndarray | None = None) -> Outcome:
    injection_ids, used, obs_heats, _rows = usable_subset(integration_payload)
    if len(obs_heats) < MODELS[kind]["n_params"]:
        return Outcome.fail("semantic_unsolvable", [Diagnostic(
            "insufficient_injections",
            f"{kind}: 有效注射 {len(obs_heats)} 个 < 参数数 "
            f"{MODELS[kind]['n_params']}，模型语义无解",
            f"fit.{kind}")])
    try:
        res = _lm(kind, obs_heats, cond, budget, used, p0)
    except Exception as exc:  # BudgetExhausted -> 独立状态
        from .errors import BudgetExhausted
        if isinstance(exc, BudgetExhausted):
            return Outcome.fail("budget_exhausted", [Diagnostic(
                "budget_exhausted", str(exc), f"fit.{kind}")])
        raise
    fitted = res["residuals"] + obs_heats
    residuals = [
        {"injection_id": iid, "obs_uJ": float(o), "fit_uJ": float(f),
         "resid_uJ": float(r)}
        for iid, o, f, r in zip(injection_ids, obs_heats, fitted,
                                res["residuals"])]
    return Outcome.good({
        "kind": "fit", "model": kind,
        "param_version": PARAM_VERSION,
        "params": _export_params(kind, res["params"]),
        "params_raw": [float(x) for x in res["params"]],
        "chi2": res["chi2"], "dof": len(obs_heats) - MODELS[kind]["n_params"],
        "iterations": res["iterations"],
        "residuals": residuals,
        "units": {"heat": "uJ", "K_unit": "1/mM", "dH": "J/mol"},
    })


def _export_params(kind: str, p: np.ndarray) -> dict[str, float]:
    if kind == "one_site":
        return {"K_per_M": float(_safe_exp(p[0]) * 1e3),  # 1/mM -> 1/M
                "dH_J_per_mol": float(p[1]), "n": float(p[2])}
    if kind == "two_site":
        return {"K1_per_M": float(_safe_exp(p[0]) * 1e3),
                "dH1_J_per_mol": float(p[1]), "n1": float(p[2]),
                "K2_per_M": float(_safe_exp(p[3]) * 1e3),
                "dH2_J_per_mol": float(p[4]), "n2": float(p[5])}
    return {"q0_uJ": float(p[0]), "q1_uJ_per_mM": float(p[1])}


def mark_equivalent(fits: list[dict[str, Any]]) -> None:
    """χ² 落在同一容差带的候选互为等优；不挑选唯一胜者。"""
    if not fits:
        return
    chi = [f["chi2"] for f in fits]
    cmin = min(chi)
    band = max(EQUIV_ABS, EQUIV_REL * cmin)
    group = [i for i, c in enumerate(chi) if abs(c - cmin) <= band]
    for i, f in enumerate(fits):
        f["equivalent"] = len(group) > 1 and i in group
        f["equivalent_group"] = f"eq:{sorted(fits[j]['model'] for j in group)}" \
            if len(group) > 1 and i in group else None
        f["selected"] = None  # 无法唯一决定时保留全部合法候选


def observed_from_integration(payload: dict[str, Any]
                              ) -> tuple[list[str], list[float]]:
    """从积分结果取拟合观测：只取 usable_for_fit 的注射。"""
    ids, heats = [], []
    for r in payload["injections"]:
        if r["usable_for_fit"]:
            ids.append(r["injection_id"])
            heats.append(r["heat_net_uJ"])
    return ids, heats
