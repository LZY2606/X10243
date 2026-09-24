"""绑定模型：一位点（Wiseman）、两位点、纯稀释（常数热）。

模型侧与数据归一化侧共用同一个严格的增量热公式：
    V_i 为针前池内累计体积，Qcum_i 为该时刻池内累计结合热，则
    dQ_i = Qcum_i - Qcum_{i-1} * (1 - dV_i/V0)
拟合与审阅残差都在归一化空间（kcal/mol 注入配体）。
自由能 dG = -R T ln Ka（cal/mol），报告 kcal/mol。
"""
from __future__ import annotations

import math
from typing import Any

R_CAL = 1.987204258  # cal/(mol*K)

MODELS = ("one_site", "two_site", "dilution")
MODEL_LABELS = {
    "one_site": "一位点",
    "two_site": "两位点",
    "dilution": "纯稀释（常数热）",
}

PARAM_SPECS: dict[str, list[tuple[str, str, float, float]]] = {
    "one_site": [("n", "sites", 0.2, 6.0),
                 ("logK", "log10(1/M)", 3.0, 10.0),
                 ("dH", "kcal/mol", -40.0, 40.0)],
    "two_site": [("n1", "sites", 0.05, 4.0),
                 ("logK1", "log10(1/M)", 3.0, 10.0),
                 ("dH1", "kcal/mol", -40.0, 40.0),
                 ("n2", "sites", 0.05, 4.0),
                 ("logK2", "log10(1/M)", 3.0, 10.0),
                 ("dH2", "kcal/mol", -40.0, 40.0)],
    "dilution": [("q0", "kcal/mol", -20.0, 20.0)],
}


def _fraction_bound(logk: float, c_ligand_m: float, n: float,
                    c_macromol_m: float) -> float:
    """单类独立位点的 Wiseman 填充率。

    X = K a, T = K n M（a=游离配体浓度, M=大分子浓度）；
    填充率 = 0.5 * (1 + X + T - sqrt((1+X+T)^2 - 4 X T)) / T。
    """
    k = 10.0 ** logk
    x = k * c_ligand_m
    t_term = k * n * c_macromol_m
    if t_term <= 0:
        return 0.0
    b = 1.0 + x + t_term
    disc = max(b * b - 4.0 * x * t_term, 0.0)
    return 0.5 * (b - math.sqrt(disc)) / t_term


def cumulative_heat(model: str, theta: dict[str, float],
                    ligand_m: float, c_cell_now: float,
                    v0: float) -> float:
    """池内累计结合热 Qcum（µcal）。"""
    if model == "dilution":
        return 0.0
    if model == "one_site":
        n, logk = theta["n"], theta["logK"]
        f = _fraction_bound(logk, ligand_m, n, c_cell_now)
        q = n * f * theta["dH"]
        return q * 1e6 * c_cell_now * v0
    q = 0.0
    for suf in ("1", "2"):
        n = theta[f"n{suf}"]
        q += n * _fraction_bound(theta[f"logK{suf}"], ligand_m, n,
                                 c_cell_now) \
            * theta[f"dH{suf}"]
    return q * 1e6 * c_cell_now * v0


def simulate(model: str, theta: dict[str, float],
             points: list[dict[str, Any]], v0: float, c_cell: float,
             c_syringe: float) -> list[float | None]:
    """按针前累计体积逐步模拟预测归一化热（kcal/mol）。"""
    preds: list[float | None] = []
    cum_v = 0.0
    q_prev = 0.0
    for p in points:
        dv = p.get("dv")
        if dv is None or not v0 or not c_cell or not c_syringe:
            preds.append(None)
            continue
        if model == "dilution":
            preds.append(float(theta["q0"]))
            cum_v += dv
            continue
        c_now = c_cell * v0 / (v0 + cum_v)
        ligand_m = c_syringe * cum_v / (v0 + cum_v)
        q = cumulative_heat(model, theta, ligand_m, c_now, v0)
        dq = q - q_prev * (1.0 - dv / v0)
        preds.append((dq) / (c_syringe * dv) / 1e6)
        q_prev = q
        cum_v += dv
    return preds


def canonical_theta(model: str, theta: dict[str, float]) -> dict[str, float]:
    """两位点标签交换视为同一模型：按 logK 降序规范化。"""
    if model != "two_site":
        return dict(theta)
    a = (theta["logK1"], theta["n1"], theta["dH1"])
    b = (theta["logK2"], theta["n2"], theta["dH2"])
    if a < b:
        a, b = b, a
    return {"n1": a[1], "logK1": a[0], "dH1": a[2],
            "n2": b[1], "logK2": b[0], "dH2": b[2]}


def delta_g(logk: float, temperature_k: float) -> float:
    return -R_CAL * temperature_k * math.log(10.0 ** logk) / 1000.0
