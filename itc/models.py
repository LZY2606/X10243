"""绑定模型候选：一位点 / 多位点 / 稀释假设。

- 等价输入 -> 稳定顺序（先按 BIC，再按模型名，再按参数键）。
- 多个等优模型（delta BIC <= tie_delta）全部保留为合法候选，不强行唯一化。
- 资源预算耗尽是与格式损坏、语义无解不同的第三种结果。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import BudgetExhausted, Diagnostic, SemanticInvalid
from .identity import result_id
from .materials import CONCENTRATIONS, CONDITIONS, SCHEDULE
from .rules import ModelRules


def _diag(code: str, message: str, location: str) -> dict:
    return Diagnostic(code, message, location).to_dict()


@dataclass
class FitOutcome:
    id: str
    candidates: list[dict[str, Any]]
    ties: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    input_summary: dict[str, Any]
    parent_ids: list[str]
    rule_version: str
    rule_id: str
    stale: bool = False
    stale_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "model_fit",
            "rule_version": self.rule_version,
            "rule_id": self.rule_id,
            "input_summary": self.input_summary,
            "parent_ids": self.parent_ids,
            "candidates": self.candidates,
            "ties": self.ties,
            "selected": self.selected,
            "diagnostics": self.diagnostics,
            "stale": self.stale,
            "stale_reason": self.stale_reason,
        }


def _x_m(injections: list[dict[str, Any]], cell_volume: float,
         conc_syringe: float, conc_cell: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.array([i["volume_l"] for i in injections], dtype=float)
    idx = np.array([i["index"] for i in injections], dtype=float)
    vc = np.cumsum(v)
    # 每注后池内总体积（第 i 注：V0 + sum_{k<=i} dv_k）
    vol = cell_volume + vc
    total_lig = conc_syringe * vc
    x = total_lig / vol
    m = conc_cell * cell_volume / vol
    return idx, x, m


def _theta1(x: np.ndarray, m: np.ndarray, k: float) -> np.ndarray:
    b = x + m + 1.0 / k
    disc = b * b - 4.0 * x * m
    disc = np.maximum(disc, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * m)


def _theta2(x: np.ndarray, m: np.ndarray, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray]:
    # 顺序两位点 n1=n2=1；求一元四次的物理根（数值稳健，逐点）
    th1 = np.empty_like(x)
    th2 = np.empty_like(x)
    for j, (xt, mt) in enumerate(zip(x, m)):
        roots = np.roots([
            k1 * k1 * k2,
            -k1 * (k1 * k2 * xt - k1 - 2 * k1 * k2 * mt - k2),
            k1 * (k2 * xt + 1 + 2 * k1 * mt + 2 * k2 * mt) - k1 * k1,
            -(k1 * xt + k1 * mt + k2 * mt + 1.0),
            xt,
        ])
        real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
        feasible = real_roots[(real_roots > 0) & (real_roots < mt * (1 + 1e-9))]
        if len(feasible) == 0:
            th1[j] = min(xt, mt) / mt
            th2[j] = th1[j]
        else:
            L = float(np.max(feasible))
            th1[j] = k1 * L / (1 + k1 * L)
            th2[j] = k2 * L / (1 + k2 * L)
    return th1, th2


def _lstsq(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return coef, float(resid @ resid)


def _bic(sse: float, n: int, k: int) -> float:
    sse = max(sse, 1e-300)
    return n * np.log(sse / n) + k * np.log(n)


def fit_models(
    integration: dict[str, Any],
    materials: dict,
    rules: ModelRules,
    *,
    budget_override: int | None = None,
) -> FitOutcome:
    diagnostics: list[Diagnostic] = []
    schedule = materials[SCHEDULE].canonical
    conc = materials[CONCENTRATIONS].canonical
    cond = materials[CONDITIONS].canonical

    if conc["cell"] is None or conc["syringe"] is None:
        raise SemanticInvalid("浓度不完整，无法拟合绑定模型", [
            Diagnostic("unconvertible_value", "浓度缺失，绑定模型需要浓度", "concentrations")])
    if cond["cell_volume"] is None:
        raise SemanticInvalid("池体积不完整", [
            Diagnostic("unconvertible_value", "池体积缺失", "conditions.cell_volume")])

    injections = integration["injections"]
    kept = [i for i in injections if not i.get("discarded") and i.get("q_net") is not None]
    if len(kept) < 3:
        raise SemanticInvalid("可用注射热不足，无法拟合模型", [
            Diagnostic("too_few_injections", f"仅 {len(kept)} 个可用注射热（至少需要 3 个）", "injections")])

    idx_full = np.array([i["index"] for i in injections], dtype=float)
    volume_by_index = {i["index"]: i["volume_l"] for i in injections}
    kept_inj = [{"index": i["index"], "volume_l": volume_by_index[i["index"]]} for i in kept]
    _, x, m = _x_m(kept_inj, cond["cell_volume"], conc["syringe"], conc["cell"])
    qobs = np.array([i["q_net"] for i in kept], dtype=float)  # cal

    n = len(kept)
    v0 = cond["cell_volume"]
    budget = rules.eval_budget if budget_override is None else budget_override
    used = 0
    candidates: list[dict[str, Any]] = []

    grid = np.linspace(3.0, 9.0, 25)

    def consume(need: int) -> None:
        nonlocal used
        used += need
        if used > budget:
            raise BudgetExhausted(f"模型评估预算耗尽（{used} > {budget}）", budget, used)

    if 1 in rules.n_sites:
        for logk in grid:
            consume(1)
            k = 10.0 ** logk
            theta = _theta1(x, m, k)
            dtheta = v0 * np.diff(np.concatenate([[0.0], theta * m]))
            design = np.column_stack([dtheta, np.ones(n)])
            coef, sse = _lstsq(design, qobs)
            candidates.append({
                "model": "one_site",
                "params": {"logK": round(float(logk), 6), "n_sites": 1},
                "coef": {"dH_cal_per_mol": float(coef[0]), "offset_cal": float(coef[1])},
                "K_M": float(k),
                "sse": sse,
                "bic": _bic(sse, n, 3),
                "k_params": 3,
            })

    if 2 in rules.n_sites:
        for a, logk1 in enumerate(grid):
            for logk2 in grid:
                consume(1)
                k1, k2 = 10.0 ** logk1, 10.0 ** logk2
                th1, th2 = _theta2(x, m, k1, k2)
                d1 = v0 * np.diff(np.concatenate([[0.0], th1 * m]))
                d2 = v0 * np.diff(np.concatenate([[0.0], th2 * m]))
                design = np.column_stack([d1, d2, np.ones(n)])
                coef, sse = _lstsq(design, qobs)
                candidates.append({
                    "model": "two_site",
                    "params": {"logK1": round(float(logk1), 6), "logK2": round(float(logk2), 6),
                               "n_sites": 2},
                    "coef": {"dH1_cal_per_mol": float(coef[0]), "dH2_cal_per_mol": float(coef[1]),
                             "offset_cal": float(coef[2])},
                    "sse": sse,
                    "bic": _bic(sse, n, 5),
                    "k_params": 5,
                })

    if rules.include_dilution:
        consume(1)
        if rules.dilution_degree == 1:
            design = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
            coef, sse = _lstsq(design, qobs)
            params = {"degree": 1}
            kpar = 3
        else:
            design = np.ones((n, 1))
            coef, sse = _lstsq(design, qobs)
            params = {"degree": 0}
            kpar = 2
        candidates.append({
            "model": "dilution",
            "params": params,
            "coef": {"constant_cal": float(coef[0]),
                     "slope_cal_per_injection": float(coef[1]) if len(coef) > 1 else 0.0},
            "sse": sse,
            "bic": _bic(sse, n, kpar),
            "k_params": kpar,
        })

    # 稳定顺序：BIC 升序，模型名，参数键
    candidates.sort(key=lambda c: (round(c["bic"], 9), c["model"],
                                   tuple(sorted((str(k), str(v)) for k, v in c["params"].items()))))
    best_bic = candidates[0]["bic"]
    ties = [c for c in candidates if c["bic"] - best_bic <= rules.tie_delta]
    # 去重同一模型族（只保留该族最佳），但跨族等优全部保留
    best_by_model: dict[str, dict] = {}
    for c in ties:
        if c["model"] not in best_by_model:
            best_by_model[c["model"]] = c
    selected = list(best_by_model.values())

    parent_ids = sorted({integration["id"], materials[SCHEDULE].id,
                         materials[CONCENTRATIONS].id, materials[CONDITIONS].id, rules.id})
    input_summary = {
        "integration_result": integration["id"],
        "n_kept": n,
        "kept_indices": [i["index"] for i in kept],
        "cell_M": conc["cell"],
        "syringe_M": conc["syringe"],
        "cell_volume_L": cond["cell_volume"],
        "temperature_K": cond["temperature"],
    }
    rid = result_id("model_fit", input_summary, rules.content(), parent_ids)
    diag_dicts = [d.to_dict() if isinstance(d, Diagnostic) else d for d in diagnostics]
    return FitOutcome(
        id=rid,
        candidates=candidates,
        ties=ties,
        selected=selected,
        diagnostics=diag_dicts,
        input_summary=input_summary,
        parent_ids=parent_ids,
        rule_version=rules.version,
        rule_id=rules.id,
    )
