"""绑定模型候选：一位点、两位点（多位点）与稀释假设（NumPy 实现）。

全部输入为规范化单位：热 cal、体积 L、浓度 mol/L、温度 K。
模型拟合只依赖派生层（净热、注射体积、浓度、池体积），服务端可直接调用。

判定：
- 无法唯一决定（如有效注射点不足、参数不收敛）返回 UNRESOLVABLE；
- 预算耗尽（网格计算超限）返回 RESOURCE_EXHAUSTED；
- 多个等优模型全部保留，按稳定顺序排序，绝不自动二选一。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .results import Diagnostic, Outcome
from .units import FACTORS

MICROLITER_TO_LITER = FACTORS["volume"]["µL"]


class ModelKey(str, Enum):
    ONE_SITE = "one_site"
    TWO_SITE = "two_site"
    DILUTION_ONLY = "dilution_only"


MODEL_ORDER = (ModelKey.DILUTION_ONLY, ModelKey.ONE_SITE, ModelKey.TWO_SITE)
EQUAL_SSR_TOLERANCE = 1e-9
MIN_POINTS_ONE_SITE = 4
MIN_POINTS_TWO_SITE = 7
# 资源预算：模型评估次数上限
MAX_GRID_EVALUATIONS = 120_000


@dataclass
class FitPoint:
    injection_index: int
    volume_l: float
    ligand_cum_l: float
    macromolecule_m: float
    net_heat_cal: float


@dataclass
class ModelCandidate:
    key: ModelKey
    params: dict
    ssr: float
    aic: float
    predicted_cal: tuple[float, ...]
    residual_cal: tuple[float, ...]
    equivalent_rank: int = 0
    accepted: bool = False
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def param_fingerprint(self) -> str:
        return hashlib.sha256(
            repr(tuple(sorted(self.params.items()))).encode()
        ).hexdigest()[:12]

    @property
    def candidate_id(self) -> str:
        return f"cand_{self.key.value}_{self.param_fingerprint()}"

    def as_dict(self) -> dict:
        return {
            "key": self.key.value,
            "candidate_id": self.candidate_id,
            "params": self.params,
            "ssr": self.ssr,
            "aic": self.aic,
            "predicted_cal": list(self.predicted_cal),
            "residual_cal": list(self.residual_cal),
            "equivalent_rank": self.equivalent_rank,
            "accepted": self.accepted,
            "diagnostics": [vars(d) for d in self.diagnostics],
        }


@dataclass
class FitResult:
    points: tuple[FitPoint, ...]
    candidates: tuple[ModelCandidate, ...]
    unique_best: bool
    status: str                       # ok | unresolvable | resource_exhausted
    diagnostics: list[Diagnostic] = field(default_factory=list)
    input_digest: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "unique_best": self.unique_best,
            "points": [vars(p) for p in self.points],
            "candidates": [c.as_dict() for c in self.candidates],
            "diagnostics": [vars(d) for d in self.diagnostics],
            "input_digest": self.input_digest,
        }


def fit_id(depends: tuple, tail_factor: float = 0.0) -> str:
    body = repr((tuple(depends), round(tail_factor, 6)))
    return "fit_" + hashlib.sha256(body.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 一位点顺序热（binding polynomial，二次式稳定解）
# ---------------------------------------------------------------------------

def one_site_free_concentration(xt: np.ndarray, mt: float, ka: float,
                                n: float) -> np.ndarray:
    """给定总配体浓度 xt 与总大分子浓度 mt，求自由配体 x（二次式稳定形式）。

    Ka*(nMt + x + 1/Ka - xt) ... 解 [X]：
      Ka*x^2 + (1 + Ka*(n*M - Xt))*x - Xt = 0
    稳定小根：x = 2 Xt / (b + sqrt(b^2 + 4 Ka Xt))，
      b = 1 + Ka(nM - Xt)
    """
    xt = np.asarray(xt, dtype=float)
    a = ka
    b = 1.0 + ka * (n * mt - xt)
    disc = np.sqrt(b * b + 4.0 * a * xt)
    return 2.0 * xt / (b + disc)


def one_site_heats(points: list[FitPoint], ka: float, dh: float, n: float,
                   cell_volume_l: float) -> np.ndarray:
    """每次注射的观测热 q_i（含置换修正），单位 cal。"""
    v0 = cell_volume_l
    q = np.zeros(len(points))
    prev_bound = 0.0
    for i, p in enumerate(points):
        mt = p.macromolecule_m * (1.0 - p.ligand_cum_l / (2.0 * v0))
        x = one_site_free_concentration(np.array([p.ligand_cum_l / v0]), mt, ka, n)[0]
        theta = ka * x / (1.0 + ka * x)
        bound = mt * theta * n
        q[i] = v0 * dh * (bound - prev_bound) + \
            p.volume_l * dh * (bound + prev_bound) / 2.0
        prev_bound = bound
    return q


def _linear_dh(points: list[FitPoint], predicted_shape: np.ndarray) -> tuple[float, np.ndarray]:
    """给定形状向量（与 dh 成正比），最小二乘求 dh 与预测热。"""
    q = np.array([p.net_heat_cal for p in points], dtype=float)
    denom = float(predicted_shape @ predicted_shape)
    if denom <= 0:
        return 0.0, np.zeros_like(q)
    dh = float(predicted_shape @ q / denom)
    return dh, dh * predicted_shape


def _aic(ssr: float, n: int, k: int) -> float:
    if ssr <= 0 or not math.isfinite(ssr):
        return -math.inf
    return n * math.log(ssr / n) + 2 * k


def _shape_one_site(points: list[FitPoint], ka: float, n: float,
                    v0: float) -> np.ndarray:
    return one_site_heats(points, ka, 1.0, n, v0)


def fit_one_site(points: list[FitPoint], cell_volume_l: float,
                 budget: int = MAX_GRID_EVALUATIONS) -> Outcome[ModelCandidate]:
    n = len(points)
    if n < MIN_POINTS_ONE_SITE:
        return Outcome.unresolvable(
            "INSUFFICIENT_POINTS",
            f"一位点模型至少需要 {MIN_POINTS_ONE_SITE} 个有效净热点，当前 {n}")
    log_ka_grid = np.linspace(2.0, 9.0, 36)          # 1e2..1e9 M^-1
    n_grid = np.linspace(0.3, 5.0, 24)
    if log_ka_grid.size * n_grid.size > budget:
        return Outcome.resource(
            "GRID_BUDGET_EXCEEDED",
            f"一位点网格评估 {log_ka_grid.size * n_grid.size} 超过预算 {budget}")
    q = np.array([p.net_heat_cal for p in points])
    best: tuple[float, float, float, np.ndarray] | None = None
    for lk in log_ka_grid:
        ka = 10.0 ** lk
        for nn in n_grid:
            shape = _shape_one_site(points, ka, nn, cell_volume_l)
            dh, pred = _linear_dh(points, shape)
            ssr = float(np.sum((q - pred) ** 2))
            if best is None or ssr < best[0]:
                best = (ssr, ka, nn, pred)
    assert best is not None
    # 局部细化：在最优点附近二次细化
    ssr0, ka0, n0, pred0 = best
    lk0 = math.log10(ka0)
    for lk in np.linspace(max(2.0, lk0 - 0.25), min(9.0, lk0 + 0.25), 13):
        for nn in np.linspace(max(0.3, n0 - 0.15), min(5.0, n0 + 0.15), 13):
            shape = _shape_one_site(points, 10.0 ** lk, nn, cell_volume_l)
            dh, pred = _linear_dh(points, shape)
            ssr = float(np.sum((q - pred) ** 2))
            if ssr < ssr0:
                ssr0, ka0, n0, pred0 = ssr, 10.0 ** lk, nn, pred
    shape = _shape_one_site(points, ka0, n0, cell_volume_l)
    dh, pred = _linear_dh(points, shape)
    params = {"Ka_M-1": float(ka0), "N": float(n0), "dH_cal_per_mol": float(dh)}
    cand = ModelCandidate(ModelKey.ONE_SITE, params, ssr0, _aic(ssr0, n, 3),
                          tuple(float(x) for x in pred),
                          tuple(float(x) for x in (q - pred)))
    return Outcome.success(cand)


# ---------------------------------------------------------------------------
# 两位点模型（binding polynomial 三次式，Newton 迭代求自由配体）
# ---------------------------------------------------------------------------

def two_site_free_x(xt: float, mt: float, ka1: float, ka2: float,
                    n1: float, n2: float) -> float:
    a = ka1 * ka2
    b = ka1 + ka2 + a * (n1 + n2) * mt - a * xt
    c = 1.0 + (ka1 * n1 + ka2 * n2) * mt - (ka1 + ka2) * xt
    d = -xt
    # Newton，初值取自由配体 = 总配体（弱结合限）
    x = max(xt, 1e-20)
    for _ in range(60):
        f = a * x ** 3 + b * x ** 2 + c * x + d
        fp = 3 * a * x ** 2 + 2 * b * x + c
        if abs(fp) < 1e-300:
            break
        step = f / fp
        x -= step
        if x < 0:
            x = 1e-20
        if abs(step) <= 1e-12 * max(1.0, abs(x)):
            break
    return max(x, 0.0)


def two_site_heats(points: list[FitPoint], ka1: float, ka2: float,
                   n1: float, n2: float, dh1: float, dh2: float,
                   v0: float) -> np.ndarray:
    q = np.zeros(len(points))
    prev1 = prev2 = 0.0
    for i, p in enumerate(points):
        mt = p.macromolecule_m * (1.0 - p.ligand_cum_l / (2.0 * v0))
        xt = p.ligand_cum_l / v0
        x = two_site_free_x(xt, mt, ka1, ka2, n1, n2)
        th1 = n1 * mt * ka1 * x / (1.0 + ka1 * x + ka2 * x + ka1 * ka2 * x * x)
        th2 = n2 * mt * ka2 * x / (1.0 + ka1 * x + ka2 * x + ka1 * ka2 * x * x)
        q[i] = v0 * (dh1 * (th1 - prev1) + dh2 * (th2 - prev2)) + \
            p.volume_l * (dh1 * (th1 + prev1) + dh2 * (th2 + prev2)) / 2.0
        prev1, prev2 = th1, th2
    return q


def fit_two_site(points: list[FitPoint], cell_volume_l: float,
                 budget: int = MAX_GRID_EVALUATIONS) -> Outcome[ModelCandidate]:
    n = len(points)
    if n < MIN_POINTS_TWO_SITE:
        return Outcome.unresolvable(
            "INSUFFICIENT_POINTS",
            f"两位点模型至少需要 {MIN_POINTS_TWO_SITE} 个有效净热点，当前 {n}")
    grid = (12, 4, 4, 4)       # logka1, ratio-ka2, n1, n2
    nevals = int(np.prod(grid))
    if nevals > budget:
        return Outcome.resource(
            "GRID_BUDGET_EXCEEDED",
            f"两位点网格评估 {nevals} 超过预算 {budget}")
    q = np.array([p.net_heat_cal for p in points])
    lk1g = np.linspace(2.5, 8.5, grid[0])
    ratios = (0.25, 1.0, 4.0, 16.0)
    n1g = np.linspace(0.5, 3.0, grid[2])
    n2g = np.linspace(0.5, 3.0, grid[3])
    best = None
    for lk1 in lk1g:
        for ratio in ratios:
            ka1, ka2 = 10.0 ** lk1, 10.0 ** lk1 * ratio
            for nn1 in n1g:
                for nn2 in n2g:
                    shape1 = two_site_heats(points, ka1, ka2, nn1, nn2, 1.0, 0.0,
                                            cell_volume_l)
                    shape2 = two_site_heats(points, ka1, ka2, nn1, nn2, 0.0, 1.0,
                                            cell_volume_l)
                    A = np.column_stack([shape1, shape2])
                    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
                    pred = A @ coef
                    ssr = float(np.sum((q - pred) ** 2))
                    if best is None or ssr < best[0]:
                        best = (ssr, ka1, ka2, nn1, nn2, float(coef[0]),
                                float(coef[1]), pred)
    assert best is not None
    ssr, ka1, ka2, nn1, nn2, dh1, dh2, pred = best
    params = {"Ka1_M-1": float(ka1), "Ka2_M-1": float(ka2),
              "N1": float(nn1), "N2": float(nn2),
              "dH1_cal_per_mol": float(dh1), "dH2_cal_per_mol": float(dh2)}
    cand = ModelCandidate(ModelKey.TWO_SITE, params, ssr, _aic(ssr, n, 6),
                          tuple(float(x) for x in pred),
                          tuple(float(x) for x in (q - pred)))
    return Outcome.success(cand)


# ---------------------------------------------------------------------------
# 稀释假设：q_i = a + b * V_inj（与绑定无关）
# ---------------------------------------------------------------------------

def fit_dilution_only(points: list[FitPoint]) -> Outcome[ModelCandidate]:
    n = len(points)
    if n < 2:
        return Outcome.unresolvable(
            "INSUFFICIENT_POINTS", "稀释假设至少需要 2 个有效净热点")
    q = np.array([p.net_heat_cal for p in points])
    v = np.array([p.volume_l for p in points])
    A = np.column_stack([np.ones_like(v), v])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    pred = A @ coef
    ssr = float(np.sum((q - pred) ** 2))
    params = {"offset_cal": float(coef[0]),
              "slope_cal_per_L": float(coef[1])}
    cand = ModelCandidate(ModelKey.DILUTION_ONLY, params, ssr, _aic(ssr, n, 2),
                          tuple(float(x) for x in pred),
                          tuple(float(x) for x in (q - pred)))
    return Outcome.success(cand)


# ---------------------------------------------------------------------------
# 顶层：候选比较（保留全部等优模型）
# ---------------------------------------------------------------------------

def _input_digest(points: list[FitPoint], cell_volume_l: float,
                  temperature_k: float) -> str:
    body = repr(([(p.injection_index, p.volume_l, p.ligand_cum_l,
                   p.macromolecule_m, p.net_heat_cal) for p in points],
                 round(cell_volume_l, 15), round(temperature_k, 9)))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def fit_binding_models(points: list[FitPoint], *, cell_volume_l: float,
                       temperature_k: float,
                       budget: int = MAX_GRID_EVALUATIONS,
                       keys: tuple[ModelKey, ...] = MODEL_ORDER
                       ) -> Outcome[FitResult]:
    digest = _input_digest(points, cell_volume_l, temperature_k)
    if not points:
        return Outcome.unresolvable("NO_POINTS",
                                    "没有任何有效净热点，无法拟合绑定模型")
    candidates: list[ModelCandidate] = []
    diags: list[Diagnostic] = []
    resource_hit = False
    for key in keys:
        if key is ModelKey.DILUTION_ONLY:
            oc = fit_dilution_only(points)
        elif key is ModelKey.ONE_SITE:
            oc = fit_one_site(points, cell_volume_l, budget)
        else:
            oc = fit_two_site(points, cell_volume_l, budget)
        if oc.kind == "resource_exhausted":
            resource_hit = True
            diags.extend(oc.diagnostics)
            continue
        if oc.kind == "unresolvable":
            diags.extend(oc.diagnostics)
            continue
        if oc.value is not None:
            candidates.append(oc.value)
    if not candidates:
        if resource_hit:
            return Outcome.fail("resource_exhausted", diags)
        return Outcome.fail("unresolvable", diags or [
            Diagnostic("NO_CANDIDATES", "所有模型均无法构造候选")])
    candidates.sort(key=lambda c: (c.ssr, c.aic, MODEL_ORDER.index(c.key),
                                   c.param_fingerprint()))
    best_ssr = candidates[0].ssr
    scale = max(1.0, abs(best_ssr))
    rank = 0
    prev = None
    for c in candidates:
        if prev is None or abs(c.ssr - prev) > EQUAL_SSR_TOLERANCE * scale:
            rank += 1
        c.equivalent_rank = rank
        prev = c.ssr
    tied = [c for c in candidates if c.equivalent_rank == 1]
    unique_best = len(tied) == 1
    if not unique_best:
        diags.append(Diagnostic(
            "MULTIPLE_EQUALLY_OPTIMAL_MODELS",
            "存在多个等优模型，全部保留为合法候选，等待人工决定",
            "models:" + ",".join(c.key.value for c in tied)))
    result = FitResult(tuple(points), tuple(candidates), unique_best,
                       "ok", diags, digest)
    return Outcome.success(result)
