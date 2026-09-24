"""离线领域案例包：滴定热谱（不是通用表格）。

生成 ITC 正向模型的合成功率曲线：一位点注射热序列 -> 指数衰减峰叠加，
内置数据缺口、NA 采样、重叠峰，空白曲线单独给出。
原始表示携带单位；所有材料在 bundles 中按版本分开。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import _theta1

CASE_TITLE = "滴定热谱：ITC 注射积分、稀释热与绑定模型审议"

# 时间 / 采样（原始单位故意混用 ucal/s、min、°C，验证规范化）
DT = 0.5                 # s
TAU = 22.0               # 峰衰减时间常数 s
RISE = 1.5               # s


def _injection_heat_sequence(
    conc_cell: float, conc_syringe: float, v0: float,
    volumes: list[float], k: float, dh: float,
) -> list[float]:
    vc = np.cumsum(np.array(volumes))
    vol = v0 + vc
    x = conc_syringe * vc / vol
    m = conc_cell * v0 / vol
    theta = _theta1(x, m, k)
    bound = np.concatenate([[0.0], theta * m])
    dq = v0 * dh * np.diff(bound)
    return [float(v) for v in dq]


def _peak(q: float, t0: float, grid: np.ndarray) -> np.ndarray:
    # q(cal) = ∫P dt；指数峰面积 = A*tau，上升段用 (1-exp(-t/rise))
    a = q / TAU
    dt = grid - t0
    safe = np.maximum(dt, 0.0)
    shape = np.where(dt >= 0, (1.0 - np.exp(-safe / RISE)) * np.exp(-safe / TAU), 0.0)
    return a * shape


def build_alpha_bundle(*, variant: str = "v1") -> dict[str, Any]:
    rng = np.random.default_rng(20260924)
    # 条件（SI 真值，但原始表示用 min / °C / uM / ucal/s）
    conc_cell = 20e-6
    conc_syringe = 240e-6
    v0 = 250e-6
    temp_c = 25.0
    k = 1.0e7
    dh = -8000.0          # cal/mol

    n_inj = 14
    base_interval = 180.0
    volumes = [6.0e-6] * n_inj
    times = []
    t_cursor = 120.0
    intervals = [base_interval] * n_inj
    # 边界：第 8 注与第 9 注重叠（间隔 < 窗）
    intervals[8] = 70.0
    for i in range(n_inj):
        times.append(t_cursor)
        t_cursor += intervals[i]
    t_end = times[-1] + 300.0

    qs = _injection_heat_sequence(conc_cell, conc_syringe, v0, volumes, k, dh)
    # 端点检验情形：最后一注给一个在窗内不衰减完的长拖尾（窗口外仍高）
    grid = np.arange(0.0, t_end, DT)
    power = np.zeros_like(grid)
    for t0, q in zip(times, qs):
        power += _peak(q, t0, grid)
    # 第 13 注叠加固定面积(-3 ucal)的长拖尾：积分窗内不回归基线 -> 端点检验
    t13 = times[12]
    safe = np.maximum(grid - t13, 0.0)
    tail_blob = np.where(grid >= t13,
                         (-20.0e-6 / 300.0) * (1.0 - np.exp(-safe / 1.5)) * np.exp(-safe / 300.0), 0.0)
    power += tail_blob
    noise = rng.normal(0.0, 0.02e-6, size=grid.shape)  # 0.02 ucal/s
    baseline_level = 0.10e-6
    power += baseline_level + noise

    # 数据缺口：物理移除一段时间窗（第 11 注窗内）制造 gap
    gap_t0, gap_t1 = times[10] + 30.0, times[10] + 48.0
    keep = ~((grid >= gap_t0) & (grid <= gap_t1))
    # NA token：单点缺失（保留行、值为 "NA"），第 5 注上升沿
    na_index = int((times[4] + 3.0) / DT)
    samples = []
    for gi, tt in enumerate(grid):
        if not keep[gi]:
            continue
        p_ucal = power[gi] * 1e6  # cal/s -> ucal/s
        if gi == na_index:
            samples.append([round(tt / 60.0, 6), "NA"])   # 时间用 min
        else:
            samples.append([round(tt / 60.0, 6), round(p_ucal, 5)])

    injections = [
        {"index": i + 1, "time": round(times[i] / 60.0, 6), "volume": round(volumes[i] * 1e6, 4)}
        for i in range(n_inj)
    ]

    baseline_block = {
        "time_unit": "s", "power_unit": "ucal/s", "method": "piecewise_linear",
        "anchors": [
            {"t": 0.0, "p": 0.10},
            {"t": times[n_inj // 2], "p": 0.10},
            {"t": t_end, "p": 0.10},
        ],
    }

    # 空白：故意少最后两注，触发“空白对应缺失”；稀释热为小的常值注射热
    blank_n = n_inj - 2
    blank_times = times[:blank_n]
    q_blank = -3.0e-6  # -3 ucal 每注（机械/稀释热）
    bpower = np.zeros_like(grid)
    for t0 in blank_times:
        a = q_blank / TAU
        dt = grid - t0
        safe = np.maximum(dt, 0.0)
        shape = np.where(dt >= 0, (1.0 - np.exp(-safe / 1.5)) * np.exp(-safe / TAU), 0.0)
        bpower += a * shape
    bpower += 0.05e-6 + rng.normal(0.0, 0.02e-6, size=grid.shape)
    bkeep = np.ones_like(grid, dtype=bool)
    blank_samples = []
    for gi, tt in enumerate(grid):
        if not bkeep[gi]:
            continue
        blank_samples.append([round(tt / 60.0, 6), round(bpower[gi] * 1e6, 5)])
    blank_injections = [
        {"index": i + 1, "time": round(blank_times[i] / 60.0, 6), "volume": 6.0}
        for i in range(blank_n)
    ]

    bundle: dict[str, Any] = {
        "case": CASE_TITLE,
        "schema_version": "itc-case/1.0",
        "power_series": {
            "time_unit": "min", "power_unit": "ucal/s",
            "samples": samples,
            "note": "第8/9注重叠；第11注窗内有数据缺口；第5注上升沿单点 NA；最后一注长拖尾触发端点检验",
        },
        "injection_schedule": {"time_unit": "min", "volume_unit": "uL", "injections": injections},
        "concentrations": {"conc_unit": "uM", "cell": 20.0, "syringe": 240.0},
        "conditions": {
            "temperature": temp_c, "temperature_unit": "°C",
            "cell_volume": 250.0, "cell_volume_unit": "uL",
        },
        "baseline": baseline_block,
        "blank_run": {
            "power_series": {"time_unit": "min", "power_unit": "ucal/s", "samples": blank_samples},
            "injection_schedule": {"time_unit": "min", "volume_unit": "uL",
                                   "injections": blank_injections},
            "note": "空白仅含前 12 注，末尾两注无对应",
            "baseline": {"time_unit": "s", "power_unit": "ucal/s", "method": "piecewise_linear",
                         "anchors": [{"t": 0.0, "p": 0.05}, {"t": t_end, "p": 0.05}]},
        },
    }

    if variant == "v2_concentration":
        # 浓度修订：只改浓度（源数据修订），积分不变、模型应过期
        bundle["concentrations"] = {"conc_unit": "uM", "cell": 20.0, "syringe": 210.0}
        bundle["variant_note"] = "注射器浓度 240 -> 210 uM 修订"
    if variant == "v2_baseline_local":
        # 局部依赖的源改动：只抬高第 11 注附近基线锚点
        bundle["baseline"] = {
            "time_unit": "s", "power_unit": "ucal/s", "method": "piecewise_linear",
            "anchors": [
                {"t": 0.0, "p": 0.10},
                {"t": times[n_inj // 2], "p": 0.10},
                {"t": times[10] - 5.0, "p": 0.10},
                {"t": times[10] + 25.0, "p": 0.18},
                {"t": times[10] + 55.0, "p": 0.10},
                {"t": t_end, "p": 0.10},
            ],
        }
        bundle["variant_note"] = "第 11 注附近基线局部修订"
    return bundle


def build_tie_bundle() -> dict[str, Any]:
    """无绑定信号、仅常值稀释热+较大噪声：一位点与稀释假设等优。"""
    rng = np.random.default_rng(7)
    n_inj = 12
    interval = 180.0
    volumes = [6.0e-6] * n_inj
    times = [120.0 + i * interval for i in range(n_inj)]
    t_end = times[-1] + 260.0
    grid = np.arange(0.0, t_end, DT)
    power = np.zeros_like(grid)
    for t0 in times:
        a = (-2.0e-6) / TAU
        dt = grid - t0
        safe = np.maximum(dt, 0.0)
        shape = np.where(dt >= 0, (1.0 - np.exp(-safe / 1.5)) * np.exp(-safe / TAU), 0.0)
        power += a * shape
    power += 0.20e-6 + rng.normal(0.0, 0.25e-6, size=grid.shape)  # 噪声大
    samples = [[round(tt / 60.0, 6), round(power[i] * 1e6, 5)] for i, tt in enumerate(grid)]
    return {
        "case": CASE_TITLE + "（等优分支）",
        "schema_version": "itc-case/1.0",
        "power_series": {"time_unit": "min", "power_unit": "ucal/s", "samples": samples},
        "injection_schedule": {
            "time_unit": "min", "volume_unit": "uL",
            "injections": [{"index": i + 1, "time": round(times[i] / 60.0, 6), "volume": 6.0}
                           for i in range(n_inj)],
        },
        "concentrations": {"conc_unit": "uM", "cell": 20.0, "syringe": 240.0},
        "conditions": {"temperature": 25.0, "temperature_unit": "°C",
                       "cell_volume": 250.0, "cell_volume_unit": "uL"},
        "baseline": {"time_unit": "s", "power_unit": "ucal/s", "method": "piecewise_linear",
                     "anchors": [{"t": 0.0, "p": 0.20}, {"t": t_end, "p": 0.20}]},
        "blank_run": {
            "power_series": {"time_unit": "min", "power_unit": "ucal/s",
                             "samples": [[round(tt / 60.0, 6), round(0.0 * p + 0.20, 5)]
                                         for tt, p in zip(grid, power)]},
            "injection_schedule": {
                "time_unit": "min", "volume_unit": "uL",
                "injections": [{"index": i + 1, "time": round(times[i] / 60.0, 6), "volume": 6.0}
                               for i in range(n_inj)],
            },
            "baseline": {"time_unit": "s", "power_unit": "ucal/s", "method": "piecewise_linear",
                         "anchors": [{"t": 0.0, "p": 0.20}, {"t": t_end, "p": 0.20}]},
        },
    }
