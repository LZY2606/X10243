"""生成 ITC 原始材料（功率采样/注射时序/空白），用于离线案例包。

直接以实验参数产生功率曲线 P(t)=baseline + 每注高斯峰叠加，
随后积分引擎与模型拟合在这些“原始”材料上运行，不使用任何隐藏真值。
"""
from __future__ import annotations

import math
import numpy as np

from .models_fit import one_site_q, two_site_q


def gaussian_peak(t, center, area_j, width_s):
    # area (J) 的高斯功率峰：功率单位 W
    return area_j / (width_s * math.sqrt(2 * math.pi)) * np.exp(
        -0.5 * ((t - center) / width_s) ** 2)


def make_run(*, t_end_s=1600.0, dt_s=0.5, n_inj=15, inj_interval_s=100.0,
             inj_volume_ul=10.0, width_s=18.0, baseline_uw=12.0,
             heat_j=None, noise_uw=0.0, gap_windows=(), rng_seed=7,
             start_s=40.0, extra_baseline_drift_uw_per_s=0.0):
    """返回 (points[{t,p,missing?}], injections[{time,volume}])。

    heat_j: 每注峰面积数组（J）。gap_windows: [(t0,t1), ...] 缺功率样本。
    """
    rng = np.random.default_rng(rng_seed)
    n = int(t_end_s / dt_s) + 1
    t = np.arange(n) * dt_s
    centers = start_s + np.arange(n_inj) * inj_interval_s
    if heat_j is None:
        heat_j = np.full(n_inj, -3.0e-4)
    power_w = np.full(n, baseline_uw * 1e-6)
    power_w += extra_baseline_drift_uw_per_s * 1e-6 * t
    for c, h in zip(centers, heat_j):
        power_w += gaussian_peak(t, c, h, width_s)
    power_w += rng.normal(0.0, noise_uw * 1e-6, size=n)
    missing = np.zeros(n, dtype=bool)
    for t0, t1 in gap_windows:
        missing |= (t >= t0) & (t <= t1)
    points = []
    for ti, pi, mi in zip(t, power_w, missing):
        if mi:
            points.append({"t": f"{ti:.1f} s", "p": None, "missing": True,
                           "note": "采集缺口（原始文件缺失段）"})
        else:
            points.append({"t": f"{ti:.1f} s", "p": f"{pi * 1e6:.5f} uW"})
    injections = [{"time": f"{c:.1f} s", "volume": f"{inj_volume_ul:.2f} uL"}
                  for c in centers]
    return points, injections, centers
