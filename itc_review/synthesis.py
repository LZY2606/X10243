"""离线案例包的原始功率曲线合成（确定性，无随机）。

用与拟合一致的前向模型生成每注射热，再卷积成“µW-秒”功率轨迹，
保证案例自洽：一位点/两位点数据能拟合回原参数。
"""
from __future__ import annotations

import math

import numpy as np

from .models import dilution_only, m_titles, one_site, two_site

# µW*s = µJ；µcal -> µJ
UJ_PER_UCAL = 4.184


def _kernel(t: np.ndarray, tau_s: float) -> np.ndarray:
    return np.exp(-np.maximum(t, 0.0) / tau_s) / tau_s


def build_power(times_s: np.ndarray, injection_centers: list[float], heats_ucal: list[float],
                tau_s: float = 18.0, baseline_fn=None) -> np.ndarray:
    p = np.zeros_like(times_s, dtype=float)
    if baseline_fn is not None:
        p += baseline_fn(times_s)
    for center, heat in zip(injection_centers, heats_ucal):
        impulse_uw = heat * UJ_PER_UCAL
        p += impulse_uw * _kernel(times_s - center, tau_s)
    return p


def heats_for(model: str, volumes_ul: np.ndarray, syringe_mm: float, cell_mm: float,
              v0_ul: float, params: dict) -> np.ndarray:
    mt = m_titles(volumes_ul, syringe_mm, v0_ul)
    if model == "one_site":
        return one_site(mt, cell_mm, v0_ul, params["n"], 10 ** params["log10K"], params["dH"], volumes_ul)
    if model == "two_site":
        return two_site(mt, cell_mm, v0_ul,
                        params["n1"], 10 ** params["log10K1"], params["dH1"],
                        params["n2"], 10 ** params["log10K2"], params["dH2"], volumes_ul)
    return dilution_only(mt, params["q_dil_per_uL"], v0_ul, volumes_ul)
