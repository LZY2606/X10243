"""绑定模型前向函数（一位点 / 两位点 / 稀释假设）。

浓度轴：每次注射后注射器组分在池中的标称浓度 M(i) = sum_j V_j / V0 * [M]，
dQ/d[i](i) 使用标准 Wiseman 差分（含池体积 V0/(V0-dV/2) 校正）。
单位：浓度 mM、池体积 µL、热 µcal -> kcal/mol 需要换算。
"""
from __future__ import annotations

import numpy as np

UCAL_PER_MMOL_TO_KCAL_PER_MOL = 1e-3 / 1e3 * 1e6  # µcal/mmol = 1e-3 kcal/mol


def m_titles(inj_volumes_ul: np.ndarray, syringe_mm: float, cell_volume_ul: float) -> np.ndarray:
    """每次注射后池内注射器组分的累计名义浓度（mM）。"""
    return np.cumsum(inj_volumes_ul) / cell_volume_ul * syringe_mm


def _wiseman_diff(q_cum: np.ndarray, volumes: np.ndarray, v0: float) -> np.ndarray:
    """累计池热 Q_i -> 每次注射观测量（MicroCap 形式的体积校正）。"""
    q_prev = np.concatenate(([0.0], q_cum[:-1]))
    v_mid = volumes / 2.0
    return q_cum - q_prev * (v0 - v_mid) / (v0 + v_mid)


def one_site(m_titles: np.ndarray, macromolecule_mm: float, v0_ul: float,
             n: float, k_m: float, dh_kcal: float, volumes_ul: np.ndarray) -> np.ndarray:
    # k 单位 M^-1；浓度换算 M
    mt = m_titles / 1000.0
    x0 = macromolecule_mm / 1000.0
    bound = (n * x0 / 2.0) * (
        1.0 + mt / (n * x0) + 1.0 / (k_m * n * x0)
        - np.sqrt((1.0 + mt / (n * x0) + 1.0 / (k_m * n * x0)) ** 2 - 4.0 * mt / (n * x0))
    )
    q_cum = dh_kcal * 1e6 * bound * (v0_ul / 1000.0)  # kcal/mol * mol/L * L -> µcal
    return _wiseman_diff(q_cum, volumes_ul, v0_ul)


def two_site(m_titles: np.ndarray, macromolecule_mm: float, v0_ul: float,
             n1: float, k1_m: float, dh1: float,
             n2: float, k2_m: float, dh2: float,
             volumes_ul: np.ndarray) -> np.ndarray:
    mt = m_titles / 1000.0
    x0 = macromolecule_mm / 1000.0

    def site(n, k, dh):
        b = (n * x0 / 2.0) * (
            1.0 + mt / (n * x0) + 1.0 / (k * n * x0)
            - np.sqrt((1.0 + mt / (n * x0) + 1.0 / (k * n * x0)) ** 2 - 4.0 * mt / (n * x0))
        )
        return dh * 1e6 * b * (v0_ul / 1000.0)

    return _wiseman_diff(site(n1, k1_m, dh1) + site(n2, k2_m, dh2), volumes_ul, v0_ul)


def dilution_only(m_titles: np.ndarray, q_dil_per_inj: float, _v0: float,
                  volumes_ul: np.ndarray) -> np.ndarray:
    """纯稀释假设：每注入单位体积恒定稀释热。"""
    return q_dil_per_inj * volumes_ul


MODEL_PARAMS = {
    "one_site": ["n", "log10K", "dH"],
    "two_site": ["n1", "log10K1", "dH1", "n2", "log10K2", "dH2"],
    "dilution": ["q_dil_per_uL"],
}
