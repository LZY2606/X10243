"""Synthetic, domain-specific offline ITC case material.

NOT a generic spreadsheet: generates instrument-style power traces (uW vs s),
injection timing/volumes, syringe/cell concentrations, temperature, cell
volume, separate baseline and a separate blank control trace - everything the
"滴定热谱" narrative needs.

The base case is a clean one-site titration; archive branches mutate exactly
one direct source at a time so blast radius can be audited.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import thermo

# Instrument settings of the bundled experiment
V0_U_L = 200.0
M0_U_M = 20.0
SYRINGE_U_M = 400.0
TEMP_C = 25.0
N_INJECTIONS = 16
INJ_VOLUME_U_L = 5.0
SPACING_S = 240.0
BASELINE_U_W = 0.0
DT_S = 1.0

# ground truth binding (one strong site)
TRUE_K_1_PER_M = 1.0e6          # = 1 1/uM
TRUE_DH_KCAL = -8.0
TRUE_N = 1.0


def ground_truth_heats(n: int = N_INJECTIONS,
                       volume: float = INJ_VOLUME_U_L,
                       v0: float = V0_U_L, m0: float = M0_U_M,
                       syringe: float = SYRINGE_U_M,
                       k_1_per_m: float = TRUE_K_1_PER_M,
                       dh: float = TRUE_DH_KCAL, nsites: float = TRUE_N,
                       dilution_per_uL: float = 0.02,
                       dilution_offset: float = 0.05,
                       seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dv = np.full(n, volume)
    x = np.cumsum(dv) / v0 * syringe
    k_um = k_1_per_m * 1e-6
    q_bind = thermo.wiseman_q(x, m0, v0, dv,
                              (math.log10(k_um), dh, nsites))
    q_dil = dilution_offset + dilution_per_uL * dv
    q = q_bind + q_dil + rng.normal(0.0, 0.02, n)
    return {"q_uJ": q.tolist(), "q_binding_uJ": q_bind.tolist(),
            "q_dilution_uJ": q_dil.tolist(), "dv_uL": dv.tolist(),
            "x_uM": x.tolist()}


def peak_shape(t: np.ndarray, area_uJ: float, *, tau_s: float = 18.0,
               rise_s: float = 1.2) -> np.ndarray:
    """Instrument response: sharp rise then exponential decay (exothermic <0)."""
    rise = 1.0 - np.exp(-t / rise_s)
    rise[t < 0] = 0.0
    decay = np.exp(-np.maximum(t, 0) / tau_s)
    shape = rise * decay
    # normalize to unit area so peak integral equals requested area
    tt = np.arange(0, SPACING_S + 60, DT_S)
    unit = (1.0 - np.exp(-tt / rise_s)) * np.exp(-tt / tau_s)
    norm = np.trapezoid(unit, tt)
    return area / norm * shape


def power_trace(n: int = N_INJECTIONS, *, spacing: float = SPACING_S,
                dt: float = DT_S, volumes: np.ndarray | None = None,
                areas: np.ndarray | None = None, noise: float = 0.03,
                tau_s: float = 18.0, baseline: float = BASELINE_U_W,
                seed: int = 11, extra_tail: float = 200.0,
                overlap: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if volumes is None:
        volumes = np.full(n, INJ_VOLUME_U_L)
    if areas is None:
        gt = ground_truth_heats(n)
        areas = np.array(gt["q_uJ"])
    this_spacing = spacing * 0.35 if overlap else spacing
    times_inj = np.arange(n) * this_spacing
    t_end = times_inj[-1] + spacing + extra_tail
    t = np.arange(0.0, t_end, dt)
    p = np.full_like(t, baseline)
    tt = np.arange(0, spacing + 60, dt)
    unit = (1.0 - np.exp(-tt / 1.2)) * np.exp(-tt / tau_s)
    norm = np.trapezoid(unit, tt)
    for t0, area in zip(times_inj, areas):
        mask = t >= t0
        m = mask.sum()
        shape = area / norm * (1.0 - np.exp(-np.arange(m) * dt / 1.2)) * \
            np.exp(-np.arange(m) * dt / tau_s)
        p[mask] += shape
    p += rng.normal(0.0, noise, len(t))
    samples = [{"t": float(tt_), "p": float(pp)}
               for tt_, pp in zip(t, p)]
    injections = [{"index": i + 1, "time": {"value": float(t0), "unit": "s"},
                  "volume": {"value": float(v), "unit": "uL"}}
                 for i, (t0, v) in enumerate(zip(times_inj, volumes))]
    return {"samples": samples, "injections": injections,
            "injection_times": times_inj.tolist(), "dt_s": dt}


def blank_trace(n: int = N_INJECTIONS, *, volume: float = INJ_VOLUME_U_L,
                dilution_per_uL: float = 0.02,
                dilution_offset: float = 0.05, seed: int = 23
                ) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dv = np.full(n, volume)
    areas = (dilution_offset + dilution_per_uL * dv +
             rng.normal(0.0, 0.015, n))
    tr = power_trace(n, volumes=dv, areas=areas, noise=0.02, tau_s=15.0,
                     seed=seed + 1)
    tr["areas_uJ"] = areas.tolist()
    return tr


def base_bundle() -> dict[str, Any]:
    """The complete initial import: seven separately versioned streams."""
    tr = power_trace()
    bl = blank_trace()
    return {
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "power": {"samples": tr["samples"], "time_origin": "first_sample",
                  "sampling_interval": {"value": tr["dt_s"], "unit": "s"}},
        "injections": tr["injections"],
        "concentrations": {"syringe": {"value": SYRINGE_U_M, "unit": "uM"},
                           "cell": {"value": M0_U_M, "unit": "uM"}},
        "temperature": {"value": TEMP_C, "unit": "C"},
        "cell_volume": {"value": V0_U_L, "unit": "uL"},
        "baseline": {"kind": "constant",
                     "value": {"value": BASELINE_U_W, "unit": "uW"}},
        "blank": {
            "power": {"samples": bl["samples"], "time_origin": "first_sample",
                      "sampling_interval": {"value": bl["dt_s"], "unit": "s"}},
            "injections": [{"index": i + 1,
                            "time": inj["time"], "volume": inj["volume"]}
                           for i, inj in enumerate(bl["injections"])],
            "match_rule": "by_index"},
    }
