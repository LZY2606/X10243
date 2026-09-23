"""Deterministic synthetic ITC physics for the offline case pack."""

from __future__ import annotations

import math
import numpy as np

UCAL_S = 4.184e-6  # W per microcal/s


def theta(x, c, n=1.0):
    a = x + n + 1.0 / c
    return 0.5 * (a - np.sqrt(np.maximum(a * a - 4.0 * n * x, 0.0)))


def binding_heats_j(n_inj, v_l, V_l, M0, L0, k, dH_j, *, n_sites=1.0,
                    disp=True):
    vols = np.full(n_inj, v_l, dtype=float)
    cum = np.cumsum(vols)
    if disp:
        Lt = L0 * (cum / V_l) / (1.0 - cum / V_l)
    else:
        Lt = L0 * cum / V_l
    x = Lt / M0
    c = k * M0
    th = theta(x, c, n_sites)
    Q = M0 * V_l * dH_j * th
    Qp = np.concatenate([[0.0], Q[:-1]])
    dq = Q - Qp + (vols / (2.0 * V_l)) * (Q + Qp)
    return dq, vols, x, th


def power_trace(times_s, inject_times_s, heats_j, tau_s=60.0, noise_w=0.0,
                seed=0):
    rng = np.random.default_rng(seed)
    power = np.zeros_like(times_s, dtype=float)
    t = np.asarray(times_s, dtype=float)
    for ti, q in zip(inject_times_s, heats_j):
        amp = q / tau_s  # W ; integral of amp*exp(-dt/tau) = amp*tau = q
        mask = t >= ti
        power[mask] += amp * np.exp(-(t[mask] - ti) / tau_s)
    if noise_w:
        power = power + rng.normal(0.0, noise_w, size=power.shape)
    return power


def make_schedule(n_inj, spacing_s, *, start_s=120.0, v_l=5e-6, skip=()):
    inj, times = [], []
    ordinal = 0
    for i in range(n_inj):
        ordinal += 1
        if ordinal in skip:
            continue
        times.append(start_s + i * spacing_s)
        inj.append((ordinal, times[-1], v_l))
    return inj
