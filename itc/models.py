"""Binding-model candidates: one-site, two-site (multi-site) and dilution-only.

Rules:
  - all legally equivalent optima are kept (multiple equivalent models are a
    first-class result, not collapsed to one winner)
  - every candidate carries a stable id derived from its inputs digest, so
    re-import / recompute of equivalent inputs yields a stable order
  - fits record their parameter version and parent dependencies
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

PARAMS_VERSION = "bindfit/1"
EQUIV_REL_TOL = 0.02  # rss <= best * (1 + tol) => equivalent optimum


def _bound_complex(ka: float, xt: np.ndarray, mt: np.ndarray) -> np.ndarray:
    """Bound-complex concentration for 1:1 binding (quadratic root)."""
    b = 1.0 / ka + xt + mt
    disc = np.maximum(b * b - 4.0 * xt * mt, 0.0)
    return (b - np.sqrt(disc)) / 2.0


def _site_heats(ka: float, dh_kj: float, xt: np.ndarray, mt: np.ndarray,
                v0_uL: float, dv_uL: np.ndarray) -> np.ndarray:
    """Per-injection heat (uJ) for one site with cell dilution correction."""
    xb = _bound_complex(ka, xt, mt)
    xb_prev = np.concatenate(([0.0], xb[:-1]))
    dil = np.concatenate(([1.0], 1.0 - dv_uL[:-1] / v0_uL))
    # uM * uL = pmol ; pmol * kJ/mol = 1e-3 uJ -> factor 1e-3
    return dh_kj * v0_uL * (xb - xb_prev * dil) * 1e-3


def cumulative_concs(n: int, xs_uM: float, m0_uM: float, v0_uL: float, dv_uL: np.ndarray):
    """Total ligand / macromolecule concentration in the cell after each injection."""
    cumvol = np.cumsum(dv_uL)
    xt = xs_uM * cumvol / v0_uL
    mt = m0_uM * (1.0 - cumvol / v0_uL)
    return xt, np.maximum(mt, 0.0)


def _linear_solve(design: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    rss = float(np.sum((y - pred) ** 2))
    return coef, pred, rss


def _grid1(y, xt, mt, v0, dv, kas):
    best = None
    for ka in kas:
        g = _site_heats(ka, 1.0, xt, mt, v0, dv)
        design = np.column_stack([g, np.ones_like(g)])
        (dh, q0), _, rss = _linear_solve(design, y)
        if best is None or rss < best[0]:
            best = (rss, ka, dh, q0)
    return best


def fit_candidates(heats_uJ: list, xs_uM: float, m0_uM: float, v0_uL: float,
                   dv_uL: list, inputs_digest: str) -> dict:
    """Fit one-site / two-site / dilution-only and keep all equivalent optima."""
    y = np.asarray(heats_uJ, dtype=float)
    dv = np.asarray(dv_uL, dtype=float)
    xt, mt = cumulative_concs(len(y), xs_uM, m0_uM, v0_uL, dv)
    kas = np.logspace(3.0, 8.0, 41)

    rss1, ka1, dh1, q01 = _grid1(y, xt, mt, v0_uL, dv, kas)
    fine = np.logspace(np.log10(ka1) - 0.3, np.log10(ka1) + 0.3, 25)
    r = _grid1(y, xt, mt, v0_uL, dv, fine)
    if r[0] < rss1:
        rss1, ka1, dh1, q01 = r

    best2 = None
    kas2 = np.logspace(3.0, 8.0, 17)
    for ka_a in kas2:
        ga = _site_heats(ka_a, 1.0, xt, mt, v0_uL, dv)
        for ka_b in kas2:
            gb = _site_heats(ka_b, 1.0, xt, mt, v0_uL, dv)
            design = np.column_stack([ga, gb, np.ones_like(ga)])
            (dha, dhb, q0), _, rss = _linear_solve(design, y)
            if best2 is None or rss < best2[0]:
                best2 = (rss, ka_a, ka_b, dha, dhb, q0)
    rss2, ka_a, ka_b, dha, dhb, q02 = best2

    qd = float(np.mean(y))
    rssd = float(np.sum((y - qd) ** 2))

    cands = [
        {"model": "one_site",
         "params": {"Ka_1_per_uM": ka1, "dH1_kJ_mol": dh1, "q0_uJ": q01},
         "rss": rss1},
        {"model": "two_site",
         "params": {"Ka_1_per_uM": ka_a, "Ka_2_per_uM": ka_b,
                    "dH1_kJ_mol": dha, "dH2_kJ_mol": dhb, "q0_uJ": q02},
         "rss": rss2},
        {"model": "dilution_only",
         "params": {"q0_uJ": qd},
         "rss": rssd},
    ]
    best_rss = min(c["rss"] for c in cands)
    for c in cands:
        c["equivalent"] = bool(c["rss"] <= best_rss * (1.0 + EQUIV_REL_TOL))
        cid_src = json.dumps({"m": c["model"], "in": inputs_digest}, sort_keys=True)
        c["candidate_id"] = "cand:" + hashlib.sha256(cid_src.encode()).hexdigest()[:12]
    cands.sort(key=lambda c: (not c["equivalent"], c["rss"], c["model"]))
    return {"params_version": PARAMS_VERSION, "candidates": cands,
            "equivalent_count": sum(1 for c in cands if c["equivalent"])}


def predicted_heats(candidate: dict, xs_uM: float, m0_uM: float, v0_uL: float,
                    dv_uL: list) -> list:
    dv = np.asarray(dv_uL, dtype=float)
    xt, mt = cumulative_concs(len(dv), xs_uM, m0_uM, v0_uL, dv)
    p = candidate["params"]
    if candidate["model"] == "one_site":
        h = _site_heats(p["Ka_1_per_uM"], p["dH1_kJ_mol"], xt, mt, v0_uL, dv) + p["q0_uJ"]
    elif candidate["model"] == "two_site":
        h = (_site_heats(p["Ka_1_per_uM"], p["dH1_kJ_mol"], xt, mt, v0_uL, dv)
             + _site_heats(p["Ka_2_per_uM"], p["dH2_kJ_mol"], xt, mt, v0_uL, dv)
             + p["q0_uJ"])
    else:
        h = np.full(len(dv), p["q0_uJ"])
    return [float(v) for v in h]
