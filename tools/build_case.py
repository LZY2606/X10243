"""Build the offline domain case pack ``cases/itc_titration_review/``.

This is genuine ITC material, not a generic spreadsheet:

  * power sampling traces (with a lead-in and one retained unconvertible cell)
  * injection times + volumes (small prime first injection, an overlapping
    injection pair, per-volume metadata)
  * concentrations (v1 + a later revision)
  * temperature reference and cell volume
  * separately-versioned baseline curve
  * blank control curve with its own injection schedule

A fixed RNG seed makes the review conclusions reproducible.  Run:

    .venv/bin/python tools/build_case.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from itc_app.binding import BindingInput, fit_models, predict_one_site  # noqa: E402

CASE_DIR = os.path.join(ROOT, "cases", "itc_titration_review")

N = 20
VOL_L = 5.0                      # microlitres per regular injection
PRIME_VOL_UL = 2.0
CELL_ML = 1.4
LIGAND_MM = 0.50
LIGAND_MM_REV = 0.47
MACRO_UM = 20.0
TEMP_C = 25.0
DT = 0.5
TAU = 8.0                        # seconds
PERIOD = 120.0
LEAD = 10.0
TAIL = 110.0


def volumes():
    return [PRIME_VOL_UL] + [VOL_L] * (N - 1)


def times():
    t = [0.0]
    for i in range(1, N):
        if i == 6:                       # injection #7
            t.append(t[-1] + PERIOD)
        elif i == 7:                     # injection #8 deliberately overlaps #7
            t.append(t[-1] + 25.0)
        else:
            t.append(t[-1] + PERIOD)
    return t


def make_heats(seed, k, n_sites, dh, noise_frac, constant_offset=0.0):
    vols = [v * 1e-9 for v in volumes()]
    bi = BindingInput(
        heats_joules=[0.0] * N,
        injection_volumes_m3=vols,
        ligand_conc_syringe_molm3=LIGAND_MM,
        macromolecule_conc_cell_molm3=MACRO_UM / 1000.0,
        cell_volume_m3=CELL_ML * 1e-6,
        temperature_k=TEMP_C + 273.15,
        injection_indices=list(range(1, N + 1)),
    )
    y = np.asarray(predict_one_site(bi, k=k, n=n_sites, dh=dh)) + constant_offset
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0.0, noise_frac * float(np.max(np.abs(y))), N)
    return y.tolist()


def find_tie_noise(seed=7):
    """Noise fraction where one_site and dilution fall in the same AICc band."""
    lo, hi = 0.02, 0.6
    chosen = None
    for _ in range(16):
        mid = (lo + hi) / 2
        heats = make_heats(seed, k=2000.0, n_sites=1.0, dh=-25000.0,
                           noise_frac=mid)
        vols = [v * 1e-9 for v in volumes()]
        bi = BindingInput(heats_joules=[h for i, h in enumerate(heats) if i != 0],
                          injection_volumes_m3=vols[1:],
                          ligand_conc_syringe_molm3=LIGAND_MM,
                          macromolecule_conc_cell_molm3=MACRO_UM / 1000.0,
                          cell_volume_m3=CELL_ML * 1e-6,
                          temperature_k=TEMP_C + 273.15,
                          injection_indices=list(range(2, N + 1)))
        res = fit_models(bi)
        by = {c["model"]: c for c in res["candidates"]}
        delta = by["dilution"]["aicc"] - by["one_site"]["aicc"]
        if delta > 1.5:
            lo = mid
        elif delta < 0.3:
            hi = mid
        else:
            chosen = mid
            break
    if chosen is None:
        chosen = (lo + hi) / 2
    return chosen


def baseline_drift(t):
    """Slow power drift in uW captured by the separate baseline material."""
    return 10.0 + 0.0020 * t


def render_trace(path, times_inj, heats, *, include_baseline_drift, blank=False,
                 gap_at=None, unconvertible_at=-LEAD / 2, local_patch=None,
                 vol_weights=None):
    t_start = -LEAD
    t_end = times_inj[-1] + TAIL
    grid = np.round(np.arange(t_start, t_end + DT / 2, DT), 3)
    lines = []
    if blank:
        lines.append("ITC-BLANK/1")
    else:
        lines.append("ITC-POWER/1")
    lines.append("time-unit: s")
    lines.append("power-unit: uW")
    lines.append(f"time-origin: inj1@0.0s")
    lines.append("valid-from: -10 s")
    lines.append(f"valid-to: {t_end:g} s")
    lines.append("@data")
    inj = np.asarray(times_inj)
    for t in grid:
        p = 0.0
        # post-injection peaks are allowed to overlap: sum all active decays
        active = t >= inj
        if np.any(active):
            idx = np.where(active)[0]
            weights = vol_weights if vol_weights is not None else [1.0] * len(inj)
            decays = np.sum([heats[i] * weights[i] / TAU
                             * np.exp(-(t - inj[i]) / TAU) for i in idx])
            p += decays * 1e6            # W -> uW (amp = Q/tau)
        if include_baseline_drift:
            p += baseline_drift(t)
        rng = np.random.default_rng(int(round((t + LEAD) * 1000)) % (2**31))
        p += rng.normal(0.0, 0.02)
        if local_patch is not None:
            lo_t, hi_t, bump = local_patch
            if lo_t <= t < hi_t:
                p += bump
        token = f"{p:.5f}"
        if gap_at is not None and abs(t - gap_at) < DT / 2:
            token = "NA"
        if unconvertible_at is not None and abs(t - unconvertible_at) < DT / 2:
            token = "12.3 furlong/h"       # retained in place with diagnostic
        lines.append(f"{t:g} {token}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if blank:
        lines2 = ["@injections"]
        for i, t in enumerate(times_inj, start=1):
            lines2.append(f"{i} {t:g} {volumes()[i-1]:g}")
        with open(path, "a") as fh:
            fh.write("\n".join(lines2) + "\n")


def render_injections(path, times_inj, suffix=""):
    lines = ["ITC-INJECTIONS/1", "time-unit: s", "volume-unit: uL",
             f"time-origin: inj1@0.0s{suffix}", "@injections"]
    for i, (t, v) in enumerate(zip(times_inj, volumes()), start=1):
        note = "  # prime injection, reviewer may discard" if i == 1 else ""
        lines.append(f"{i} {t:g} {v:g}{note}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def render_header_file(path, magic, pairs, section=None):
    lines = [magic]
    for k, v in pairs:
        lines.append(f"{k}: {v}")
    if section:
        lines.append(section[0])
        lines.extend(section[1])
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def render_baseline(path):
    t_end = times()[-1] + TAIL
    grid = np.arange(-LEAD, t_end + DT / 2, DT)
    lines = ["ITC-BASELINE/1", "time-unit: s", "power-unit: uW",
             "baseline-version: v1", "time-origin: inj1@0.0s",
             "valid-from: -10 s", f"valid-to: {t_end:g} s", "@data"]
    for t in grid:
        lines.append(f"{t:g} {baseline_drift(t):.5f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(CASE_DIR, exist_ok=True)
    inj_times = times()

    # ---- Run A: clear one-site titration -------------------------------
    heats_a = make_heats(11, k=2000.0, n_sites=1.0, dh=-25000.0, noise_frac=0.004)
    weights = [v / VOL_L for v in volumes()]
    render_trace(os.path.join(CASE_DIR, "runA_power_v1.itcpower"),
                 inj_times, heats_a, include_baseline_drift=True,
                 gap_at=inj_times[11] + 55.0, vol_weights=weights)
    # local-only correction: window of injection #10 only
    render_trace(os.path.join(CASE_DIR, "runA_power_v2.itcpower"),
                 inj_times, heats_a, include_baseline_drift=True,
                 gap_at=inj_times[11] + 55.0,
                 local_patch=(inj_times[9], inj_times[10], 0.6),
                 vol_weights=weights)
    render_injections(os.path.join(CASE_DIR, "runA_injections_v1.itcinj"),
                      inj_times)
    render_header_file(os.path.join(CASE_DIR, "concentration_v1.itcconc"),
                       "ITC-CONCENTRATION/1",
                       [("ligand_syringe", f"{LIGAND_MM}"),
                        ("ligand-unit", "mM"),
                        ("macromolecule_cell", f"{MACRO_UM}"),
                        ("macromolecule-unit", "uM"),
                        ("concentration-version", "v1")])
    render_header_file(os.path.join(CASE_DIR, "concentration_v2.itcconc"),
                       "ITC-CONCENTRATION/1",
                       [("ligand_syringe", f"{LIGAND_MM_REV}"),
                        ("ligand-unit", "mM"),
                        ("macromolecule_cell", f"{MACRO_UM}"),
                        ("macromolecule-unit", "uM"),
                        ("concentration-version", "v2"),
                        ("revision-reason", "stock re-quantified after prep")])
    render_header_file(os.path.join(CASE_DIR, "temperature_v1.itctemp"),
                       "ITC-TEMPERATURE/1",
                       [("temperature", f"{TEMP_C}"), ("unit", "C"),
                        ("temperature-version", "v1")])
    render_header_file(os.path.join(CASE_DIR, "cell_v1.itccell"), "ITC-CELL/1",
                       [("cell-volume", f"{CELL_ML}"), ("unit", "mL"),
                        ("cell-version", "v1")])
    render_baseline(os.path.join(CASE_DIR, "baseline_v1.itcbase"))

    # blank: constant small dilution heat per injection, own schedule,
    # unconvertible cell retained in the lead-in where it cannot be dropped.
    blank_heats = [8.0e-5 if i else 4.0e-5 for i in range(N)]
    render_trace(os.path.join(CASE_DIR, "blank_v1.itcblank"),
                 inj_times, blank_heats, include_baseline_drift=False,
                 blank=True, vol_weights=weights)

    # ---- Run B: ambiguous one_site vs dilution titration ----------------
    tie_frac = find_tie_noise()
    heats_b = make_heats(7, k=2000.0, n_sites=1.0, dh=-25000.0,
                         noise_frac=tie_frac)
    render_trace(os.path.join(CASE_DIR, "runB_power_v1.itcpower"),
                 inj_times, heats_b, include_baseline_drift=True,
                 vol_weights=weights)
    render_injections(os.path.join(CASE_DIR, "runB_injections_v1.itcinj"),
                      inj_times)

    expected = {
        "case": "itc_titration_review",
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "rule_versions": {
            "integration": "integrate-v1.0",
            "binding": "binding-v1.0",
            "blank_correspondence": "blank-corr-v1",
        },
        "run_a": {
            "injections": N,
            "prime_injection": 1,
            "overlapping_injections": [7, 8],
            "gap_injection": 12,
            "gap_time_s": inj_times[11] + 55.0,
            "local_correction_injection": 10,
            "expected_model": "one_site",
            "expected_K_m3_per_mol": 2000.0,
            "expected_dH_J_per_mol": -25000.0,
        },
        "run_b": {
            "tie_noise_fraction": tie_frac,
            "min_tie_models": ["one_site", "dilution"],
        },
        "units": {
            "power": "uW source / W derived",
            "time": "s relative to inj1@0.0s",
            "volume": "uL source / m3 derived",
            "concentration": "mM,uM source / mol m-3 derived",
            "temperature": "C source / K derived",
            "heat": "J derived",
        },
    }
    with open(os.path.join(CASE_DIR, "expected.json"), "w") as fh:
        json.dump(expected, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print("case pack written to", CASE_DIR)
    print("runB tie noise fraction:", tie_frac)


if __name__ == "__main__":
    main()
