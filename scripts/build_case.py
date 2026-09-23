"""Build the offline domain case pack: titration thermograms (ITC).

A complete ITC narrative (not a generic spreadsheet): raw power samples,
separately versioned injection schedule, concentrations, temperature/cell
conditions, an independently measured baseline curve, and a ligand-into-buffer
blank control.

Boundary material in the initial version:
  * injections 7 and 8 are spaced 90 s apart -> overlapping post-injection peaks;
  * injection 9 has a missing-sample gap inside its window;
  * one power sample is unparseable (raw text kept + diagnostic, never dropped);
  * blank keeps main indices but omits 11 -> correspondence missing by index;
  * baseline and blank curves are separate files with separate versions.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.binding import cumulative_ligand_concentration  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "cases", "itc_titration_heatgrams", "source")

V0_UL = 1423.0
SYRINGE_M = 0.010
MACRO_M = 0.00010
N_INJ = 20
DV_UL = 2.0
DT_S = 240.0
OVERLAP_SPACING_S = 90.0
SAMPLE_DT = 2.0
TRUE_N = 1.0
TRUE_LOGK = 5.8
TRUE_DH = -1.0e7           # µcal/mol (-10 kcal/mol)
TAU_S = 22.0               # peak decay; ~90 s recovery -> 7/8 overlap
PEAK_WIDTH = 120.0
SEED = 20260924


def wiseman_one_q(x, dvol, n, logk, dh, m0, v0):
    ka = 10.0 ** logk
    a = x / (n * m0) + 1.0 + 1.0 / (ka * n * m0)
    theta = np.clip(
        0.5 * (a - np.sqrt(np.maximum(a * a - 4.0 * x / (n * m0), 0.0))),
        0.0, 1.0)
    q = v0 * dh * m0 * n * theta
    dq = np.empty_like(q)
    dq[0] = q[0]
    for i in range(1, len(q)):
        dq[i] = q[i] + dvol[i] / v0 * 0.5 * (q[i] + q[i - 1]) - q[i - 1]
    return dq


def injection_times():
    times = [60.0]
    for i in range(1, N_INJ):
        step = OVERLAP_SPACING_S if i == 7 else DT_S
        times.append(times[-1] + step)
    return times


def peak_kernel(area_ucal):
    support = np.arange(0.0, PEAK_WIDTH, SAMPLE_DT)
    shape = (support / TAU_S) * np.exp(-support / TAU_S)
    shape = shape / np.trapezoid(shape, support)
    return support, shape * area_ucal


def baseline_curve(grid, rng, noise=0.004, drift=0.00004):
    return (0.006 * np.sin(grid / 420.0)
            + drift * (grid - grid[0])
            + rng.normal(0.0, noise, size=grid.shape))


def build_power(areas, times, rng, baseline_meas, noise=0.012,
                gap_injection=9, bad_sample_near=10):
    t_end = times[-1] + 240.0
    grid = np.arange(0.0, t_end, SAMPLE_DT)
    signal = np.zeros_like(grid)
    for area, t0 in zip(areas, times):
        support, kernel = peak_kernel(area)
        idx0 = int(round(t0 / SAMPLE_DT))
        end = min(len(grid), idx0 + len(kernel))
        signal[idx0:end] += kernel[:end - idx0]
    power = signal + baseline_meas[:len(grid)]
    power += rng.normal(0.0, noise, size=grid.shape)

    rows = [("time [s]", "power [µcal/s]")]
    gap_rows = set()
    if gap_injection is not None:
        tg0 = times[gap_injection - 1] + 24.0
        gap_rows.update(range(int(tg0 / SAMPLE_DT),
                              int((tg0 + 10.0) / SAMPLE_DT)))
    bad_row = None
    if bad_sample_near is not None:
        bad_row = int((times[bad_sample_near - 1] + 14.0) / SAMPLE_DT) + 2
    for i, (tt, pp) in enumerate(zip(grid, power), start=2):
        if i - 2 in gap_rows:
            rows.append((f"{tt:.3f}", ""))            # missing sample
        elif i == bad_row:
            rows.append((f"{tt:.3f}", "N/A_µcal/s"))  # unparseable, raw kept
        else:
            rows.append((f"{tt:.3f}", f"{pp:.5f}"))
    return rows, sorted(gap_rows), bad_row


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(",".join(str(c) for c in r) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    vols = np.full(N_INJ, DV_UL)
    times = injection_times()
    x = cumulative_ligand_concentration(vols, V0_UL, SYRINGE_M)
    q = wiseman_one_q(x, vols, TRUE_N, TRUE_LOGK, TRUE_DH, MACRO_M, V0_UL)

    t_end = times[-1] + 240.0
    grid = np.arange(0.0, t_end, SAMPLE_DT)
    baseline_meas = baseline_curve(grid, rng)

    rows, gap_rows, bad_row = build_power(q, times, rng, baseline_meas)
    write_csv(os.path.join(OUT, "cell_power.csv"), rows)

    sched = [("injection", "time [s]", "volume [µL]")]
    for i, (tt, vv) in enumerate(zip(times, vols), start=1):
        sched.append((i, f"{tt:.1f}", f"{vv:.2f}"))
    write_csv(os.path.join(OUT, "injections.csv"), sched)

    write_csv(os.path.join(OUT, "concentrations_v1.csv"), [
        ("role", "compound", "concentration [mM]"),
        ("cell", "macromolecule", "0.10"),
        ("syringe", "ligand", "10.0"),
    ])
    write_csv(os.path.join(OUT, "conditions.csv"), [
        ("field", "value", "note"),
        ("temperature", "25.0 °C", "isothermal block temperature"),
        ("cell_volume", "1.423 mL", "active cell volume"),
        ("reference_power", "10 µcal/s", "reference power setting"),
    ])
    write_csv(os.path.join(OUT, "baseline_v1.csv"),
              [("time [s]", "power [µcal/s]")] +
              [(f"{tt:.3f}", f"{bb:.5f}") for tt, bb in zip(grid, baseline_meas)])

    # blank: ligand into buffer, small dilution heat (~1.5 kcal/mol-equivalent)
    blank_rng = np.random.default_rng(SEED + 1)
    blank_baseline = baseline_curve(
        np.arange(0.0, times[-1] + 240.0, SAMPLE_DT), blank_rng,
        drift=0.00003)
    blank_keep = [i for i in range(1, N_INJ + 1) if i != 11]
    blank_areas = 2.0e4 * blank_rng.normal(1.0, 0.05, size=len(blank_keep))
    blank_times = [times[i - 1] for i in blank_keep]
    brows, _, _ = build_power(blank_areas, blank_times, blank_rng,
                              blank_baseline, gap_injection=None,
                              bad_sample_near=None)
    write_csv(os.path.join(OUT, "blank_power.csv"), brows)
    bsched = [("injection", "time [s]", "volume [µL]")]
    for main_index, tt in zip(blank_keep, blank_times):
        bsched.append((main_index, f"{tt:.1f}", f"{DV_UL:.2f}"))
    write_csv(os.path.join(OUT, "blank_injections.csv"), bsched)

    meta = {
        "case_id": "itc_titration_heatgrams",
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "narrative": "biophysics lab imports raw power curves, injection timing "
                     "and a blank control; reviews per-injection integrals and "
                     "binding-model candidates.",
        "true_parameters": {"n": TRUE_N, "log10_K": TRUE_LOGK,
                            "dH_ucal_per_mol": TRUE_DH,
                            "cell_concentration_M": MACRO_M,
                            "syringe_concentration_M": SYRINGE_M,
                            "cell_volume_ul": V0_UL, "temperature_K": 298.15},
        "boundary_features": {
            "overlapping_injections": [7, 8],
            "overlap_spacing_s": OVERLAP_SPACING_S,
            "gap_injection": 9,
            "gap_sample_rows": gap_rows,
            "unparseable_sample_row": bad_row,
            "blank_missing_main_injection": 11,
            "first_injection_discarded_by_decision": 1,
        },
        "time_basis": "experiment_clock(s)",
        "files": ["cell_power.csv", "injections.csv", "concentrations_v1.csv",
                  "conditions.csv", "baseline_v1.csv", "blank_power.csv",
                  "blank_injections.csv"],
    }
    with open(os.path.join(OUT, "..", "scenario.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print("case pack written to", OUT)


if __name__ == "__main__":
    main()
