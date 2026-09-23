"""Build the committed offline domain case pack:

cases/titration-heat-review/v1/materials/<series>/{power,injections,...}.csv

The story is one ITC titration session with:
  main    - 20 injections, clean one-site thermogram + dilution blank
  overlap - tight spacing so peaks overlap (windows clip at next injection)
  gap     - a data gap around one injection + one unconvertible power token
  tie     - engineered so one-site, two-site and dilution are near-equivalent
"""

from __future__ import annotations

import os
import numpy as np

from tools.itc_synth import UCAL_S, binding_heats_j, make_schedule, power_trace

ROOT = os.path.join("cases", "titration-heat-review", "v1", "materials")
DT = 2.0


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def power_csv(times, power_w, *, bad_index=None):
    lines = ["# time_unit: s", "# power_unit: µcal/s", "time,power"]
    for i, (t, p) in enumerate(zip(times, power_w)):
        if bad_index is not None and i in bad_index:
            lines.append(f"{t:g},#N/A")
        else:
            lines.append(f"{t:g},{p / UCAL_S:.6g}")
    return "\n".join(lines) + "\n"


def schedule_csv(inj):
    lines = ["# time_unit: s", "# volume_unit: µL", "ordinal,time,volume"]
    for ordinal, t, v in inj:
        lines.append(f"{ordinal},{t:g},{v * 1e6:g}")
    return "\n".join(lines) + "\n"


def concentrations_csv(ligand_m, macro_m, *, ligand_unit="µM", macro_unit="µM"):
    return ("# version: 1\nrole,value,unit\n"
            f"ligand,{ligand_m * _unit(ligand_unit):g},{ligand_unit}\n"
            f"macromolecule,{macro_m * _unit(macro_unit):g},{macro_unit}\n")


def _unit(u):
    return {"µM": 1e6, "mM": 1e3, "M": 1.0, "nM": 1e9}[u]


def conditions_csv(temp_c=25.0, cell_ml=1.4):
    return ("# version: 1\nkey,value,unit\n"
            f"temperature,{temp_c:g},°C\ncell_volume,{cell_ml:g},mL\n")


def baseline_csv(anchors):
    lines = ["# time_unit: s", "# power_unit: µcal/s", "# method: linear_anchors",
             "time,level"]
    for t, level_w in anchors:
        lines.append(f"{t:g},{level_w / UCAL_S:.6g}")
    return "\n".join(lines) + "\n"


def _series(name, inj, times, power_w, *, ligand_m, macro_m, temp_c=25.0,
            cell_ml=1.4, bad_index=None, baseline_anchors=None):
    if baseline_anchors is None:
        baseline_anchors = ((0.0, 0.0), (float(times[-1]), 0.0))
    base = os.path.join(ROOT, name)
    _write(os.path.join(base, "power.csv"), power_csv(times, power_w, bad_index=bad_index))
    _write(os.path.join(base, "injections.csv"), schedule_csv(inj))
    _write(os.path.join(base, "concentrations.csv"),
           concentrations_csv(ligand_m, macro_m))
    _write(os.path.join(base, "conditions.csv"), conditions_csv(temp_c, cell_ml))
    _write(os.path.join(base, "baseline.csv"), baseline_csv(baseline_anchors))


def build_main():
    n, spacing, V = 20, 300.0, 1.4e-3
    inj = make_schedule(n, spacing, v_l=5e-6)
    itimes = [t for _, t, _ in inj]
    M0, L0, k, dH = 50e-6, 1.0e-3, 1.0e7, -25_000.0
    heats, *_ = binding_heats_j(n, 5e-6, V, M0, L0, k, dH)
    dilution = np.full(n, 3.0 * 4.184e-6)  # 3 microcal/injection
    times = np.arange(0.0, itimes[-1] + 320.0, DT)
    pw = power_trace(times, itimes, heats, tau_s=55.0, noise_w=0.02 * UCAL_S, seed=11)
    pb = power_trace(times, itimes, dilution, tau_s=18.0, noise_w=0.01 * UCAL_S, seed=12)
    _series("main", inj, times, pw, ligand_m=L0, macro_m=M0)
    _series("blank_dilution", inj, times, pb, ligand_m=L0, macro_m=M0)


def build_overlap():
    n, spacing, V = 14, 150.0, 1.4e-3
    inj = make_schedule(n, spacing, start_s=100.0, v_l=5e-6)
    itimes = [t for _, t, _ in inj]
    M0, L0, k, dH = 40e-6, 0.8e-3, 2.0e7, -30_000.0
    heats, *_ = binding_heats_j(n, 5e-6, V, M0, L0, k, dH, n_sites=1.0)
    times = np.arange(0.0, itimes[-1] + 320.0, DT)
    pw = power_trace(times, itimes, heats, tau_s=70.0, noise_w=0.02 * UCAL_S, seed=21)
    _series("overlap", inj, times, pw, ligand_m=L0, macro_m=M0,
            baseline_anchors=((0.0, 0.0), (itimes[-1], 0.0)))


def build_gap():
    n, spacing, V = 18, 300.0, 1.4e-3
    inj = make_schedule(n, spacing, v_l=5e-6)
    itimes = [t for _, t, _ in inj]
    M0, L0, k, dH = 60e-6, 1.2e-3, 8.0e6, -20_000.0
    heats, *_ = binding_heats_j(n, 5e-6, V, M0, L0, k, dH)
    end = itimes[-1] + 320.0
    # remove a 150 s span straddling injection ordinal 9
    cut0, cut1 = itimes[8] - 60.0, itimes[8] + 90.0
    times = np.asarray([t for t in np.arange(0.0, end, DT) if not (cut0 <= t <= cut1)])
    pw = power_trace(times, itimes, heats, tau_s=55.0, noise_w=0.02 * UCAL_S, seed=31)
    # one unconvertible power token kept at origin (during a quiet region)
    bad = set(np.where(np.isclose(times, 250.0))[0].tolist())
    _series("gap", inj, times, pw, ligand_m=L0, macro_m=M0, bad_index=bad)


def build_tie():
    # weak binding + low c-value gives nearly flat per-injection heats:
    # one-site, two-site and dilution are within ~1 AICc.
    n, spacing, V = 16, 300.0, 1.4e-3
    inj = make_schedule(n, spacing, v_l=5e-6)
    itimes = [t for _, t, _ in inj]
    M0, L0 = 10e-6, 0.18e-3
    k, dH = 3.0e4, -6_000.0
    heats, *_ = binding_heats_j(n, 5e-6, V, M0, L0, k, dH)
    heats = heats + np.full(n, np.mean(heats) * 0.0 + 4.0e-6 * 55.0)
    times = np.arange(0.0, itimes[-1] + 320.0, DT)
    pw = power_trace(times, itimes, heats, tau_s=55.0, noise_w=0.03 * UCAL_S, seed=41)
    _series("tie", inj, times, pw, ligand_m=L0, macro_m=M0)


def main():
    build_main()
    build_overlap()
    build_gap()
    build_tie()
    print("case pack built at", ROOT)


if __name__ == "__main__":
    main()
