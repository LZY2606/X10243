"""Offline case package: "ITC injection integration, dilution heat and
binding-model review" for a biophysics lab.

The package is a complete narrative, not a bag of unrelated buttons:

1. initial import versions power sampling, injection volumes/times,
   concentrations, temperature, cell volume, baseline and the blank run
   separately;
2. one parameter revision (integration window rule v1 -> v2);
3. one source-data change with local dependency impact (power patch around
   injection 7 only);
4. one pair of conflicting human decisions (accept then dispute one-site),
   both retained.

Replay walks through overlapping injection peaks, first-injection discard,
missing blank counterpart, concentration revision, multiple equi-optimal
models, a data gap and integration endpoints. A second branch
(``branch_overlap_gap``) stresses the overlap/gap rules in isolation.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from itc import engine
from itc.materials import (
    build_blank,
    cut_power_gap,
    parse_baseline,
    parse_cell_volume,
    parse_concentration,
    parse_injections,
    parse_power,
    parse_temperature,
    patch_power_region,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR = os.path.join(HERE, "materials")

# --- ground truth used to synthesise the run ---------------------------------
N_INJ = 16
DT = 0.5
T_FIRST = 120.0
SPACING = 180.0
TAU = 18.0
VOLUME_UL = 3.0
CELL_VOLUME_L = 1.4e-3
LIGAND_M = 2.0e-3
MACRO_M = 2.0e-4
# two-site truth with a weak second site so one-site and two-site end up
# statistically indistinguishable (delta AICc <= 2) on the integrated heats
TRUE = dict(n1=1.0, K1=2.0e5, dH1=-4.5e4, n2=1.0, K2=6.0e3, dH2=-9.0e3)
BASELINE_UW = 0.40
BLANK_HEAT_J = 0.8e-6
NOISE_UW = 0.004


def _heats_true() -> np.ndarray:
    from itc.models import cumulative_composition, heats_two_site

    volumes = np.full(N_INJ, VOLUME_UL * 1e-6)
    xt, mt = cumulative_composition(volumes, CELL_VOLUME_L, LIGAND_M, MACRO_M)
    return heats_two_site(
        xt, mt, CELL_VOLUME_L, TRUE["n1"], TRUE["K1"], TRUE["dH1"], TRUE["n2"], TRUE["K2"], TRUE["dH2"]
    )


def _power_csv(heats_J: np.ndarray, *, blank: bool = False, skip_peak: int | None = None) -> str:
    rng = np.random.default_rng(20260919)
    t_end = T_FIRST + (N_INJ - 1) * SPACING + 240.0
    times = np.arange(0.0, t_end, DT)
    power = np.full_like(times, BASELINE_UW)
    for i, heat in enumerate(heats_J):
        if skip_peak is not None and (i + 1) == skip_peak:
            continue
        t0 = T_FIRST + i * SPACING
        mask = times >= t0
        power[mask] += (heat * 1e6 / TAU) * np.exp(-(times[mask] - t0) / TAU)
    power += rng.normal(0.0, NOISE_UW, size=times.size)
    lines = ["time (s),power (µW)"]
    lines += [f"{t:.1f},{p:.6f}" for t, p in zip(times, power)]
    return "\n".join(lines) + "\n"


def _injections_csv(*, missing_volume_row: int | None = None, spacing: float = SPACING) -> str:
    lines = ["inj,time (s),volume (µL)"]
    for i in range(N_INJ):
        t0 = T_FIRST + i * spacing
        vol: str = f"{VOLUME_UL:.2f}"
        if missing_volume_row is not None and (i + 1) == missing_volume_row:
            vol = "n/a"
        lines.append(f"{i + 1},{t0:.1f},{vol}")
    return "\n".join(lines) + "\n"


def _concentration_csv(ligand_mM: float) -> str:
    lines = ["inj,syringe (mM),cell (µM)"]
    for i in range(N_INJ):
        lines.append(f"{i + 1},{ligand_mM:.3f},{MACRO_M * 1e6:.1f}")
    return "\n".join(lines) + "\n"


def generate_materials() -> dict[str, str]:
    """Deterministic raw CSV texts (the offline package on disk mirrors these)."""
    sample_heats = _heats_true()
    blank_heats = np.full(N_INJ, BLANK_HEAT_J)
    return {
        "run1_power.csv": _power_csv(sample_heats),
        "run1_injections.csv": _injections_csv(),
        "run1_concentration_v1.csv": _concentration_csv(2.0),
        "run1_concentration_v2.csv": _concentration_csv(2.2),
        "run1_temperature.csv": "temperature (°C)\n25.0\n",
        "run1_cell_volume.csv": "volume (µL)\n1400.0\n",
        "run1_baseline.csv": "baseline (µW)\n0.40\n",
        "blank_power.csv": _power_csv(blank_heats, blank=True, skip_peak=5),
        "blank_injections.csv": _injections_csv(missing_volume_row=5),
    }


def write_materials_dir() -> None:
    os.makedirs(MATERIALS_DIR, exist_ok=True)
    for name, text in generate_materials().items():
        with open(os.path.join(MATERIALS_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(text)


def _load(name: str) -> str:
    path = os.path.join(MATERIALS_DIR, name)
    if not os.path.exists(path):
        write_materials_dir()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def build_case_project() -> dict[str, Any]:
    """Replay the full case narrative and return the in-memory project."""
    project = engine.new_project("itc-titration-review")

    def commit(branch: str, etype: str, payload: dict[str, Any], note: str = "") -> dict[str, Any]:
        event = engine.make_event(project, branch, etype, payload, note=note)
        return engine.commit_event(project, event)

    # 1. initial import: every material versioned separately
    power_v1 = parse_power("main", _load("run1_power.csv"), "run1_power.csv")
    inj_v1 = parse_injections("main", _load("run1_injections.csv"), "run1_injections.csv")
    conc_v1 = parse_concentration("main", _load("run1_concentration_v1.csv"), "run1_concentration_v1.csv")
    temp = parse_temperature("main", _load("run1_temperature.csv"), "run1_temperature.csv")
    cellvol = parse_cell_volume("main", _load("run1_cell_volume.csv"), "run1_cell_volume.csv")
    baseline = parse_baseline("main", _load("run1_baseline.csv"), "run1_baseline.csv")
    blank_power = parse_power("blank_power", _load("blank_power.csv"), "blank_power.csv")
    blank_inj = parse_injections("blank_injections", _load("blank_injections.csv"), "blank_injections.csv")
    blank = build_blank("main", blank_power, blank_inj)

    for material, head in (
        (power_v1, True),
        (inj_v1, True),
        (conc_v1, True),
        (temp, True),
        (cellvol, True),
        (baseline, True),
        (blank_power, False),
        (blank_inj, False),
        (blank, True),
    ):
        commit("main", "IMPORT_MATERIAL", {"material": material, "role": material["role"], "head": head},
               note=f"import {material['source']['filename']}")

    commit("main", "SET_BLANK_RULE", {"correspondence": "ordinal", "scale": 1.0},
           note="dilution heat: blank injection i subtracted from sample injection i")
    commit("main", "SET_WINDOW_RULE", {"pre_s": 5.0, "post_s": 120.0}, note="window rule v1")
    # 2. parameter revision: window rule v2
    commit("main", "SET_WINDOW_RULE", {"pre_s": 8.0, "post_s": 100.0}, note="window rule v2 (parameter revision)")
    commit("main", "DROP_FIRST", {"discard": True}, note="first injection discarded (pre-equilibration artefact)")

    # fork the boundary-combination branch before later main-line changes
    commit("main", "FORK_BRANCH", {"name": "branch_overlap_gap"}, note="branch for overlap/gap rules")

    fit_v3 = project["branches"]["main"]["state"]["current_fit_id"]
    commit("main", "MODEL_DECISION",
           {"fit_id": fit_v3, "candidate": "one_site", "verdict": "accept"},
           note="reviewer A: one-site isotherm accepted")

    # concentration revision: fits become stale, stored integrals untouched
    conc_v2 = parse_concentration("main", _load("run1_concentration_v2.csv"), "run1_concentration_v2.csv")
    commit("main", "IMPORT_MATERIAL", {"material": conc_v2, "role": "main", "head": True},
           note="concentration revised 2.0 -> 2.2 mM (models expire, integrals unchanged)")

    # conflicting human decision, both kept
    commit("main", "MODEL_DECISION",
           {"fit_id": fit_v3, "candidate": "one_site", "verdict": "reject",
            "conflicts_with": "earlier accept"},
           note="reviewer B disputes the one-site acceptance; both decisions retained")

    # 3. local source-data change: patch power only around injection 7
    power_v2 = patch_power_region(power_v1, 1185.0, 1265.0, 1.15)
    commit("main", "IMPORT_MATERIAL", {"material": power_v2, "role": "main", "head": True},
           note="power patched locally around injection 7 only")

    # data gap introduced inside injection 10's window
    power_v3 = cut_power_gap(power_v2, 1755.0, 1785.0)
    commit("main", "IMPORT_MATERIAL", {"material": power_v3, "role": "main", "head": True},
           note="recorder dropout 1755-1785 s: integration window must not cross the gap")

    # --- branch_overlap_gap: dense schedule => overlapping post-peak windows ---
    dense = parse_injections("main", _injections_csv(spacing=50.0), "run1_injections_dense.csv")
    commit("branch_overlap_gap", "IMPORT_MATERIAL",
           {"material": dense, "role": "main", "head": True},
           note="dense 50 s spacing: post-injection windows overlap")
    commit("branch_overlap_gap", "SET_WINDOW_RULE", {"pre_s": 5.0, "post_s": 60.0},
           note="windows overlap neighbouring injections on purpose")
    gapped = cut_power_gap(power_v1, 420.0, 450.0)
    commit("branch_overlap_gap", "IMPORT_MATERIAL", {"material": gapped, "role": "main", "head": True},
           note="gap 420-450 s clips injection 7 window on this branch")
    return project


if __name__ == "__main__":
    write_materials_dir()
    proj = build_case_project()
    state = proj["branches"]["main"]["state"]
    print("project:", proj["id"], "events:", len(proj["events"]))
    print("integration:", state["current_integration_id"])
    print("fit:", state["current_fit_id"])
