"""Author the dedicated offline ITC case pack:
"Titration thermograms: ITC injection integration, dilution heat and binding
model review".

The pack is domain material, not a generic spreadsheet: versioned files for
power sampling, injection volume/time, concentration, temperature, cell volume,
baseline and the separately versioned dilution blank, plus a replay script
describing the parameter revision, local-only source change and conflicting
human decisions.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .synthetic import build_trace, peak_shape, wiseman_heats


def _samples_payload(trace: dict, *, time_ref: str) -> dict:
    return {
        "time_reference": time_ref,
        "samples": [
            {"t": {"value": str(t), "unit": "s"},
             "p": {"value": f"{p:.9e}", "unit": "cal/s"}}
            for t, p in zip(trace["times"], trace["powers"])],
        "gaps": ([[ {"value": str(g), "unit": "s"} for g in trace["gap"]]]
                 if trace.get("gap") else []),
    }


def injections_payload(times_s: list[float], volumes_ul: list[float],
                       *, discard_ordinal: int | None = 1,
                       time_ref: str = "t=0 at acquisition start") -> dict:
    rows = []
    for i, (t, v) in enumerate(zip(times_s, volumes_ul), start=1):
        row = {"ordinal": i,
               "time": {"value": str(int(t)) if float(t).is_integer() else str(t),
                        "unit": "s"},
               "volume": {"value": str(v), "unit": "µL"}}
        if discard_ordinal is not None and i == discard_ordinal:
            row["discard_first"] = True
        rows.append(row)
    return {"time_reference": time_ref, "injections": rows}


def concentration_payload(syringe_mm: float, cell_um: float, *,
                          syringe_raw: str | None = None,
                          cell_raw: float | None = None) -> dict:
    return {
        "syringe": {"species": "ligand L",
                    "concentration": {"value": syringe_raw if syringe_raw is not None
                                      else syringe_mm, "unit": "mM"}},
        "cell": {"species": "protein M",
                 "concentration": {"value": cell_um if cell_raw is None
                                   else cell_raw, "unit": "µM"}},
    }


def temperature_payload(celsius: float) -> dict:
    return {"temperature": {"value": celsius, "unit": "°C"},
            "note": "jacket temperature"}


def cell_volume_payload(ml: float) -> dict:
    return {"cell_volume": {"value": ml, "unit": "mL"},
            "note": "active cell volume"}


def baseline_payload(level_cal_s: float, *, slope_cal_s2: float = 0.0) -> dict:
    return {"baseline": {"method": "linear" if slope_cal_s2 else "constant",
                         "level": {"value": f"{level_cal_s:.3e}", "unit": "cal/s"},
                         "slope": ({"value": f"{slope_cal_s2:.3e}",
                                    "unit": "cal/s/s"} if slope_cal_s2 else None),
                         "time_reference": "t=0 at acquisition start"}}


def blank_link_payload(blank_run: str, *, rule: str = "ordinal",
                       discard_first: bool = False, note: str = "") -> dict:
    return {"blank_run": blank_run,
            "correspondence": {"rule": rule,
                               "discard_first_in_blank": discard_first},
            "note": note}


# --------------------------------------------------------------------------- #
# Scenario numbers
# --------------------------------------------------------------------------- #

# Physical setup (canonical SI in comments)
V0_ML = 0.2034                    # cell volume 2.034e-4 L
TEMP_C = 25.0
CELL_UM = 20.0                    # 2.0e-5 M protein
SYRINGE_MM = 0.40                 # 4.0e-4 M ligand (v1, later revised to 0.32)

# 8 shots. Timings deliberately create:
#  1:30 / 2:120 / 3:210 / 4:270 / 5:330 / 6:390 / 7:480 / 8:570
# shots 3 and 4 are 60 s apart with slow decay -> overlapping peaks
INJ_TIMES_S = [30, 120, 210, 270, 330, 390, 480, 570]
INJ_VOLUMES_UL = [2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0]

# a data gap sits inside shot 5's natural window (330 -> 390)
GAP = (355.0, 372.0)

# one-site binding truth used to generate sample peak areas
N_SITES = 1.0
K_ASSOC = 8.0e4
DH_CAL_MOL = -6000.0              # exothermic 6 kcal/mol

# blank: near constant dilution heat per shot; first blank shot is later
# removed by the correspondence rule, and blank shot 6 is deliberately absent
BLANK_DILUTION_CAL = 3.0e-5       # 30 µcal per matched shot


def sample_areas_v1() -> list[float]:
    v0_l = V0_ML * 1e-3
    c_m = CELL_UM * 1e-6
    syringe_m = SYRINGE_MM * 1e-3
    ratios = []
    cell_ligand = 0.0
    for v_ul in INJ_VOLUMES_UL:
        v = v_ul * 1e-6
        cell_ligand = cell_ligand * (v0_l - v) / v0_l + syringe_m * v
        ratios.append(cell_ligand / v0_l / c_m)
    heats = wiseman_heats(ratios, N_SITES, K_ASSOC, c_m, DH_CAL_MOL, v0_l)
    # dilution adds ~30 µcal per shot; peak area = -(heat) to be exothermic up
    areas = [-h + BLANK_DILUTION_CAL for h in heats]
    return areas


def sample_areas_revised_syringe() -> list[float]:
    """Same raw integration, new molar ratios: only models go stale.

    Areas themselves are NOT regenerated from the revised concentration -
    exactly the required semantics: concentration revision invalidates models
    but never rewrites the original per-injection integrals.
    """
    return sample_areas_v1()


def blank_areas() -> list[float]:
    # 7 shots in the blank: counterpart for sample ordinals 1..5 and 7..8 is
    # missing ordinal 6 once discard_first shifts the correspondence
    return [BLANK_DILUTION_CAL] * 7


BLANK_INJ_TIMES_S = [20, 100, 180, 240, 300, 420, 510]
BLANK_VOLUMES_UL = [2.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0]


def make_sample_trace_v1() -> dict:
    areas = sample_areas_v1()
    decays = [12.0, 12.0, 34.0, 30.0, 18.0, 18.0, 14.0, 14.0]
    return build_trace(INJ_TIMES_S, areas, dt_s=1.0,
                       t_end_s=700.0, baseline_cal_s=1.0e-6,
                       rise_s=2.0, decays_s=decays, gap=GAP,
                       seed_offset=0.13)


def make_blank_trace() -> dict:
    return build_trace(BLANK_INJ_TIMES_S, blank_areas(), dt_s=1.0,
                       t_end_s=640.0, baseline_cal_s=0.8e-6,
                       rise_s=2.0, decays_s=10.0, seed_offset=0.71)


def make_flat_trace() -> dict:
    """Flat, dilution-dominated titration for the equivalent-models branch."""
    flat_areas = [2.5e-5] * 8
    return build_trace(INJ_TIMES_S, flat_areas, dt_s=1.0,
                       t_end_s=700.0, baseline_cal_s=1.0e-6,
                       rise_s=2.0, decays_s=8.0, seed_offset=1.91)


def patched_power_payload() -> dict:
    """Local-only source change: re-sample only shot 8's tail (490-560s).

    Only integrations that read that fragment can change - everything earlier
    keeps identical derived identities.
    """
    trace = make_sample_trace_v1()
    fragment_times = [t for t in trace["times"] if 490.0 <= t <= 560.0]
    # add a small extra exothermic bump in the local fragment
    rows = []
    for t in fragment_times:
        extra = peak_shape(t, 500.0, 1.0e-5, rise_s=2.0, decay_s=9.0)
        p = 1.0e-6 + extra
        rows.append({"t": {"value": str(int(t)), "unit": "s"},
                     "p": {"value": f"{p:.9e}", "unit": "cal/s"},
                     "flag": "local_patch"})
    return {"replaces_interval_s": [490.0, 560.0], "samples": rows,
            "time_reference": "t=0 at acquisition start"}
