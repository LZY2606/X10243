"""Offline case pack: "滴定热谱 — ITC 注射积分、稀释热与绑定模型审议".

Deterministically synthesises a realistic ITC run and replays the full
narrative: initial import (each material separately versioned), integration,
dilution-heat subtraction with a stated correspondence rule, model review
with equipotent candidates, a concentration revision, a localised source
change, and two conflicting human decisions on separate branches.

Boundary situations exercised by the replay:
  overlapping injection peaks, first-injection discard, missing blank
  correspondence, concentration revision, multiple equipotent models,
  data gaps and integration endpoints, and an unconvertible unit that
  stays raw with a diagnostic.
"""
from __future__ import annotations

import numpy as np

from .binding import bound_conc
from .domain import MATERIAL_KINDS

TAU_S = 12.0
TRUE_K = 10.0        # 1/mM
TRUE_DH = -30.0      # kJ/mol
CELL_MM = 0.1
SYRINGE_MM = 1.0
CELL_ML = 1.4
GAP = (360.0, 380.0)
INJ_TIMES = [60.0, 150.0, 240.0, 330.0, 420.0, 495.0, 530.0, 620.0, 710.0]
INJ_VOLS = [2.0] + [5.0] * 8
BLANK_HEAT_UJ = 2.0
N_BLANK_INJ = 7      # blank covers fewer injections -> missing correspondence
T_END = 900.0


def true_heats(vols=None, cs=SYRINGE_MM):
    vols = vols or INJ_VOLS
    V0 = CELL_ML * 1000.0
    cum = np.cumsum(vols)
    X = cs * cum / V0
    B = bound_conc(TRUE_K, CELL_MM, X)
    dB = np.diff(np.concatenate([[0.0], B]))
    return (TRUE_DH * dB * V0).tolist()


def _trace_samples(inj_times, heats, gap=None):
    t = np.arange(0.0, T_END, 1.0)
    p = 0.5 + 0.0005 * t
    for ti, h in zip(inj_times, heats):
        m = t >= ti
        p[m] += (h / TAU_S) * np.exp(-(t[m] - ti) / TAU_S)
    p = p + 0.02 * np.sin(0.7 * t) + 0.01 * np.sin(2.3 * t)
    keep = np.ones_like(t, dtype=bool)
    if gap:
        keep = (t < gap[0]) | (t >= gap[1])
    return [{"t": float(ti), "p": float(pi)} for ti, pi in zip(t[keep], p[keep])]


def _baseline_samples():
    t = np.arange(0.0, T_END, 5.0)
    return [{"t": float(ti), "p": float(0.5 + 0.0005 * ti)} for ti in t]


def _schedule_payload(times, vols, odd_unit_index=None):
    injs = []
    for i, (ti, vi) in enumerate(zip(times, vols)):
        unit = "arb" if odd_unit_index is not None and i == odd_unit_index else "uL"
        injs.append({"id": f"I{i+1:02d}",
                     "time": {"value": ti, "unit": "s"},
                     "volume": {"value": vi, "unit": unit}})
    return {"injections": injs}


def build_materials():
    """The seven separately versioned immutable materials, raw with units."""
    heats = true_heats()
    power = {
        "time_unit": "s", "power_unit": "uW",
        "time_base": "elapsed_seconds_from_run_start",
        "samples": _trace_samples(INJ_TIMES, heats, gap=GAP),
    }
    schedule = _schedule_payload(INJ_TIMES, INJ_VOLS, odd_unit_index=4)  # I05 'arb'
    concentrations = {"cell": {"value": CELL_MM, "unit": "mM"},
                      "syringe": {"value": SYRINGE_MM, "unit": "mM"}}
    temperature = {"value": 25.0, "unit": "C"}
    cell_volume = {"value": CELL_ML, "unit": "mL"}
    baseline = {"time_unit": "s", "power_unit": "uW",
                "time_base": "elapsed_seconds_from_run_start",
                "samples": _baseline_samples()}
    blank = {
        "trace": {"time_unit": "s", "power_unit": "uW",
                  "time_base": "elapsed_seconds_from_run_start",
                  "samples": _trace_samples(INJ_TIMES[:N_BLANK_INJ],
                                            [BLANK_HEAT_UJ] * N_BLANK_INJ)},
        "schedule": _schedule_payload(INJ_TIMES[:N_BLANK_INJ], INJ_VOLS[:N_BLANK_INJ]),
    }
    return {"power_trace": power, "injection_schedule": schedule,
            "concentrations": concentrations, "temperature": temperature,
            "cell_volume": cell_volume, "baseline": baseline, "blank_trace": blank}


def revised_concentrations():
    """Parameter revision: syringe re-weighed, 1.0 -> 1.15 mM."""
    return {"cell": {"value": CELL_MM, "unit": "mM"},
            "syringe": {"value": 1.15, "unit": "mM"}}


def revised_schedule():
    """Localised source change: I03 volume corrected 5.0 -> 4.8 uL."""
    vols = list(INJ_VOLS)
    vols[2] = 4.8
    return _schedule_payload(INJ_TIMES, vols, odd_unit_index=4)


def replay(session):
    """Run the full narrative against a Session; returns landmark ids."""
    landmarks = {}
    mats = build_materials()
    for kind in MATERIAL_KINDS:
        session.import_material(kind, mats[kind], note="initial import (case pack v1)")
    landmarks["integration_v1"] = session.run_integration()
    landmarks["dilution_v1"] = session.run_dilution(rule="by_index")
    landmarks["fit_v1"] = session.run_fit()
    session.decide("accept", landmarks["fit_v1"] + "#one_site",
                   "reviewer A: one-site model sufficient for this isotherm",
                   branch="main", reviewer="A")
    # parameter revision: stales fits, never the original integrals
    session.import_material("concentrations", revised_concentrations(),
                            note="syringe concentration revised after re-weighing")
    landmarks["fit_v2"] = session.run_fit()
    # conflicting human decision on a second branch
    session.decide("reject", landmarks["fit_v2"] + "#one_site",
                   "reviewer B: residuals structured, prefers two-site",
                   branch="alt", reviewer="B")
    # localised source data change: only downstream of injection_schedule
    session.import_material("injection_schedule", revised_schedule(),
                            note="I03 volume corrected from calibrant log")
    landmarks["integration_v2"] = session.run_integration()
    landmarks["dilution_v2"] = session.run_dilution(rule="by_index")
    landmarks["fit_v3"] = session.run_fit()
    return landmarks
