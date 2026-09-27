"""Offline case pack: "滴定热谱：ITC 注射积分、稀释热与绑定模型审议".

Deterministic, domain-specific sample data (no generic tables). The
baseline versions every material separately: power sampling, injection
volumes/times, concentrations, temperature, cell volume, baseline and
blank curves. The scripted narrative then exercises every boundary
combination required by the review:

  - overlapping post-injection peaks   (injections 5/6, 70 s spacing)
  - first-injection discard            (2 uL prime injection)
  - missing blank counterpart          (blank stops at injection 14)
  - concentration revision             (syringe 2.0 -> 1.8 mM)
  - multiple equally-good models       (one-site ~ two-site, dAICc < 2)
  - data gap + integration endpoints   (gap clamps injection 12 window)
  - conflicting human decisions        (main vs review-b branches)

All domain quantities carry units or a coordinate basis. Raw text is
what gets stored; normalisation happens in the derived layer.
"""
from __future__ import annotations

import numpy as np

from .models import FitInput, two_site_heats

CASE_ID = "itc-casepack-001"
CASE_TITLE = "滴定热谱：ITC 注射积分、稀释热与绑定模型审议"

N_INJECTIONS = 20
FIRST_VOLUME_UL = 2.0
VOLUME_UL = 10.0
SPACING_S = 180.0
FIRST_TIME_S = 120.0
OVERLAP_PAIR = (5, 6)          # 0-based; spacing shortened to 70 s
OVERLAP_SPACING_S = 70.0
TAU_S = 25.0
SAMPLING_HZ = 1.0
NOISE_POWER_UJ_S = 0.05
SEED = 11

CELL_CONC_V1 = "0.2 mM"
SYRINGE_CONC_V1 = "2.0 mM"
SYRINGE_CONC_V2 = "1.8 mM"     # the concentration revision
TEMPERATURE = "298.15 K"
CELL_VOLUME = "1.4 mL"

# Truth used to synthesise the power trace (weak second site so that
# one-site and two-site fits end up equally good, dAICc < 2).
TRUTH = dict(logka1=2.0, n1=0.95, dh1=-42.0,
             logka2=0.5, n2=0.05, dh2=-25.0)
BLANK_HEAT_UJ = -1.5
BLANK_NOISE_UJ = 0.15
BLANK_PRESENT_UNTIL = 14       # blank entries exist for injections 0..14

GAP_START_S = 2310.0           # tail of injection 12 (t=2170 s)
GAP_END_S = 2370.0
GAP_REASON = "detector reset during run"

BASELINE_LOCAL_RANGE = (1500.0, 1700.0)   # baseline v2 changes only here
BASELINE_LOCAL_BUMP_UJ_S = 0.5


def injection_times():
    times = [FIRST_TIME_S + i * SPACING_S for i in range(N_INJECTIONS)]
    i, j = OVERLAP_PAIR
    shift = (SPACING_S - OVERLAP_SPACING_S)
    for k in range(j, N_INJECTIONS):
        times[k] -= shift
    return [float(t) for t in times]


def injection_volumes():
    return [FIRST_VOLUME_UL] + [VOLUME_UL] * (N_INJECTIONS - 1)


def truth_heats_uJ():
    inp = FitInput(volumes_uL=injection_volumes(),
                   heats_uJ=[0.0] * N_INJECTIONS,
                   cell_conc_mM=0.2, syringe_conc_mM=2.0,
                   cell_volume_uL=1400.0, temperature_K=298.15)
    return two_site_heats(
        (TRUTH["logka1"], TRUTH["n1"], TRUTH["dh1"],
         TRUTH["logka2"], TRUTH["n2"], TRUTH["dh2"]), inp)


def blank_heats_uJ():
    rng = np.random.default_rng(SEED + 1)
    return {i: float(BLANK_HEAT_UJ + rng.normal(0, BLANK_NOISE_UJ))
            for i in range(BLANK_PRESENT_UNTIL + 1)}


def build_power_trace():
    """Synthesise the raw power curve. Peak areas equal truth heat plus
    blank (dilution) heat, so blank subtraction recovers the truth."""
    rng = np.random.default_rng(SEED)
    times = injection_times()
    t_end = times[-1] + SPACING_S
    t = np.arange(0.0, t_end, 1.0 / SAMPLING_HZ)
    power = np.full_like(t, 0.3)          # steady baseline offset uJ/s
    power += 0.00002 * t                  # slight drift
    heats = truth_heats_uJ()
    blanks = blank_heats_uJ()
    for i, t0 in enumerate(times):
        t1 = times[i + 1] if i + 1 < len(times) else t_end
        area = float(heats[i]) + blanks.get(i, BLANK_HEAT_UJ)
        mask = (t >= t0) & (t < t1)
        decay = np.exp(-(t[mask] - t0) / TAU_S)
        norm = np.trapezoid(decay, t[mask])
        power[mask] += area * decay / norm
    power += rng.normal(0, NOISE_POWER_UJ_S, size=t.shape)
    # data gap: samples removed, not zeroed
    keep = ~((t >= GAP_START_S) & (t < GAP_END_S))
    return t[keep], power[keep]


def build_baseline(version: int):
    t, _ = build_power_trace()
    base = 0.3 + 0.00002 * t
    if version >= 2:
        lo, hi = BASELINE_LOCAL_RANGE
        base = base + BASELINE_LOCAL_BUMP_UJ_S * ((t >= lo) & (t < hi))
    return t, base


def build_case():
    """Return the full case-pack definition (JSON-able materials)."""
    t, power = build_power_trace()
    bt, bp = build_baseline(version=1)
    times = injection_times()
    volumes = injection_volumes()
    blanks = blank_heats_uJ()
    return {
        "case_id": CASE_ID,
        "title": CASE_TITLE,
        "seed": SEED,
        "materials": {
            "power-trace": {
                "kind": "power_trace",
                "units": {"power": "uJ/s", "time": "s",
                          "time_basis": "seconds since run start",
                          "sampling": f"{SAMPLING_HZ} Hz"},
                "content": {
                    "t_s": [round(float(x), 3) for x in t],
                    "power": [round(float(x), 6) for x in power],
                    "power_unit": "uJ/s",
                    "time_basis": "seconds since run start",
                    "gaps": [{"start_s": GAP_START_S, "end_s": GAP_END_S,
                              "reason": GAP_REASON}],
                },
            },
            "injections": {
                "kind": "injection_schedule",
                "units": {"time": "s", "volume": "uL"},
                "content": {
                    "entries": [
                        {"index": i,
                         "time": f"{times[i]:.1f} s",
                         "volume": f"{volumes[i]:.1f} uL"}
                        for i in range(N_INJECTIONS)],
                },
            },
            "concentrations": {
                "kind": "concentrations",
                "units": {"cell": "conc_mM", "syringe": "conc_mM"},
                "content": {"cell": CELL_CONC_V1, "syringe": SYRINGE_CONC_V1},
            },
            "temperature": {
                "kind": "temperature",
                "units": {"value": "temperature_K"},
                "content": {"value": TEMPERATURE},
            },
            "cell-volume": {
                "kind": "cell_volume",
                "units": {"value": "volume_uL"},
                "content": {"value": CELL_VOLUME},
            },
            "baseline": {
                "kind": "baseline_curve",
                "units": {"power": "uJ/s", "time": "s"},
                "content": {
                    "t_s": [round(float(x), 3) for x in bt],
                    "power": [round(float(x), 6) for x in bp],
                    "power_unit": "uJ/s",
                },
            },
            "blank": {
                "kind": "blank_curve",
                "units": {"heat": "uJ"},
                "content": {
                    "rule": "same_injection_index",
                    "entries": [
                        {"index": i, "heat": f"{blanks[i]:.4f} uJ"}
                        for i in sorted(blanks)],
                },
            },
        },
        "revisions": {
            "concentrations_v2": {"cell": CELL_CONC_V1,
                                  "syringe": SYRINGE_CONC_V2},
            "baseline_v2_note": (
                f"baseline re-fit: +{BASELINE_LOCAL_BUMP_UJ_S} uJ/s only in "
                f"[{BASELINE_LOCAL_RANGE[0]}, {BASELINE_LOCAL_RANGE[1]}) s"),
        },
        "decisions_script": [
            {"branch": "main", "decision": "accept",
             "target": "cand-one_site",
             "reviewer": "reviewer-A",
             "rationale": "parsimony: one-site is within dAICc 2 of two-site"},
            {"branch": "review-b", "decision": "accept",
             "target": "cand-two_site",
             "reviewer": "reviewer-B",
             "rationale": "weak second site is physically plausible"},
        ],
    }


if __name__ == "__main__":
    import json
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "casepack/casepack.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(build_case(), fh, ensure_ascii=False, indent=1)
    print(f"wrote {out}")
