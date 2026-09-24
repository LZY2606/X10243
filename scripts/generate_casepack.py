"""Generate the offline ITC case pack (deterministic, seeded).

Scenario: ligand-into-protein titration at 25 C, 1.4 mL cell, 25 injections
(first 0.4 uL, then 24 x 2.5 uL every 100 s), 1 Hz power sampling with a
data gap at [1150, 1180) s, and a blank run that stops after 20 injections
(missing correspondence for the last 5 sample injections).
"""
import csv
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from itc.binding import heats_one_site  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "casepack")
rng = np.random.default_rng(7)

# --- ground truth -------------------------------------------------------
KA, DH_KJ, N = 2.0e6, -40.0, 1.0
M0, XS = 50e-6, 3.0e-3            # M
V0 = 1.4e-3                       # L
TAU = 12.0                        # s peak decay
DIL_KJ = 0.5                      # kJ/mol injectant dilution heat
T_END, DT = 2700.0, 1.0
GAP = (1150.0, 1180.0)

times_full = np.arange(0.0, T_END + DT, DT)
inj_times = [200.0 + 100.0 * i for i in range(25)]
inj_vols = [0.4] + [2.5] * 24     # uL
dv_L = [v * 1e-6 for v in inj_vols]

q_bind = heats_one_site(KA, DH_KJ * 1000.0, N, M0, XS, V0, dv_L)  # J
q_blank = np.array([DIL_KJ * 1000.0 * dv * XS for dv in dv_L])    # J
q_sample = q_bind + q_blank
q_sample[0] *= 0.5  # anomalous first injection (typical ITC artifact)


def synth(q_uJ_total, n_inj, drift=2e-5):
    p = 0.12 + drift * times_full
    for i in range(n_inj):
        t0 = inj_times[i]
        amp_uJ_s = q_uJ_total[i] * 1e6 / TAU  # J -> uJ/s
        mask = times_full >= t0
        p[mask] += amp_uJ_s * np.exp(-(times_full[mask] - t0) / TAU) / 4.184
    p += rng.normal(0.0, 0.03, size=len(p))
    return p


sample_p = synth(q_sample, 25)
blank_p = synth(q_blank, 20)      # blank stops after 20 injections

# data gap in the SAMPLE power curve only
keep = ~((times_full >= GAP[0]) & (times_full < GAP[1]))
sample_t, sample_p = times_full[keep], sample_p[keep]

os.makedirs(OUT, exist_ok=True)


def write_power(name, t, p):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "power_ucal_s"])
        for ti, pi in zip(t, p):
            w.writerow([f"{ti:.1f}", f"{pi:.6f}"])


write_power("power_curve.csv", sample_t, sample_p)
write_power("blank_power.csv", times_full, blank_p)

with open(os.path.join(OUT, "injections.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "time_s", "volume_uL", "duration_s", "first"])
    for i, (t0, v) in enumerate(zip(inj_times, inj_vols)):
        w.writerow([i, f"{t0:.1f}", v, 2.0, 1 if i == 0 else 0])

# branch schedule: injections 18/19 moved closer -> overlapping peaks
with open(os.path.join(OUT, "injections_v2_overlap.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "time_s", "volume_uL", "duration_s", "first"])
    moved = {18: 1925.0, 19: 1950.0}
    for i, (t0, v) in enumerate(zip(inj_times, inj_vols)):
        w.writerow([i, f"{moved.get(i, t0):.1f}", v, 2.0, 1 if i == 0 else 0])

json.dump({
    "label": "ligand-into-protein, lot A",
    "syringe": {"value": "3.00", "unit": "mM"},
    "cell": {"value": "0.050", "unit": "mM"},
    "syringe_lot": {"value": "n/a", "unit": "unknown"},
}, open(os.path.join(OUT, "concentrations_v1.json"), "w"), indent=2)
json.dump({
    "label": "ligand-into-protein, lot A (re-assayed)",
    "syringe": {"value": "2.90", "unit": "mM"},
    "cell": {"value": "0.050", "unit": "mM"},
    "syringe_lot": {"value": "n/a", "unit": "unknown"},
}, open(os.path.join(OUT, "concentrations_v2.json"), "w"), indent=2)
json.dump({"value": "25.0", "unit": "C"},
          open(os.path.join(OUT, "temperature.json"), "w"), indent=2)
json.dump({"value": "1.4", "unit": "mL"},
          open(os.path.join(OUT, "cell_volume.json"), "w"), indent=2)
json.dump({"mode": "constant", "value": {"value": "0.12", "unit": "ucal/s"}},
          open(os.path.join(OUT, "baseline.json"), "w"), indent=2)

json.dump({
    "title": "Titration thermogram: ITC injection integration, dilution heat "
             "and binding-model review",
    "coordinate_basis": {"time": "s since run start", "power": "ucal/s",
                         "temperature": "C", "volumes": "uL",
                         "concentrations": "mM", "cell_volume": "mL"},
    "truth": {"Ka": KA, "dH_kJ_mol": DH_KJ, "n": N,
              "syringe_M": XS, "cell_M": M0, "cell_volume_L": V0,
              "dilution_kJ_mol": DIL_KJ},
    "gap_s": list(GAP),
    "expected_diagnostics": [
        "window_crosses_gap", "blank_missing", "unit_unknown",
        "multiple_equally_good_models", "overlap",
    ],
    "narrative": [
        "import all materials v1 (each kind versioned separately)",
        "integrate sample + blank; gap truncates injection 9 window",
        "subtract dilution heat with explicit by_index rule; blank missing "
        "for injections 20-24",
        "fit one-site / two-site / dilution candidates",
        "decision: discard first injection from fitting",
        "concentration revision v2 -> models stale, integrals unchanged",
        "conflicting decisions on main vs review branch",
        "branch 'overlap': schedule v2 moves injections 18-19 -> overlap "
        "flags, impact limited to injections 17-19",
    ],
}, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print("casepack written to", os.path.abspath(OUT))
