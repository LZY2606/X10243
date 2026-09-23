"""Generate the offline ITC case pack (run once; outputs are committed).

Scenario: 20-injection ITC titration of a 1:1 binder, first injection
small (to be discarded), injections 12/13 close enough that post-injection
peaks overlap, a data gap inside injection 15's window, and a blank
titration that only covers injections 0..17 (missing correspondence for
the last two).
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# truth: one-site n=1, K=100 1/mM, dH=-10 uJ/nmol, qd=0.5 uJ
CELL_MM, SYRINGE_MM, CELL_UL = 0.1, 1.2, 1400.0
N_INJ = 20
VOLUMES = [2.0] + [10.0] * (N_INJ - 1)
TIMES = [300.0 + 150.0 * i for i in range(N_INJ)]
TIMES[13] = TIMES[12] + 40.0          # overlapping post-injection peaks
GAP = (TIMES[15] + 60.0, TIMES[15] + 95.0)  # data gap inside window 15
TAU = 25.0
BASELINE = 0.5                        # uJ/s
T_END = 3400.0


def bound(K, mt, xt):
    a = K * (mt + xt) + 1.0
    return (a - math.sqrt(a * a - 4 * K * K * mt * xt)) / (2 * K)


def true_heats():
    mt, xt, prev_b, heats = CELL_MM, 0.0, 0.0, []
    for v in VOLUMES:
        keep = 1.0 - v / CELL_UL
        mt *= keep
        xt = xt * keep + SYRINGE_MM * v / CELL_UL
        b = bound(100.0, mt, xt)
        heats.append(CELL_UL * -10.0 * (b - prev_b * keep) + 0.5)
        prev_b = b
    return heats


def main():
    rng = np.random.RandomState(42)
    heats = true_heats()
    t_all, p_all = [], []
    t = 0.0
    inj_iter = iter(sorted(zip(TIMES, heats)))
    inj_t, inj_h = next(inj_iter)
    peak_start, peak_amp = None, 0.0
    while t <= T_END:
        if peak_start is None and t >= inj_t:
            peak_start = inj_t
            peak_amp = -inj_h / TAU      # exothermic -> negative? keep sign
            peak_amp = inj_h / TAU
        p = BASELINE
        if peak_start is not None:
            p += peak_amp * math.exp(-(t - peak_start) / TAU)
            if t - peak_start > 8 * TAU:
                peak_start = None
                try:
                    inj_t, inj_h = next(inj_iter)
                except StopIteration:
                    inj_t = float("inf")
        p += rng.normal(0.0, 0.02)
        if not (GAP[0] <= t <= GAP[1]):
            t_all.append(round(t, 3))
            p_all.append(round(p, 6))
        t += 1.0

    with open(os.path.join(HERE, "power_curve.csv"), "w") as f:
        f.write("# instrument: VP-ITC clone; time basis: instrument_clock\n")
        f.write("time_s,power_uJ_per_s\n")
        for tt, pp in zip(t_all, p_all):
            f.write(f"{tt},{pp}\n")

    with open(os.path.join(HERE, "injections.csv"), "w") as f:
        f.write("index,time_s,volume_uL\n")
        for i, (tt, vv) in enumerate(zip(TIMES, VOLUMES)):
            f.write(f"{i},{tt},{vv}\n")

    blank = [round(0.5 + rng.normal(0, 0.05), 4) for _ in range(18)]
    with open(os.path.join(HERE, "blank_heats.csv"), "w") as f:
        f.write("index,heat_uJ\n")
        for i, h in enumerate(blank):
            f.write(f"{i},{h}\n")

    metadata = {
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "sample": "SH2 domain / phosphopeptide (synthetic)",
        "concentrations": {
            "cell": {"value": 0.1, "unit": "mM"},
            "syringe": {"value": 1.2, "unit": "mM"}},
        "temperature": {"value": 25.0, "unit": "degC"},
        "cell_volume": {"value": 1.4, "unit": "mL"},
        "baseline": {"kind": "constant", "coefficients": [0.5],
                     "unit": "uJ/s"},
        "gap": {"start_s": GAP[0], "end_s": GAP[1],
                "reason": "autosampler refill; samples dropped"},
        "truth": {"model": "one_site", "n": 1.0, "log10K_1_per_mM": 2.0,
                  "dH_uJ_per_nmol": -10.0, "qd_uJ": 0.5},
    }
    with open(os.path.join(HERE, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("casepack generated:", len(t_all), "samples")


if __name__ == "__main__":
    main()
