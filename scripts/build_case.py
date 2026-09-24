"""Build the offline domain case package under ``case/``.

The package is genuine ITC material (never a generic spreadsheet):

v1  -- 18-injection one-site titration:
       * injections 14/15 deliberately close (overlapping post-injection peaks)
       * a 70 s sampling gap inside injection 9 window (gap + endpoints)
       * blank table contains 17 rows (correspondence missing for #18)
       * first injection marked to be discarded in review
       * weak-binding tail injections (15-18) create a regime where
         one-site and dilution hypotheses become statistically tied

revisions/  -- parameter revision (concentration), a local source-data fix
              (injection volume), and a conflicting manual decision is made
              at runtime via the replay/branch API.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "case")

# --------------------------------------------------------------------------
# experiment parameters (units carried in headers/JSON)
# --------------------------------------------------------------------------
V0_L = 2.0266e-7              # 202.66 uL cell volume
C0_M = 5.0e-5                 # 50 uM macromolecule in cell
X0_M = 6.0e-4                 # 600 uM ligand in syringe (v1)
DV_MAIN_L = 5.0e-6            # 5 uL injections
N_INJ = 18
KA_STRONG = 5.0e5
KA_WEAK = 2.0e3
DH_STRONG = -3.0e4            # J/mol
TAU_S = 10.0
RISE_S = 2.0
PERIOD_S = 2.0
BASE_PERIOD_S = 2.0
INTERVAL_MAIN_S = 240.0
INTERVAL_SHORT_S = 90.0       # injections 14->15 overlap
START_S = 60.0
GAP_INJECTION = 9
GAP_S = 70.0


def wiseman_q(dv_l, v0_l, c0_m, x0_m, ka, dh, n=1.0):
    v_i = v0_l + np.cumsum(dv_l)
    c_i = c0 * v0 / v_i
    x_i = x0 * np.cumsum(dv_l) / v_i
    ka_c = np.maximum(ka * c_i * n, 1e-30)
    r = x_i / np.maximum(c_i * n, 1e-30)
    a = 1.0 + r + 1.0 / ka_c
    theta = 0.5 * (a - np.sqrt(np.maximum(a * a - 4.0 * r, 0.0)))
    qtot = c_i * n * dh * v_i * theta
    qprev = np.concatenate(([0.0], qtot[:-1]))
    q = qtot - (1.0 - 0.5 * dv_l / v_i) * qprev
    return q, v_i


def peak(times, center_s, area_j, tau_s=TAU_S, rise_s=RISE_S):
    # shape normalized so its trapezoid integral = area (J) -> watts
    x = times - center_s
    shape = np.zeros_like(x)
    m = (x > 0) & (x <= rise_s)
    shape[m] = x[m] / rise_s
    f = x > rise_s
    shape[f] = np.exp(-(x[f] - rise_s) / tau_s)
    integral = np.trapz(shape, times - center_s + 0.0) if False else None
    # analytical area of shape starting at 0
    area_shape = 0.5 * rise_s + tau_s
    return shape * (area_j / area_shape)


def injection_times():
    times = [START_S]
    for i in range(1, N_INJ):
        prev = times[-1]
        # injection at index 15 (1-based) arrives 90 s after #14
        interval = INTERVAL_SHORT_S if i == 13 else INTERVAL_MAIN_S
        times.append(prev + interval)
    return times


def build_power(x0=X0_M, ka=KA_STRONG, dh=DH_STRONG, skip_gap=False):
    dv = np.full(N_INJ, DV_MAIN_L)
    q, _ = wiseman_q(dv, V0_L, C0_M, x0, ka, dh)
    itimes = injection_times()
    t_end = itimes[-1] + 170.0
    times = np.arange(0.0, t_end + 0.1, PERIOD_S)
    power_w = np.zeros_like(times)
    # mild baseline drift, provided separately too
    power_w += 0.05e-6 + 0.02e-6 * (times / t_end)
    for i, (tc, qj) in enumerate(zip(itimes, q)):
        # first injection is systematically smaller (mixing artifact)
        factor = 0.45 if i == 0 else 1.0
        power_w += peak(times, tc + 3.0, qj * factor)
    if not skip_gap:
        tc9 = itimes[GAP_INJECTION - 1]
        gap_lo, gap_hi = tc9 + 40.0, tc9 + 40.0 + GAP_S
        keep = ~((times > gap_lo) & (times < gap_hi))
        times, power_w = times[keep], power_w[keep]
    return times, power_w, itimes, q


def write_power(path, times, power_w):
    with open(path, "w") as fh:
        fh.write("time_s,power_uW\n")
        for t, p in zip(times, power_w):
            fh.write(f"{t:.3f} s,{p * 1e6:.5f} uW\n")


def write_schedule(path, itimes, volume_override=None):
    with open(path, "w") as fh:
        fh.write("inject_index,time_s,volume_uL\n")
        for i, tc in enumerate(itimes, start=1):
            vol = (volume_override or {}).get(i, DV_MAIN_L * 1e6)
            fh.write(f"{i},{tc:.1f} s,{vol:.2f} uL\n")


def write_conditions(path, x0=X0_M, c0=C0_M, temp_k=298.15, v0=V0_L):
    data = {
        "cell_concentration": f"{c0 * 1e6:.3f} uM",
        "syringe_concentration": f"{x0 * 1e3:.4f} mM",
        "temperature": f"{temp_k - 273.15:.2f} C",
        "cell_volume": f"{v0 * 1e6:.2f} uL",
        "time_basis": "run_elapsed",
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_baseline(path, t_end):
    anchors = [(0.0, 0.05), (t_end, 0.07)]
    with open(path, "w") as fh:
        fh.write("time_s,baseline_uW\n")
        for t, v in anchors:
            fh.write(f"{t:.1f} s,{v:.4f} uW\n")


def write_blank(path, itimes, n_rows=17):
    # measured dilution heats: ~ constant 1.2 uJ per 5 uL injection
    rng = np.random.default_rng(7)
    with open(path, "w") as fh:
        fh.write("inject_index,time_s,volume_uL,energy_uJ\n")
        for i, tc in enumerate(itimes[:n_rows], start=1):
            e = 1.20 + 0.06 * math.sin(i * 1.7) + float(rng.normal(0, 0.01))
            fh.write(f"{i},{tc:.1f} s,5.00 uL,{e:.3f} uJ\n")


def main():
    os.makedirs(os.path.join(ROOT, "v1"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "revisions"), exist_ok=True)
    times, power_w, itimes, q = build_power()
    t_end = times[-1]
    v1 = os.path.join(ROOT, "v1")
    write_power(os.path.join(v1, "power.csv"), times, power_w)
    write_schedule(os.path.join(v1, "schedule.csv"), itimes)
    write_conditions(os.path.join(v1, "conditions.json"))
    write_baseline(os.path.join(v1, "baseline.csv"), t_end)
    write_blank(os.path.join(v1, "blank.csv"), itimes)

    # revision 1: parameter revision -- syringe 0.600 mM -> 0.500 mM
    write_conditions(os.path.join(ROOT, "revisions", "conditions_r1.json"),
                     x0=5.0e-4)
    # revision 2: local source fix -- injection 12 volume typo 4.20 -> 5.00 uL
    write_schedule(os.path.join(ROOT, "revisions", "schedule_r2.csv"),
                   itimes, volume_override={12: 4.20})
    manifest = {
        "case": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "narrative": [
            "导入原始功率曲线、注射时序、条件、基线和空白对照(独立版本化)。",
            "核对每次注射积分；触发首注舍弃、重叠峰、数据缺口、空白对应缺失。",
            "浓度修订使归一化热与模型过期，但不改动原积分。",
            "弱结合尾部比较一位点、两位点与稀释假设并处理等优候选。",
            "在分支上做出冲突人工决定，并可回到任意旧事件。",
        ],
        "v1": {
            "power": "v1/power.csv", "schedule": "v1/schedule.csv",
            "conditions": "v1/conditions.json", "baseline": "v1/baseline.csv",
            "blank": "v1/blank.csv",
        },
        "revisions": {
            "r1_concentration": "revisions/conditions_r1.json",
            "r2_local_schedule": "revisions/schedule_r2.csv",
        },
        "designed_situations": {
            "overlap_injections": [14, 15],
            "first_injection_discard": 1,
            "blank_missing_index": 18,
            "gap_injection": 9,
            "gap_s": GAP_S,
            "local_schedule_fix_index": 12,
            "tie_region_injections": [15, 16, 17, 18],
        },
    }
    with open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("case package written to", ROOT)


if __name__ == "__main__":
    main()
