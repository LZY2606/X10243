"""Generate the offline case package for
"滴定热谱：ITC 注射积分、稀释热与绑定模型审议".

The case is fully deterministic (no noise) so replay is reproducible.
It exercises: overlapping post-injection peaks, first-injection discard,
missing blank correspondence, concentration revision, multiple equally-good
models, a data gap crossing an integration window, and window endpoints.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from itc.fit import cell_trajectory, heat_signal  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "casepack", "itc_titration_review_case.json")

# --- experiment layout -------------------------------------------------
N_INJ = 12
V0_L = 1.4e-3                 # cell volume 1.4 mL
MT0_M = 50e-6                 # cell macromolecule 50 uM
CS_M = 600e-6                 # syringe ligand 600 uM (v1; revised to 580 later)
TEMP_C = 25.0                 # degC in raw record
K_TRUE = 1.0e6                # 1/M
DH_TRUE = -4.0e4              # J/mol
N_TRUE = 1.0
TAU_S = 25.0                  # peak decay constant
DT_S = 1.0
INJ_TIMES = [60.0 + 70.0 * i for i in range(N_INJ)]   # tight spacing -> overlap
VOLUMES_UL = [0.5] + [2.5] * (N_INJ - 1)
BASE_A_UJS = 5.0              # uJ/s
BASE_B_UJS = 0.0004           # drift uJ/s per s
GAP = [INJ_TIMES[6] + 6.0, INJ_TIMES[6] + 30.0]       # crosses window of inj 6
BLANK_N_INJ = 10              # blank misses correspondence for inj 10, 11
BLANK_Q_UJ = 30.0             # constant dilution heat per blank injection


def true_heats_uj():
    dv = np.array([v * 1e-6 for v in VOLUMES_UL])
    Mt, Xt = cell_trajectory(dv, V0_L, MT0_M, CS_M)
    S = heat_signal(Mt, Xt, dv, V0_L, K_TRUE, N_TRUE)
    return DH_TRUE * S * 1e6    # J -> uJ


def make_power(heats_uj, n_inj, gap):
    duration = INJ_TIMES[n_inj - 1] + 200.0
    t = np.arange(0.0, duration, DT_S)
    p = np.full_like(t, BASE_A_UJS) + BASE_B_UJS * t
    for i in range(n_inj):
        onset = INJ_TIMES[i]
        contrib = np.where(t >= onset,
                           (heats_uj[i] / TAU_S) * np.exp(-(t - onset) / TAU_S),
                           0.0)
        p = p + contrib
    gaps = []
    if gap is not None:
        keep = ~((t >= gap[0]) & (t < gap[1]))
        t = t[keep]
        p = p[keep]
        gaps = [list(gap)]
    return {
        "time": {"values": [round(float(x), 3) for x in t], "unit": "s",
                 "basis": "seconds_since_run_start"},
        "power": {"values": [round(float(x), 6) for x in p], "unit": "uJ/s"},
        "gaps": gaps,
    }


def main():
    heats_uj = true_heats_uj()
    blank_heats_uj = np.full(BLANK_N_INJ, BLANK_Q_UJ)

    power = make_power(heats_uj, N_INJ, GAP)
    blank_power = make_power(blank_heats_uj, BLANK_N_INJ, None)

    schedule = {
        "injections": [
            {"time": {"value": INJ_TIMES[i], "unit": "s"},
             "volume": {"value": VOLUMES_UL[i], "unit": "uL"}}
            for i in range(N_INJ)
        ]
    }
    baseline = {"kind": "linear",
                "a": {"value": BASE_A_UJS, "unit": "uJ/s"},
                "b": {"value": BASE_B_UJS, "unit": "uJ/s"},
                "t_ref": {"value": 0.0, "unit": "s"}}
    blank_baseline = {"kind": "linear",
                      "a": {"value": BASE_A_UJS, "unit": "uJ/s"},
                      "b": {"value": BASE_B_UJS, "unit": "uJ/s"},
                      "t_ref": {"value": 0.0, "unit": "s"}}

    conc_v1 = {"cell": {"value": 50.0, "unit": "uM"},
               "syringe": {"value": 600.0, "unit": "uM"}}
    conc_v2 = {"cell": {"value": 50.0, "unit": "uM"},
               "syringe": {"value": 580.0, "unit": "uM"}}

    case = {
        "case_id": "itc-titration-review-001",
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "narrative": [
            "生物物理实验室导入原始功率曲线、注射时序和空白对照，审阅每次注射积分与绑定模型候选。",
            "初始版本完整呈现功率采样、注射体积和时间、浓度、温度、池体积、基线和空白曲线，分开版本化。",
            "随后安排一次浓度修订、一次只影响空白扣除的源数据改动和一次互相冲突的人工决定。",
            "边界情形：重叠注射峰、首注舍弃、空白对应缺失、浓度修订、多个等优模型、数据缺口与积分端点。",
        ],
        "materials": {
            "power_curve": power,
            "injection_schedule": schedule,
            "concentration": conc_v1,
            "temperature": {"value": TEMP_C, "unit": "degC",
                            "basis": "cell_block"},
            "cell_volume": {"value": 1.4, "unit": "mL"},
            "baseline": baseline,
            "blank_curve": {"power": blank_power, "baseline": blank_baseline},
        },
        "script": [
            {"op": "integrate",
             "params": {"discard_first": True, "max_window_s": 240.0},
             "comment": "首注舍弃；注射后峰可重叠；积分窗不跨越数据缺口"},
            {"op": "blank_subtract", "rule": {"kind": "by_index"},
             "comment": "稀释热扣除必须说明对应规则；空白只有 10 针，后两针对应缺失"},
            {"op": "fit", "params": {"equal_rel_tol": 0.5},
             "comment": "一位点、多位点与稀释假设并列比较，保留等优候选"},
            {"op": "decide", "candidate": "one_site", "action": "accept",
             "note": "主分支接受一位点模型"},
            {"op": "revise", "category": "concentration", "payload": conc_v2,
             "note": "浓度修订 600 -> 580 uM：使既有模型过期但不改原积分"},
            {"op": "fit", "params": {"equal_rel_tol": 0.5},
             "comment": "修订后的浓度产生新一轮拟合"},
            {"op": "revise", "category": "blank_curve",
             "payload": {"power": blank_power,
                         "baseline": {"kind": "linear",
                                      "a": {"value": BASE_A_UJS + 0.1,
                                            "unit": "uJ/s"},
                                      "b": {"value": BASE_B_UJS,
                                            "unit": "uJ/s"},
                                      "t_ref": {"value": 0.0, "unit": "s"}}},
             "note": "只影响空白扣除的源数据改动"},
            {"op": "blank_subtract", "rule": {"kind": "by_index"},
             "comment": "局部依赖改动只触发空白扣除重算"},
            {"op": "branch", "name": "branch-b"},
            {"op": "decide", "branch": "branch-b", "candidate": "dilution",
             "action": "accept",
             "note": "冲突决定：branch-b 认为信号以稀释热为主"},
            {"op": "decide", "candidate": "two_site", "action": "reject",
             "note": "主分支否定多位点：证据不足"},
        ],
        "expected": {
            "overlap_min_count": 4,
            "truncated_injection": 6,
            "gap": GAP,
            "excluded_first": 0,
            "missing_blank_injections": [10, 11],
            "stale_after_revision": "fit",
            "untouched_after_revision": "integration",
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(case, fh, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
