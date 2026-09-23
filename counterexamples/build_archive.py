"""Build counterexamples/archive.json (run once; output committed).

Baseline: 8-injection titration, every material kind versioned
separately.  Each case changes exactly one direct source and declares
the expected blast radius; the replay harness (itc/counterarchive.py)
verifies the impact stays inside it.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N_INJ = 8
TIMES = [200.0 + 150.0 * i for i in range(N_INJ)]
VOLUMES = [2.0] + [10.0] * (N_INJ - 1)
CELL_MM, SYR_MM, CELL_UL = 0.1, 1.2, 1400.0
TAU, BASELINE_P = 25.0, 0.5


def bound(K, mt, xt):
    a = K * (mt + xt) + 1.0
    return (a - math.sqrt(a * a - 4 * K * K * mt * xt)) / (2 * K)


def heats():
    mt, xt, prev, out = CELL_MM, 0.0, 0.0, []
    for v in VOLUMES:
        keep = 1.0 - v / CELL_UL
        mt *= keep
        xt = xt * keep + SYR_MM * v / CELL_UL
        b = bound(100.0, mt, xt)
        out.append(CELL_UL * -10.0 * (b - prev * keep) + 0.5)
        prev = b
    return out


def power_curve(times, gap=None):
    hs = heats()
    inj = sorted(zip(times, hs))
    out_t, out_p, t = [], [], 0.0
    while t <= times[-1] + 200.0:
        p = BASELINE_P
        for it, ih in inj:
            if t >= it:
                p += (ih / TAU) * math.exp(-(t - it) / TAU)
        if gap is None or not (gap[0] <= t <= gap[1]):
            out_t.append(round(t, 3))
            out_p.append(round(p, 6))
        t += 1.0
    return out_t, out_p


def materials(times, gap=None, blank_n=6, syringe=1.2):
    t, p = power_curve(times, gap)
    m = {
        "power_sampling": {
            "time": {"values": t, "unit": "s", "basis": "instrument_clock"},
            "power": {"values": p, "unit": "uJ/s"},
            "gaps": ([] if gap is None else [{
                "start": {"value": gap[0], "unit": "s"},
                "end": {"value": gap[1], "unit": "s"},
                "reason": "counterexample gap"}])},
        "injection_schedule": {"injections": [
            {"time": {"value": tt, "unit": "s", "basis": "instrument_clock"},
             "volume": {"value": vv, "unit": "uL"}}
            for tt, vv in zip(times, VOLUMES)]},
        "concentrations": {"cell": {"value": CELL_MM, "unit": "mM"},
                           "syringe": {"value": syringe, "unit": "mM"}},
        "temperature": {"value": 25.0, "unit": "degC"},
        "cell_volume": {"value": 1.4, "unit": "mL"},
        "baseline": {"kind": "constant", "coefficients": [BASELINE_P],
                     "unit": "uJ/s"},
        "blank_curve": {"heats": {"values": [0.5] * blank_n, "unit": "uJ"}},
    }
    return m


def main():
    base = materials(TIMES)
    overlap_times = list(TIMES)
    overlap_times[4] = overlap_times[3] + 30.0
    gap_times = list(TIMES)
    gap = (TIMES[2] + 30.0, TIMES[2] + 70.0)
    archive = {
        "id": "itc-counterexamples-001",
        "title": "ITC 注射积分/绑定模型 反例档案",
        "baseline": {"materials": base,
                     "note": "全部材料分开版本化；空白仅 6 条（8 次注射）"},
        "cases": [
            {"id": "overlap_peaks",
             "change": {"kind": "injection_schedule",
                        "payload": materials(overlap_times)
                        ["injection_schedule"]},
             "expected": {
                 "judgment_rules": ["post_injection_peaks_may_overlap"],
                 "integration_windows_changed": [3, 4],
                 "integration_windows_unchanged": [0, 1, 2, 5, 6, 7],
                 "fits_stale": True}},
            {"id": "first_injection_discard",
             "change": {"decision": {"target": "injection:0",
                                     "action": "discard"}},
             "expected": {
                 "integration_unchanged": True,
                 "fits_stale": True,
                 "fit_excludes": [0]}},
            {"id": "blank_missing",
             "change": {"kind": "blank_curve",
                        "payload": materials(TIMES, blank_n=5)
                        ["blank_curve"]},
             "expected": {
                 "judgment_rules": ["dilution_correction_requires_"
                                    "correspondence"],
                 "blank_missing_at": [5, 6, 7],
                 "blank_subtracted_at": [0, 1, 2, 3, 4]}},
            {"id": "concentration_revision",
             "change": {"kind": "concentrations",
                        "payload": {"cell": {"value": 0.1, "unit": "mM"},
                                    "syringe": {"value": 1.1,
                                                "unit": "mM"}}},
             "expected": {
                 "judgment_rules": ["revision_stales_dependents_"
                                    "preserves_parents"],
                 "integration_unchanged": True,
                 "integration_stale": False,
                 "fits_stale": True}},
            {"id": "equal_best_models",
             "change": {"kind": "blank_curve",
                        "payload": {"heats": {"values": None,
                                              "unit": "uJ"}},
                        "special": "flat_observed"},
             "expected": {
                 "min_equal_best_candidates": 2,
                 "stable_order_across_runs": True}},
            {"id": "data_gap",
             "change": {"kind": "power_sampling",
                        "payload": materials(gap_times, gap=gap)
                        ["power_sampling"]},
             "expected": {
                 "judgment_rules": ["integration_window_never_crosses_"
                                    "data_gap"],
                 "integration_windows_changed": [2],
                 "integration_windows_unchanged": [0, 1, 3, 4, 5, 6, 7],
                 "window_2_ends_at": gap[0]}},
            {"id": "integration_endpoint",
             "change": {"kind": "power_sampling",
                        "payload": materials(gap_times, gap=gap)
                        ["power_sampling"],
                        "params": {"post_window_s": 150.0}},
             "expected": {
                 "judgment_rules": ["integration_window_never_crosses_"
                                    "data_gap"],
                 "integration_windows_changed": [2],
                 "integration_windows_unchanged": [0, 1, 3, 4, 5, 6, 7],
                 "window_2_ends_at": gap[0],
                 "window_never_crosses_gap": True}},
        ],
        "judgments": [
            {"rule": "post_injection_peaks_may_overlap",
             "meaning": "注射后峰可重叠；重叠时前一注射积分窗在下一注射"
                        "开始处结束"},
            {"rule": "integration_window_never_crosses_data_gap",
             "meaning": "积分窗不跨越数据缺口；在缺口起点截断并记录"},
            {"rule": "dilution_correction_requires_correspondence",
             "meaning": "稀释热扣除必须说明对应规则；缺对应空白时保留"
                        "未扣除值并给出诊断"},
            {"rule": "revision_stales_dependents_preserves_parents",
             "meaning": "浓度修订使模型过期但不改原积分"},
        ],
    }
    with open(os.path.join(HERE, "archive.json"), "w") as f:
        json.dump(archive, f, indent=1, ensure_ascii=False)
    print("archive written")


if __name__ == "__main__":
    main()
