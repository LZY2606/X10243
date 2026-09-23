"""离线领域案例包生成器：滴定热谱（ITC 注射积分、稀释热与绑定模型审议）。

不使用通用表格：案例材料按仪器导出口吻组织（功率采样、注射时序、
基线曲线、空白滴定曲线、浓度单、条件单），并随附 replay.json 机器脚本。
合成数据确定性（固定随机种子），叙事完整：导入→核对→触发边界组合→
重叠峰/首注舍弃/空白缺失/浓度修订/等优模型/数据缺口/积分端点→
局部改动分支→冲突人工决定→联动复核。
"""
from __future__ import annotations

import json
import os
import numpy as np

CASE_ID = "itc_titration_review"
CASE_TITLE = "滴定热谱：ITC 注射积分、稀释热与绑定模型审议"

DT = 0.5
DURATION = 1200.0
WINDOW_S = 60.0

# 主滴定：injection 5→180s 造成与 #4 的重叠峰；#6 窗内制造数据缺口
INJ_TIMES = [120.0, 240.0, 300.0, 360.0, 390.0, 480.0, 540.0,
             600.0, 660.0, 720.0, 780.0, 840.0]
INJ_VOL_UL = 8.0
V0_UL = 200.0
CELL_M_MM = 0.20
SYR_L_MM = 4.0
TEMP_C = 25.0

# 空白只给 11 注（缺第 12 注对应）
BLANK_INJ_TIMES = [120.0, 240.0, 300.0, 360.0, 390.0, 480.0, 540.0,
                   600.0, 660.0, 720.0, 780.0]


def _kernel_response(t_rel, dH_uJ):
    """注射后峰：快速上升 + 指数衰减的卷积形状（允许重叠叠加）。"""
    rise = np.exp(-t_rel / 2.0)
    decay = np.exp(-t_rel / 18.0)
    shape = decay * (1.0 - rise)
    shape[t_rel < 0] = 0.0
    # 归一化到积分 ≈ 1（解析近似系数），再乘 dH
    denom = 18.0 - 1.0 / (1.0 / 1.0 + 1.0 / 18.0)
    return shape / max(denom, 1e-9) * dH_uJ


def _thermogram(times, inj_times, heats_uJ, seed, *, drift_slope,
                gap_windows=(), noise=0.06):
    rng = np.random.default_rng(seed)
    p = 0.05 + drift_slope * (times - times[0])
    for t_inj, q in zip(inj_times, heats_uJ):
        rel = times - t_inj
        p = p + _kernel_response(rel, q)
    p = p + rng.normal(0.0, noise, size=times.shape)
    p_raw = p.copy()
    mask = np.zeros_like(times, dtype=bool)
    for a, b in gap_windows:
        mask |= (times >= a) & (times <= b)
    p[mask] = np.nan
    return p, p_raw, mask


def _q_forward():
    """按一位点模型正演生成每注射热（µJ），真实衰减形态。"""
    from .binding import _theta_one
    K = 5.0  # mM^-1（结合较强，呈明显 S 型衰减）
    dH = -2500.0  # µJ/µmol（放热）
    heats = []
    cum = 0.0
    for _ in INJ_TIMES:
        cum += INJ_VOL_UL
        X = cum * SYR_L_MM / V0_UL / CELL_M_MM
        th = float(_theta_one(np.array([X]), K)[0])
        # q ≈ dH * V0(µL) * θ（µJ/µmol × mM × µL ≈ µJ 量级一致）
        heats.append(dH * V0_UL * th)
    return heats, K, dH


def build_case_pack(out_dir: str) -> dict:
    times = np.arange(0.0, DURATION + DT / 2, DT)
    heats, K_true, dH_true = _q_forward()

    # ---- 功率曲线：#6 窗内部缺口 + #10 左端点缺口（积分端点检验）----
    gap_windows = [(486.0, 494.0), (720.0, 724.0)]
    power, _power_clean, gap_mask = _thermogram(
        times, INJ_TIMES, heats, seed=20260923, drift_slope=2.0e-4,
        gap_windows=gap_windows)

    # ---- 基线曲线：缓慢漂移（独立版本化）----
    baseline = 0.05 + 2.0e-4 * (times - times[0]) \
        + 0.008 * np.sin((times - times[0]) / 300.0)

    # ---- 空白滴定：只有小稀释热，且缺第 12 注 ----
    blank_heats = [4.0 * np.exp(-0.05 * i) for i in
                   range(len(BLANK_INJ_TIMES))]
    blank_p, _, _ = _thermogram(
        times, BLANK_INJ_TIMES, blank_heats, seed=99, drift_slope=0.0,
        gap_windows=(), noise=0.04)

    os.makedirs(os.path.join(out_dir, "materials"), exist_ok=True)
    mdir = os.path.join(out_dir, "materials")

    _write_run_csv(os.path.join(mdir, "power.csv"),
                   times, power, "power", "uW", gap_mask)
    _write_plain_csv(os.path.join(mdir, "baseline.csv"),
                     times, baseline, "uW")
    _write_blank_csv(os.path.join(mdir, "blank.csv"),
                     times, blank_p, BLANK_INJ_TIMES)
    _write_injections_csv(os.path.join(mdir, "injections.csv"))

    conc_v1 = {
        "cell_macromolecule": {"value": "0.20", "unit": "mM"},
        "syringe_ligand": {"value": "4.0", "unit": "mM"},
        "note": "初始浓度单（配液记录）",
    }
    # v2：修订配体浓度（如复称后修正）
    conc_v2 = {
        "cell_macromolecule": {"value": "0.20", "unit": "mM"},
        "syringe_ligand": {"value": "3.6", "unit": "mM"},
        "per_injection": {"7": {"value": "3.55", "unit": "mM"}},
        "note": "复称后的浓度修订（模型应过期，原积分不变）",
    }
    with open(os.path.join(mdir, "concentration_v1.json"), "w",
              encoding="utf-8") as f:
        json.dump(conc_v1, f, ensure_ascii=False, indent=2)
    with open(os.path.join(mdir, "concentration_v2_revised.json"), "w",
              encoding="utf-8") as f:
        json.dump(conc_v2, f, ensure_ascii=False, indent=2)

    conditions = {
        "temperature": {"value": "25.0", "unit": "degC"},
        "cell_volume": {"value": "200.0", "unit": "uL"},
        "reference_power": {"value": "10", "unit": "uW"},
        "note": "ITC200 型池；时间基准为功率序列首样本起的秒",
    }
    with open(os.path.join(mdir, "conditions.json"), "w",
              encoding="utf-8") as f:
        json.dump(conditions, f, ensure_ascii=False, indent=2)

    replay = _replay_script()
    with open(os.path.join(out_dir, "replay.json"), "w",
              encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False, indent=2)

    manifest = {
        "case_id": CASE_ID,
        "title": CASE_TITLE,
        "domain": "isothermal_titration_calorimetry",
        "time_basis": "seconds since first power sample",
        "sampling_interval_s": DT,
        "integration_window_s": WINDOW_S,
        "synthetic": True,
        "rng_seed": 20260923,
        "true_parameters": {"K_mM_inv": K_true, "dH_uJ_per_umol": dH_true},
        "materials": [
            "materials/power.csv", "materials/injections.csv",
            "materials/baseline.csv", "materials/blank.csv",
            "materials/concentration_v1.json",
            "materials/concentration_v2_revised.json",
            "materials/conditions.json",
        ],
        "replay": "replay.json",
    }
    with open(os.path.join(out_dir, "case.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def _csv_line(t, p, unit="uW"):
    return f"{t:.2f},{'' if (p is None or (isinstance(p,float) and np.isnan(p))) else f'{p:.5f}'}"


def _write_run_csv(path, times, power, value_name, unit, gap_mask=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# time_unit: s\n")
        f.write(f"# {value_name}_unit: {unit}\n")
        f.write("# time_basis: seconds since first sample\n")
        f.write("time,power\n")
        for t, p in zip(times, power):
            if p is None or (isinstance(p, float) and np.isnan(p)):
                f.write(f"{t:.2f},N/A\n")
            else:
                f.write(f"{t:.2f},{p:.5f}\n")


def _write_plain_csv(path, times, values, unit):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# time_unit: s\n")
        f.write(f"# power_unit: {unit}\n")
        f.write("time,power\n")
        for t, p in zip(times, values):
            f.write(f"{t:.2f},{p:.5f}\n")


def _write_injections_csv(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# time_unit: s\n# volume_unit: uL\n")
        f.write("index,time,volume\n")
        for i, t in enumerate(INJ_TIMES, start=1):
            f.write(f"{i},{t:.2f},{INJ_VOL_UL:.2f}\n")


def _write_blank_csv(path, times, blank_p, inj_times):
    # 空白自身的注射时序放在随行（非注射行留空），体现“空白也是一次滴定”
    inj_at = {t: (i + 1) for i, t in enumerate(inj_times)}
    with open(path, "w", encoding="utf-8") as f:
        f.write("# time_unit: s\n# power_unit: uW\n")
        f.write("# inj_time_unit: s\n# inj_volume_unit: uL\n")
        f.write("time,power,inj_index,inj_time,inj_volume\n")
        for t, p in zip(times, blank_p):
            idx = inj_at.get(round(t, 2), "")
            if idx != "":
                f.write(f"{t:.2f},{p:.5f},{idx},{t:.2f},{INJ_VOL_UL:.2f}\n")
            else:
                f.write(f"{t:.2f},{p:.5f},,,\n")


def _local_patch_power(times, power, heats):
    """只影响局部依赖的源数据改动：仅在 #4/#5 各自积分窗内增加拖尾热。

    拖尾严格截断在 [注射时刻, 注射时刻+积分窗]，因此任何其它注射窗内的
    采样点都不变（用每条积分的 input_digest 验证影响范围）。
    """
    patched = power.copy()
    valid = ~np.isnan(patched)
    for center in (360.0, 390.0):
        rel = times - center
        in_own = (rel >= 0.0) & (rel <= WINDOW_S)
        tail = np.where(in_own, 6.0 * np.exp(-rel / 45.0), 0.0)
        patched[valid] = patched[valid] + tail[valid]
    return patched


def _write_local_patch(out_dir, times, power, heats):
    patched = _local_patch_power(times, power, heats)
    path = os.path.join(out_dir, "materials", "power_patched_local.csv")
    _write_run_csv(path, times, patched, "power", "uW")
    return path


def _write_flat_branch(out_dir, times):
    """等优模型案例：平坦响应（稀释热≈恒定），一位点与稀释假设等优。"""
    rng = np.random.default_rng(7)
    p = 0.05 + rng.normal(0.0, 0.05, size=times.shape)
    flat_inj = [150.0 + 90.0 * i for i in range(10)]  # 90s 间隔，无重叠
    for ti in flat_inj:
        rel = times - ti
        p = p + _kernel_response(rel, 3.0)
    path_p = os.path.join(out_dir, "materials", "power_flat.csv")
    _write_run_csv(path_p, times, p, "power", "uW")
    path_i = os.path.join(out_dir, "materials", "injections_flat.csv")
    with open(path_i, "w", encoding="utf-8") as f:
        f.write("# time_unit: s\n# volume_unit: uL\n")
        f.write("index,time,volume\n")
        for i, t in enumerate(flat_inj, start=1):
            f.write(f"{i},{t:.2f},{INJ_VOL_UL:.2f}\n")
    return path_p, path_i, flat_inj


def _replay_script() -> dict:
    """机器可读取的重放脚本（事件式，覆盖全部要求情形）。"""
    return {
        "script": "itc-replay-1",
        "case_id": CASE_ID,
        "steps": [
            # ---- 初始导入：六类材料分开版本化，先核对原材料 ----
            {"op": "import", "chain": "main", "kind": "power",
             "file": "materials/power.csv",
             "expect": {"n_samples": int(DURATION / DT) + 1,
                        "gap_samples_diag": True}},
            {"op": "import", "chain": "main", "kind": "injections",
             "file": "materials/injections.csv",
             "expect": {"n_injections": 12}},
            {"op": "import", "chain": "main", "kind": "baseline",
             "file": "materials/baseline.csv"},
            {"op": "import", "chain": "main", "kind": "blank",
             "file": "materials/blank.csv",
             "expect": {"blank_schedule_injections": 11}},
            {"op": "import", "chain": "main", "kind": "concentration",
             "file": "materials/concentration_v1.json"},
            {"op": "import", "chain": "main", "kind": "conditions",
             "file": "materials/conditions.json",
             "expect": {"temperature_degC": TEMP_C,
                        "cell_volume_uL": V0_UL}},

            # ---- 首注舍弃 + 积分：重叠峰、缺口截断、端点裁剪同时发生 ----
            {"op": "set_first_discard", "chain": "main",
             "ordinals": [1], "window_s": WINDOW_S},
            {"op": "integrate", "chain": "main", "window_s": WINDOW_S,
             "discard_ordinals": [1],
             "expect": {
                 "n_entries": 12,
                 "overlap_ordinals": [4, 5],
                 "gap_clipped_ordinals": [6],
                 "left_endpoint_gap_ordinal": 10,
                 "discarded_ordinals": [1]}},

            # ---- 稀释热：显式 by_index 对应；第 12 注空白对应缺失 ----
            {"op": "apply_blank", "chain": "main", "rule": "by_index",
             "expect": {"matched": 10,
                        "blank_missing_ordinals": [12]}},

            # ---- 一位点拟合（主数据明显非平坦，唯一占优）----
            {"op": "fit", "chain": "main",
             "models": ["dilution_only", "one_site", "two_site"],
             "expect": {"winners": ["one_site"],
                        "under_determined": ["two_site"]}},

            # ---- 审阅者接受一位点（被采纳结果）----
            {"op": "decide", "chain": "main", "verdict": "accept",
             "target": "winner", "note": "一位点 S 型衰减、残差无结构"},
            {"op": "decide", "chain": "main", "verdict": "reject",
             "target": "model:dilution_only",
             "note": "恒定稀释热假设无法解释衰减（未采纳但保留）"},

            # ---- 浓度修订：旧拟合过期（保留），原积分不变 ----
            {"op": "revise_concentration", "chain": "main",
             "file": "materials/concentration_v2_revised.json",
             "expect": {"invalidates": "fit",
                        "integration_unchanged": True}},
            {"op": "fit", "chain": "main",
             "models": ["dilution_only", "one_site", "two_site"],
             "as": "fit_revised",
             "expect": {"winners": ["one_site"],
                        "previous_fit_marked_stale": True}},
            {"op": "decide", "chain": "main", "verdict": "accept",
             "target": "alias:fit_revised",
             "note": "浓度修订后仍接受一位点结论"},

            # ---- 局部源数据改动分支：只影响 #4/#5 的积分 ----
            {"op": "fork", "chain": "local_patch",
             "from_step": "set_first_discard", "note": "局部功率拖尾修订"},
            {"op": "patch_power", "chain": "local_patch",
             "file": "materials/power_patched_local.csv",
             "scope": "injections 4-5 tail",
             "expect": {"integration_input_changed_ordinals": [4, 5]}},
            {"op": "set_first_discard", "chain": "local_patch",
             "ordinals": [1], "window_s": WINDOW_S},
            {"op": "integrate", "chain": "local_patch", "window_s": WINDOW_S,
             "discard_ordinals": [1], "as": "int_local"},
            {"op": "apply_blank", "chain": "local_patch",
             "rule": "by_index", "as": "dil_local"},

            # ---- 等优模型分支：平坦数据，一位点与稀释假设等优，全部保留 ----
            {"op": "import", "chain": "flat", "kind": "power",
             "file": "materials/power_flat.csv", "fork_from_root": True},
            {"op": "import", "chain": "flat", "kind": "injections",
             "file": "materials/injections_flat.csv"},
            {"op": "import", "chain": "flat", "kind": "baseline",
             "file": "materials/baseline.csv"},
            {"op": "import", "chain": "flat", "kind": "blank",
             "file": "materials/blank.csv"},
            {"op": "import", "chain": "flat", "kind": "concentration",
             "file": "materials/concentration_v1.json"},
            {"op": "import", "chain": "flat", "kind": "conditions",
             "file": "materials/conditions.json"},
            {"op": "set_first_discard", "chain": "flat",
             "ordinals": [], "window_s": WINDOW_S},
            {"op": "integrate", "chain": "flat", "window_s": WINDOW_S,
             "discard_ordinals": []},
            {"op": "apply_blank", "chain": "flat", "rule": "by_index"},
            {"op": "fit", "chain": "flat",
             "models": ["dilution_only", "one_site", "two_site"],
             "as": "fit_flat",
             "expect": {"tie_includes": ["dilution_only", "one_site"],
                        "no_auto_when_tie": True}},

            # ---- 互相冲突的人工决定：同一份修订拟合，两条链 accept/reject ----
            {"op": "fork", "chain": "conflict_reject",
             "from_step": "accept_fit_revised",
             "note": "第二位审阅者持否定意见"},
            {"op": "decide", "chain": "conflict_reject",
             "verdict": "reject", "target": "alias:fit_revised",
             "note": "认为修订后浓度证据不足"},
        ],
    }
