"""离线案例包：滴定热谱——ITC 注射积分、稀释热与绑定模型审议。

完整叙事（不是彼此无关的按钮）：
1. 初始导入：功率采样、注射体积和时间、浓度、温度、池体积、基线、
   空白曲线——七类材料分开版本化，原始表示完整保留。
2. 边界组合：重叠注射峰（末四针 60s 间隔）、首注舍弃、空白对应缺失
   （空白缺第 12 针）、数据缺口（1610-1640s 落在第 10 针积分窗内）、
   积分端点截断、多个等优模型。
3. 一次参数修订（浓度 50→45 µM）：模型过期但不改原积分。
4. 一次只影响局部依赖的源数据改动（基线 v2）：只有积分及其下游重算。
5. 一次互相冲突的人工决定（main 接受一位点，review-b 否定它并接受
   多位点）：两个分支的决定都保留。
"""
from __future__ import annotations

import numpy as np

from . import binding, pipeline
from .store import Store

TAU_S = 25.0
GAP = (1610.0, 1640.0)
TRUE = {"n": 1.0, "log10_Ka": float(np.log10(2.0)), "dH": -30.0}


def _schedule():
    times = [300 + 130 * i for i in range(16)] + [2380 + 60 * i for i in range(4)]
    vols = [2.0] + [10.0] * 19
    return times, vols


def _true_heats(vols):
    ctx = {"n_inj": 20, "v0_L": 1.4e-3, "mt0_uM": 50.0, "xt_uM": 500.0,
           "inj_volumes_uL": vols}
    p = np.array([TRUE["n"], TRUE["log10_Ka"], TRUE["dH"]])
    return binding.predict("one_site", p, ctx)


def _trace_raw(times, heats, rng, bad_point=False):
    t_all = np.arange(0.0, 2700.0, 1.0)
    keep = (t_all < GAP[0]) | (t_all > GAP[1])
    t = t_all[keep]
    y = 0.5 + 0.00002 * t + rng.normal(0.0, 0.004, size=len(t))
    for ti, qi in zip(times, heats):
        m = t >= ti
        y[m] += (qi / TAU_S) * np.exp(-(t[m] - ti) / TAU_S)
    points = []
    bad_done = False
    for tv, yv in zip(t, y):
        if bad_point and not bad_done and tv >= 900.0:
            points.append([round(float(tv), 3), "n/a"])  # 无法转换：留在原位置
            bad_done = True
            continue
        points.append([round(float(tv), 3), round(float(yv), 5)])
    return {"meta": {"unit_t": "s", "unit_y": "µJ/s",
                     "basis_t": "epoch:2026-09-19T09:00:00+09:00",
                     "instrument": "sim-VP-ITC", "cell": "sample"},
            "points": points}


def build_materials():
    rng = np.random.default_rng(42)
    times, vols = _schedule()
    heats = _true_heats(vols) + rng.normal(0.0, 0.02, size=20)
    blank_times = [t for i, t in enumerate(times) if i != 12]  # 空白缺第 12 针
    blank_heats = -0.12 + rng.normal(0.0, 0.01, size=19)
    return {
        "power_trace": _trace_raw(times, heats, rng, bad_point=True),
        "injection_schedule": {
            "meta": {"unit_t": "s", "unit_volume": "µL",
                     "basis_t": "epoch:2026-09-19T09:00:00+09:00"},
            "injections": [{"t": f"{t} s", "volume": f"{v} µL"}
                           for t, v in zip(times, vols)]},
        "concentration": {"syringe": "500 umol/L", "cell": "50 µM",
                          "note": "配体→蛋白"},
        "temperature": {"value": "25 C"},
        "cell_volume": {"value": "1.4 mL"},
        "baseline": {"meta": {"unit_t": "s", "unit_y": "µJ/s",
                              "basis_t": "epoch:2026-09-19T09:00:00+09:00"},
                     "points": [[float(x), round(0.5 + 0.00002 * x, 5)]
                                for x in range(0, 2701, 5)]},
        "blank_trace": _trace_raw(blank_times, blank_heats, rng),
    }

BASELINE_V2 = {"meta": {"unit_t": "s", "unit_y": "µJ/s",
                        "basis_t": "epoch:2026-09-19T09:00:00+09:00"},
               "points": [[float(x), round(0.52 + 0.00002 * x, 5)]
                          for x in range(0, 2701, 5)]}

CONCENTRATION_V2 = {"syringe": "500 umol/L", "cell": "45 µM",
                    "note": "重新称量后修订"}


def seed(store: Store) -> None:
    mats = build_materials()
    for kind, raw in mats.items():
        store.register_material(kind, f"mat-{kind}", raw, branch="main")
    # 初始计算链：首注舍弃 + 重叠峰/缺口规则进入事件
    pipeline.compute_integrations(store, "power_trace",
                                  {"discard_first": True})
    sub_id, _, _ = pipeline.compute_blank_subtraction(
        store, {"matching": "index"})
    fit_id, _, fit = pipeline.compute_model_fit(store)
    kept = [c for c in fit.get("candidates", []) if c.get("kept")]
    one_site = f"{fit_id}#one_site"
    store.add_decision(one_site, "accept",
                       "一位点模型残差无结构，物理上最简", branch="main")
    # 参数修订：浓度 50→45 µM（模型过期，原积分不变）
    store.register_material(
        "concentration", "mat-concentration", CONCENTRATION_V2, branch="main",
        basis={"rule": "concentration_revision",
               "meaning": "浓度修订使模型过期但不改原积分"})
    # 局部源数据改动：基线 v2（只影响积分及其下游）
    store.register_material(
        "baseline", "mat-baseline", BASELINE_V2, branch="main",
        basis={"rule": "local_source_change",
               "meaning": "只影响声明了该依赖的计算"})
    # 用修订后的材料重算（旧计算保留，可回查）
    pipeline.compute_integrations(store, "power_trace",
                                  {"discard_first": True})
    pipeline.compute_blank_subtraction(store, {"matching": "index"})
    fit2_id, _, fit2 = pipeline.compute_model_fit(store)
    # 互相冲突的人工决定：另一分支否定一位点、接受多位点
    store.add_decision(one_site, "reject",
                       "复核人认为一位点过简，残差在末端有趋势",
                       branch="review-b")
    store.add_decision(f"{fit_id}#two_site", "accept",
                       "复核人偏好两位点解释", branch="review-b")
    store.add_event("casepack_sealed", {
        "kept_models": [c["model"] for c in kept],
        "basis": {"rule": "casepack_narrative",
                  "meaning": "反例档案：单一来源改动的影响必须被限制在正确范围"}},
        branch="main")
