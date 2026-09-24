"""离线案例包：滴定热谱 — ITC 注射积分、稀释热与绑定模型审议。

完整叙事（不拆成无关按钮）：
1. 初始导入：功率采样、注射体积和时间、浓度、温度、池体积、基线、
   空白曲线，七条流分开版本化（v1）。
2. 首次积分：重叠注射峰（第 6/7 针）、数据缺口钳窗（第 3 针）、
   空白对应缺失（第 8-11 针）同时出现。
3. 首次拟合：一位点 / 多位点 / 稀释假设候选，多个等优模型保留。
4. 人工决定：接受一位点、否定稀释假设。
5. 参数修订：浓度 v2（针筒浓度 500 -> 520 uM），拟合过期，积分不变。
6. 局部源数据改动：空白曲线 v2 补齐后四针，只影响第 8-11 针净热。
7. 互相冲突的人工决定：接受多位点与此前接受的一位点冲突，双方保留。
8. 分支 boundary-checks：首注舍弃 + 积分端点（缩短窗）检验。
"""

import numpy as np

from . import binding, service

TAU_S = 8.0
T_END_S = 1100.0
INJ_TIMES = [100, 190, 280, 370, 460, 550, 640, 665, 755, 845, 935, 1025]
INJ_VOLS = [2.0] + [8.0] * 11
GAP = (400.0, 415.0)
TRUE = {"log10K": 6.0, "dH_J_mol": -40000.0, "n": 1.0}
COND_V1 = {"cell_M": 50e-6, "syringe_M": 500e-6}
COND_V2 = {"cell_M": 50e-6, "syringe_M": 520e-6}
CELL_VOLUME_UL = 200.0
BLANK_HEAT_UJ = 3.0


def _true_heats(cond):
    c = {"cell_conc_M": cond["cell_M"], "syringe_conc_M": cond["syringe_M"],
         "cell_volume_uL": CELL_VOLUME_UL,
         "volumes_uL": np.array(INJ_VOLS)}
    x = [TRUE["log10K"], TRUE["dH_J_mol"] / 1e4, TRUE["n"]]
    return binding.model_heats("one_site", x, c)


def _baseline_samples():
    t = np.arange(0.0, T_END_S + 5.0, 5.0)
    return [[float(ti), float(0.8 + 0.0002 * ti)] for ti in t]


def _power_samples(cond, blank_only=False, n_blank_inj=None):
    rng = np.random.default_rng(7 if not blank_only else 11)
    t = np.arange(0.0, T_END_S + 1.0, 1.0)
    if not blank_only:
        mask = (t < GAP[0]) | (t > GAP[1])
        t = t[mask]
    heats = np.full(len(INJ_TIMES), BLANK_HEAT_UJ) if blank_only \
        else _true_heats(cond) + BLANK_HEAT_UJ
    n = n_blank_inj or len(INJ_TIMES)
    base = 0.8 + 0.0002 * t
    y = base.copy()
    for ti, q in zip(INJ_TIMES[:n], heats[:n]):
        m = t >= ti
        y[m] += (q / TAU_S) * np.exp(-(t[m] - ti) / TAU_S)
    y = y + rng.normal(0.0, 0.02, size=len(t))
    return [[float(a), float(b)] for a, b in zip(t, y)]


def _materials_v1():
    return {
        "power": {"time_unit": "s", "unit": "uJ/s",
                  "time_base": "instrument_clock_s_since_run_start",
                  "samples": _power_samples(COND_V1)},
        "injections": {"time_unit": "s", "volume_unit": "uL",
                       "injections": [[t, v] for t, v in zip(INJ_TIMES, INJ_VOLS)]},
        "concentration": {"cell_M": COND_V1["cell_M"],
                          "syringe_M": COND_V1["syringe_M"], "unit": "M",
                          "note": "蛋白 50 uM / 配体 500 uM（初版标定）"},
        "temperature": {"value": 298.15, "unit": "K"},
        "cell_volume": {"volume_uL": CELL_VOLUME_UL, "unit": "uL"},
        "baseline": {"time_unit": "s", "unit": "uJ/s",
                     "samples": _baseline_samples()},
        "blank": {"power": {"time_unit": "s", "unit": "uJ/s",
                            "samples": _power_samples(COND_V1, blank_only=True,
                                                      n_blank_inj=8)},
                  "injections": {"time_unit": "s", "volume_unit": "uL",
                                 "injections": [[t, v] for t, v in
                                                zip(INJ_TIMES[:8], INJ_VOLS[:8])]},
                  "note": "空白滴定只做了前 8 针（对应缺失是有意安排）"},
    }


def build_case(store):
    """把案例完整重放进 store。返回关键对象 id 便于测试与界面定位。"""
    ids = {}
    service.import_material  # noqa
    ev0 = store.append_event("case.init", {
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "narrative": "生物物理实验室导入原始功率曲线、注射时序和空白对照，"
                     "审阅每次注射积分与绑定模型候选",
        "boundary_situations": ["重叠注射峰", "首注舍弃", "空白对应缺失",
                                "浓度修订", "多个等优模型", "数据缺口与积分端点"]})
    ids["case_init"] = ev0["id"]

    for stream, payload in _materials_v1().items():
        service.import_material(store, stream, payload, note="案例初始版本 v1")

    p1 = {"pre_window_s": 0.0, "post_window_s": 50.0, "discard_first": False,
          "blank_rule": {"mode": "per_injection_index"}}
    out_int1 = service.run_integration(store, p1)
    ids["integration_v1"] = out_int1.get("result_id")

    out_fit1 = service.run_fit(store)
    ids["fit_v1_candidates"] = out_fit1.get("candidate_ids", [])
    cands1 = {store.get_result(r)["payload"]["model"]: r
              for r in ids["fit_v1_candidates"]}
    service.decide(store, "accept", cands1["one_site"],
                   "一位点模型残差结构与噪声一致，接受为工作模型")
    service.decide(store, "reject", cands1["dilution"],
                   "稀释假设无法解释放热衰减形状")
    ids["fit_v1_one_site"] = cands1["one_site"]

    # 参数修订：浓度 v2 —— 拟合过期，积分不变。
    service.import_material(store, "concentration",
                            {"cell_M": COND_V2["cell_M"],
                             "syringe_M": COND_V2["syringe_M"], "unit": "M",
                             "note": "重新标定针筒浓度 500 -> 520 uM"},
                            note="参数修订：浓度 v2")

    # 局部源数据改动：空白曲线 v2 补齐后四针。
    blank_v2 = {"power": {"time_unit": "s", "unit": "uJ/s",
                          "samples": _power_samples(COND_V1, blank_only=True)},
                "injections": {"time_unit": "s", "volume_unit": "uL",
                               "injections": [[t, v] for t, v in
                                              zip(INJ_TIMES, INJ_VOLS)]},
                "note": "补做后四针空白滴定"}
    service.import_material(store, "blank", blank_v2, note="局部源数据改动：空白 v2")
    out_int2 = service.run_integration(store, p1)
    ids["integration_v2"] = out_int2.get("result_id")
    out_fit2 = service.run_fit(store)
    ids["fit_v2_candidates"] = out_fit2.get("candidate_ids", [])
    cands2 = {store.get_result(r)["payload"]["model"]: r
              for r in ids["fit_v2_candidates"]}

    # 互相冲突的人工决定：先接受一位点，再接受多位点（冲突双方保留）。
    service.decide(store, "accept", cands2["one_site"],
                   "浓度修订后一位点仍为工作模型")
    conflict = service.decide(store, "accept", cands2["two_site"],
                              "审阅者乙认为多位点等优，应并列接受")
    ids["conflict_event"] = conflict["event"]["id"]

    # 分支：首注舍弃 + 积分端点检验（缩短积分窗）。
    base_seq = out_int1["event"]["seq"]
    store.create_branch("boundary-checks", base_seq)
    store.append_event("branch.create", {"name": "boundary-checks",
                                         "base_seq": base_seq,
                                         "purpose": "首注舍弃与积分端点边界检验"})
    p2 = {"pre_window_s": 0.0, "post_window_s": 30.0, "discard_first": True,
          "blank_rule": {"mode": "per_injection_index"}}
    out_int3 = service.run_integration(store, p2, branch="boundary-checks")
    ids["integration_branch"] = out_int3.get("result_id")
    return ids


def load_case_into(store):
    """幂等入口：空库时构建案例。"""
    if store.is_empty():
        return build_case(store)
    return None
