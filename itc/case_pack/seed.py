"""Seed a project from the offline case pack and replay the full scenario.

Replay really exercises: overlapping post-injection peaks, first-injection
discard, missing blank correspondence, concentration revision, multiple
equi-optimal models, data gaps and integration endpoints.
"""
from __future__ import annotations

from domain.services import Service
from domain.state import DEFAULT_BRANCH

from .generate import MANIFEST, read_case_file


def build(store=None, *, boundary_max_evals: int = 200) -> Service:
    svc = Service(store=store)

    # ---- main branch: titration narrative -----------------------------------
    svc.import_material("power", "sample", read_case_file("sample_power"),
                        version="v1", note="原始功率采样")
    svc.import_material("schedule", "sample",
                        read_case_file("sample_schedule"), version="v1",
                        note="注射体积和时间")
    svc.import_material("concentration", "sample",
                        read_case_file("sample_concentration"), version="v1",
                        note="浓度")
    svc.import_material("conditions", "sample",
                        read_case_file("sample_conditions"), version="v1",
                        note="温度与池体积")
    svc.import_material("baseline", "sample",
                        read_case_file("sample_baseline"), version="v1",
                        note="基线独立版本化")
    # second branch forks BEFORE blank controls are imported: the boundary
    # branch therefore genuinely has no blank correspondence available.
    boundary_base = svc.project.branch_events(DEFAULT_BRANCH)[-1]
    svc.create_branch("boundary", DEFAULT_BRANCH, boundary_base.event_id,
                      note="边界分支：重叠峰/数据缺口/积分端点（分叉时无空白）")
    svc.import_material("blank", "control",
                        read_case_file("blank_power"), version="v1",
                        note="空白对照功率曲线")
    svc.import_material("schedule", "blank",
                        read_case_file("blank_schedule"), version="v1",
                        note="空白注射时序")
    svc.import_material("baseline", "blank",
                        read_case_file("blank_baseline"), version="v1",
                        note="空白基线独立版本化")

    # initial computations without first-injection discard: blank has only 6
    # injections vs 12 sample injections -> later injections have no blank
    # correspondence (BLANK_CORRESPONDENCE_MISSING).
    svc.compute_integrations(note="首次积分：包含首注")
    svc.compute_blank_integrations(note="空白积分")
    svc.decide_blank_rule("BLANK_MATCH_BY_ORDINAL", True,
                          note="按序号对应空白，按体积缩放")
    svc.compute_net_heats(note="净热：后6注缺空白对应")
    fit_ev1 = svc.fit_models(equiopt_tol=0.15, note="初始模型候选")

    # conflicting human decisions about the first injection.
    svc.decide_first_injection(False,
                               note="冲突决定A：保留首注（被后决定取代）")
    svc.decide_first_injection(True,
                               note="冲突决定B：舍弃首注（当前生效）")
    svc.compute_net_heats(note="净热：首注舍弃后重算")
    fit_ev2 = svc.fit_models(equiopt_tol=0.15,
                             note="舍弃首注后的模型候选")

    # concentration revision: models become stale, integrals are untouched.
    svc.import_material("concentration", "sample",
                        read_case_file("sample_concentration_v2"),
                        version="v2", note="浓度修订：蛋白 50->52 uM")
    fit_ev3 = svc.fit_models(equiopt_tol=0.15, note="浓度修订后重拟合并过期旧模型")

    # reviewer decision recorded on the review path
    svc.review("adjust", "model_fit", fit_ev3.event_id,
               adjustment={"reason": "保留全部等优候选，比较一位点/两位点/稀释"},
               note="最终复核：不强行唯一化")
    svc.review("accept", "integration_set",
               "integration", note="接受积分证据（峰重叠已在端点标注）")

    svc.import_material("power", "sample",
                        read_case_file("boundary_power"),
                        version="v1", branch="boundary",
                        note="边界功率：含数据缺口与重叠峰")
    svc.import_material("schedule", "sample",
                        read_case_file("boundary_schedule"),
                        version="v1", branch="boundary",
                        note="边界注射时序")
    svc.import_material("concentration", "sample",
                        read_case_file("boundary_concentration"),
                        version="v1", branch="boundary",
                        note="边界浓度")
    svc.import_material("conditions", "sample",
                        read_case_file("boundary_conditions"),
                        version="v1", branch="boundary",
                        note="边界条件：温度用摄氏度导入后规范化")
    svc.import_material("baseline", "sample",
                        read_case_file("boundary_baseline"),
                        version="v1", branch="boundary", note="边界基线")
    # narrow window so the 35 s-separated peaks overlap and the final 26 s
    # window clips at the series end
    svc.decide_integration_params(
        {"max_window_s": 26.0, "guard_before_next_s": 1.0,
         "return_tolerance": 6.0}, "integ-boundary-v1",
        branch="boundary", note="窄积分窗：强制重叠与端点检验")
    svc.compute_integrations(branch="boundary",
                             note="边界积分：重叠/跨缺口/短末窗")

    # local source edit that changes only the schedule (local dependency),
    # moving the first two injections even closer together
    svc.import_material("schedule", "sample",
                        read_case_file("boundary_schedule_v2"),
                        version="v2", branch="boundary",
                        note="局部源改动：仅修订注射时序")
    svc.compute_integrations(branch="boundary",
                             note="时序修订后边界积分（仅积分依赖变化）")

    # no blank material exists on this branch -> every net heat must report
    # missing correspondence rather than defaulting to zero dilution heat
    svc.decide_first_injection(True, branch="boundary", note="边界分支舍弃首注")
    svc.decide_blank_rule("BLANK_MATCH_BY_ORDINAL", True,
                          branch="boundary", note="边界分支声明空白规则")
    svc.compute_net_heats(branch="boundary",
                          note="边界净热：空白对应全部缺失")
    # deliberately constrained budget: dilution & sites still attempted; the
    # multi-parameter fits may return BUDGET_EXHAUSTED as a distinct outcome
    svc.fit_models(branch="boundary", equiopt_tol=0.12,
                   max_evals=boundary_max_evals,
                   note="边界模型：低预算下区分预算耗尽")
    svc.review("reject", "integration_set", "integration",
               branch="boundary",
               note="拒绝跨缺口积分：窗口不得跨越数据缺口")

    # ---- third branch: weak binding -> multiple equi-optimal models ---------
    svc.create_branch("equiopt", DEFAULT_BRANCH, boundary_base.event_id,
                      note="弱结合分支：两位点与稀释等优，一位点无解")
    eq_materials = [
        ("power", "sample", "equiopt_power"),
        ("schedule", "sample", "equiopt_schedule"),
        ("concentration", "sample", "equiopt_concentration"),
        ("conditions", "sample", "equiopt_conditions"),
        ("baseline", "sample", "equiopt_baseline"),
    ]
    for kind, role, key in eq_materials:
        svc.import_material(kind, role, read_case_file(key), version="v1",
                            branch="equiopt",
                            note=f"弱结合材料 {kind}/{role}")
    svc.compute_integrations(branch="equiopt", note="弱结合积分")
    svc.decide_blank_rule("BLANK_MATCH_BY_ORDINAL", True,
                          branch="equiopt", note="声明空白规则但无空白材料")
    svc.compute_net_heats(branch="equiopt",
                          note="弱结合净热：无空白对应，记录缺失")
    # with no blank net heats available, demonstrate the equi-optimum on
    # observed heats by accepting an integration-only review first; fits use
    # available net heats only when blank correspondence exists, so a
    # dedicated equiopt blank set is supplied below
    svc.import_material("blank", "control", read_case_file("equiopt_power"),
                        version="v1", branch="equiopt",
                        note="弱结合空白（缓冲液稀释，用同一时序）")
    svc.import_material("schedule", "blank",
                        read_case_file("equiopt_schedule"), version="v1",
                        branch="equiopt", note="弱结合空白时序")
    svc.compute_blank_integrations(branch="equiopt", note="弱结合空白积分")
    svc.compute_net_heats(branch="equiopt", note="弱结合净热（对应齐全）")
    fit_eq = svc.fit_models(branch="equiopt", equiopt_tol=0.15,
                            note="两个合法等优候选全部保留")
    svc.review("adjust", "model_fit", fit_eq.event_id, branch="equiopt",
               adjustment={"decision": "不唯一化：同时保留两位点与稀释候选"},
               note="无法唯一决定时保留全部合法候选")
    return svc


MANIFEST_INFO = MANIFEST
