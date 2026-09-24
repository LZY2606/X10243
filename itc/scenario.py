"""离线案例包的叙事重放脚本（确定性）。

主分支：初始导入(6 类材料分开版本化) -> 积分(重叠/首注/缺口/端点) ->
空白对应缺失 -> 一位点/多位点/稀释 -> 接受 -> 浓度修订(模型过期、积分不变) ->
局部基线改动 -> 冲突人工决定。
分支 B：等优模型全部保留 + 预算耗尽是第三种结果。
"""
from __future__ import annotations

from typing import Any

from .casepack import CASE_TITLE, build_alpha_bundle, build_tie_bundle
from .errors import BudgetExhausted, SemanticInvalid
from .materials import BASELINE, CONCENTRATIONS
from .rules import BlankRules, IntegrationRules, ModelRules
from .service import ProjectService
from .store import Store

RULES_INTEGRATION_V1 = IntegrationRules(version="v1")
RULES_INTEGRATION_PARTITION = IntegrationRules(version="v2-partition", overlap_policy="partition")
RULES_BLANK_V1 = BlankRules(version="v1")
RULES_MODEL_V1 = ModelRules(version="v1")
RULES_MODEL_TIGHT_BUDGET = ModelRules(version="v1", eval_budget=30)


def run_alpha(store: Store, project_id: str = "alpha") -> dict[str, Any]:
    svc = ProjectService(store, project_id=project_id)
    trace: list[dict[str, Any]] = []

    # 0) 初始导入：6 类材料分开版本化，原始单位保留
    mats = svc.import_bundle(build_alpha_bundle(variant="v1"), note="初始导入：六类材料分开版本化")
    trace.append({"step": "import", "materials": [m.id for m in mats],
                  "tokens": {m.kind: m.tokens for m in mats if m.tokens}})

    # 1) 滴定积分：重叠峰 / 首注舍弃 / 缺口不跨越 / 端点检验同时经过
    integ = svc.run_integration(RULES_INTEGRATION_V1,
                                note="名义窗 120s；重叠峰独立保留；首注舍弃；窗不跨缺口")
    trace.append({"step": "titration_integration", "result_id": integ["id"]})

    # 2) 空白积分（空白独立曲线、独立基线）
    blank = svc.run_integration(IntegrationRules(version="v1-blank"), is_blank=True,
                                note="空白曲线单独积分")
    trace.append({"step": "blank_integration", "result_id": blank["id"]})

    # 3) 稀释热扣除：same_index 对应 -> 末尾两注对应缺失（显式诊断）
    corrected = svc.correct_blank(blank["id"], RULES_BLANK_V1,
                                  note="按注射序号对应；缺失不静默扣除")
    missing = [r["index"] for r in corrected["injections"]
               if any(d["code"] == "blank_correspondence_missing" for d in r["diagnostics"])]
    trace.append({"step": "blank_correction", "result_id": corrected["id"],
                  "missing_correspondence": missing})

    # 4) 绑定模型候选：一位点 / 两位点 / 稀释
    fit = svc.fit(RULES_MODEL_V1, corrected["id"], note="一位点、两位点与稀释假设比较")
    trace.append({"step": "fit", "result_id": fit["id"],
                  "selected": [c["model"] for c in fit["selected"]],
                  "n_candidates": len(fit["candidates"])})

    # 5) 接受一位点（即便两位点可能并列，审阅者显式选择）— 采纳结果
    svc.decide("one_site", "accepted", fit["id"],
               rationale="结合 K≈1e7 M^-1 与等价位点化学解释，采纳一位点",
               params={"model": "one_site"})
    trace.append({"step": "accept_one_site"})

    # 6) 参数调整：换 partition 重叠策略重算一次（证明重叠可按规则处理）
    integ_part = svc.run_integration(RULES_INTEGRATION_PARTITION,
                                     note="重叠峰 partition 策略：窗在下一注射处截断")
    blank_part = svc.run_integration(IntegrationRules(version="v2-partition-blank",
                                                      overlap_policy="partition"),
                                     is_blank=True)
    corr_part = svc.correct_blank(blank_part["id"], BlankRules(version="v2-partition"),
                                  note="partition 策略下的稀释校正")
    fit_part = svc.fit(ModelRules(version="v2"), corr_part["id"],
                       note="partition 策略下模型复核")
    trace.append({"step": "partition_review", "fit_id": fit_part["id"]})

    # 7) 浓度修订（源数据）：模型过期但原积分不变
    svc.import_bundle(build_alpha_bundle(variant="v2_concentration"), revision=2,
                      note="注射器浓度 240->210 uM 修订")
    view = svc.current_view()
    trace.append({"step": "concentration_revised",
                  "fit_stale": view["fit"]["stale"],
                  "fit_stale_reason": view["fit"]["stale_reason"],
                  "integration_id_unchanged": view["integration"]["id"] == integ["id"]})

    # 8) 局部基线改动：只影响落入局部区间的注射
    svc.import_bundle(build_alpha_bundle(variant="v2_baseline_local"), revision=3,
                      note="第 11 注附近基线局部修订")
    view2 = svc.current_view()
    trace.append({"step": "baseline_local_revised",
                  "latest_baseline_revision": next(
                      m["revision"] for m in view2["materials"] if m["kind"] == BASELINE)})

    # 9) 互相冲突的人工决定：在主分支接受一位点；对同一拟合在 tie 分支保留全部候选
    svc.decide("mainline_one_site", "accepted", fit["id"],
               rationale="主线决定：一位点", conflict_with="decision:tie_branch:keep_all")
    trace.append({"step": "conflicting_decision_main"})

    return {"title": CASE_TITLE, "project_id": project_id, "trace": trace}


def run_tie_branch(store: Store, project_id: str = "alpha", branch: str = "tie_branch") -> None:
    svc = ProjectService(store, project_id=project_id)
    svc.create_branch(branch, from_branch="main")
    # 在分支导入“无绑定信号”数据集，验证多个等优模型全部保留
    mats = svc.import_bundle(build_tie_bundle(), source_ref="tie-control",
                             branch=branch, note="等优对照导入（独立来源）")
    integ = svc.run_integration(IntegrationRules(version="v1"), branch=branch,
                                note="对照积分")
    blank = svc.run_integration(IntegrationRules(version="v1-blank"), branch=branch,
                                is_blank=True)
    corrected = svc.correct_blank(blank["id"], BlankRules(version="v1"), branch=branch)
    fit = svc.fit(ModelRules(version="v1", tie_delta=4.0), corrected["id"], branch=branch,
                  note="高噪声无信号：一位点与稀释等优，全部保留")
    tied = [c["model"] for c in fit["selected"]]

    # 冲突决定：分支上拒绝单选，保留全部合法候选
    svc.decide("tie_keep_all", "conflict", fit["id"], branch=branch,
               rationale="无法唯一决定，保留全部等优候选，拒绝单选",
               conflict_with="decision:mainline_one_site", params={"tied": tied})

    # 预算耗尽：第三种结果（区别于格式损坏与语义无解）
    budget_event = None
    try:
        svc.fit(RULES_MODEL_TIGHT_BUDGET, corrected["id"], branch=branch,
                note="故意压低评估预算")
    except BudgetExhausted as exc:
        budget_event = {"kind": "budget_exhausted", "budget": exc.budget, "used": exc.used}
    return {"tied_models": tied, "budget_event": budget_event}


def bootstrap(store: Store, project_id: str = "alpha") -> dict[str, Any]:
    """空库引导：完整重放案例包，返回摘要。"""
    main = run_alpha(store, project_id)
    tie = run_tie_branch(store, project_id)
    main["tie_branch"] = tie
    return main
