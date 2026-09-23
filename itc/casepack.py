"""离线领域案例包加载与叙事重放。

从 casepack/ 原始材料出发，按实验叙事依次产生事件：
  初始导入(v1 全套) -> 积分(重叠/首注/缺口/空白缺失) -> 模型候选
  -> 浓度修订(v2, 模型过期) -> 空白修订(v2, 仅局部净热改变) -> 重积分重拟合
  -> 互斥人工决定(main 与 review 分支)
所有步骤可重放、可从最终状态反查材料版本、规则版本与决定。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from . import binding, ingest, integrate
from .errors import Budget, Outcome
from .project import DEFAULT_BRANCH, Project

CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "casepack")

FILES = {
    "power": ("power_curve_v1.csv", ingest.parse_power),
    "schedule": ("injections_v1.csv", ingest.parse_schedule),
    "baseline": ("baseline_v1.csv", ingest.parse_baseline),
    "blank": ("blank_curve_v1.csv", lambda t: ingest.parse_power(t, "blank", "blank")),
    "concentrations": ("concentrations_v1.json", ingest.parse_concentrations),
    "temperature": ("temperature_v1.json", ingest.parse_temperature),
    "cell_volume": ("cell_volume_v1.json", ingest.parse_cell_volume),
}
FILES_V2 = {
    "blank": ("blank_curve_v2.csv",
              lambda t: ingest.parse_power(t, "blank", "blank")),
    "concentrations": ("concentrations_v2.json", ingest.parse_concentrations),
}

PARAMS = integrate.IntegrationParams(
    max_window_s=240.0, gap_factor=3.0, overlap_decay_s=60.0,
    first_injection_min_uL=1.0, dilution_rule="per_injection_blank")
BUDGET = Budget(max_ops=20000)


def read_case(version: dict[str, str] | None = None) -> dict[str, Outcome]:
    version = version or {}
    out: dict[str, Outcome] = {}
    for kind, (fname, parser) in FILES.items():
        fname = version.get(kind, fname)
        with open(os.path.join(CASE_DIR, fname), encoding="utf-8") as fh:
            text = fh.read()
        out[kind] = (text, parser(text))
    return out


def seed(project: Project, verbose: bool = False) -> dict[str, Any]:
    """把完整叙事重放进一个空 project，返回每步的可读句柄。"""
    log: dict[str, Any] = {"steps": []}

    def note(step: str, **kw: Any) -> None:
        log["steps"].append({"step": step, **kw})
        if verbose:
            print(step, kw)

    parsed = read_case()
    # 1) 初始导入：七种材料分开版本化
    for kind, (_text, outcome) in parsed.items():
        if not outcome.ok:
            raise RuntimeError(f"案例材料 {kind} 导入失败: {outcome.status}")
        _, ver, seq = project.import_material(kind, outcome.value, _text)
        note("import", kind=kind, version=ver, seq=seq,
             diagnostics=[d.to_dict() for d in outcome.diagnostics])

    # 2) 逐注射积分（重叠/首注/缺口/空白对应缺失在此全部触发）
    r1 = project.run_integration(PARAMS, BUDGET)
    if not r1.ok:
        raise RuntimeError(f"首次积分失败: {r1.status}")
    note("integrate_v1", result_id=r1.value["result_id"],
         diagnostics=[d.to_dict() for d in r1.diagnostics],
         rule_events=r1.value["payload"]["rule_events"])

    # 3) 绑定模型候选：一位点 / 两位点 / 稀释
    f1 = project.fit_models(["one_site", "two_site", "dilution"], BUDGET)
    if not f1.ok:
        raise RuntimeError(f"首次拟合失败: {f1.status}")
    note("fit_v1", result_id=f1.value["result_id"],
         models=_fit_summary(f1.value["fits"]),
         diagnostics=[d.to_dict() for d in f1.diagnostics])

    # 4) 参数修订：浓度 v2 —— 模型过期，原积分不变
    with open(os.path.join(CASE_DIR, FILES_V2["concentrations"][0]),
              encoding="utf-8") as fh:
        conc_text = fh.read()
    conc2 = ingest.parse_concentrations(conc_text)
    _, cver, cseq = project.revise_material(
        "concentrations", conc2.value, conc_text,
        reason="复称后滴定剂浓度 1.80 -> 1.76 mM")
    note("revise_concentrations", version=cver, seq=cseq,
         effect="models_stale_integrals_unchanged")

    # 5) 局部源数据改动：空白 v2 覆盖全程，仅局部净热改变
    with open(os.path.join(CASE_DIR, FILES_V2["blank"][0]),
              encoding="utf-8") as fh:
        blank_text = fh.read()
    blank2 = ingest.parse_power(blank_text, "blank", "blank")
    _, bver, bseq = project.revise_material(
        "blank", blank2.value, blank_text,
        reason="空白对照补齐至 2400 s，恢复稀释热对应")
    note("revise_blank", version=bver, seq=bseq,
         effect="only_missing_correspondence_injections_change")

    # 6) 重积分（幂等身份：未变注射输入不变）
    r2 = project.run_integration(PARAMS, BUDGET)
    note("integrate_v2", result_id=r2.value["result_id"],
         changed_injections=_changed_net_heats(
             r1.value["payload"], r2.value["payload"]))

    # 7) 重新拟合（基于浓度 v2 与积分 v2）
    f2 = project.fit_models(["one_site", "two_site", "dilution"], BUDGET)
    note("fit_v2", result_id=f2.value["result_id"],
         models=_fit_summary(f2.value["fits"]))

    # 8) 互斥人工决定：main 接受一位点；review 分支否定一位点/接受两位点
    onesite_id = f"model:one_site@{f2.value['result_id']}"
    d1 = project.decide(onesite_id, "accept",
                        "一位点 χ² 最优且参数物理合理", reviewer="pi-a")
    note("decide_main", **d1, target=onesite_id)
    d2 = project.decide(onesite_id, "reject",
                        "坚持两位点解释：存在次级弱位点", reviewer="pi-b")
    note("decide_conflict", **d2, target=onesite_id)
    twosite_id = f"model:two_site@{f2.value['result_id']}"
    d3 = project.decide(twosite_id, "accept", "在冲突分支上接受两位点",
                        branch=d2["branch"], reviewer="pi-b")
    note("decide_accept_two_site", **d3, target=twosite_id)

    log["ids"] = {"integration_v1": r1.value["result_id"],
                  "fit_v1": f1.value["result_id"],
                  "integration_v2": r2.value["result_id"],
                  "fit_v2": f2.value["result_id"],
                  "review_branch": d2["branch"]}
    return log


def _fit_summary(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for f in fits:
        if f.get("kind") != "fit":
            out.append({"model": f.get("model"), "status": f.get("status")})
            continue
        out.append({"model": f["model"], "chi2": f["chi2"],
                    "equivalent": f.get("equivalent"),
                    "equivalent_group": f.get("equivalent_group"),
                    "params": f["params"]})
    return out


def _changed_net_heats(p1: dict[str, Any], p2: dict[str, Any]
                       ) -> list[dict[str, Any]]:
    a = {r["injection_id"]: r.get("heat_net_uJ")
         for r in p1["injections"]}
    changes = []
    for r in p2["injections"]:
        iid = r["injection_id"]
        if a.get(iid) != r.get("heat_net_uJ"):
            changes.append({"injection_id": iid,
                            "heat_net_v1_uJ": a.get(iid),
                            "heat_net_v2_uJ": r.get("heat_net_uJ")})
    return changes
