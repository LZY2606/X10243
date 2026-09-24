"""Offline case pack: load raw domain files and replay the full narrative.

The scenario is one continuous story (import -> review -> edge combinations ->
final decision), not unrelated buttons. Forks preserve alternative decisions;
every discrepancy can be traced back to a material version, a rule version or
a decision event.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib

from .service import Workflow
from .storage import sid


def read_case(case_dir: str | pathlib.Path) -> dict:
    case_dir = pathlib.Path(case_dir)
    manifest = json.loads((case_dir / "manifest.json").read_text())

    def r(name: str) -> str:
        return (case_dir / name).read_text()

    files = {
        "power": r(manifest["materials"]["power"]),
        "baseline": r(manifest["materials"]["baseline"]),
        "blank": r(manifest["materials"]["blank"]),
        "schedule": r(manifest["materials"]["schedule"]),
        "blank_schedule": r(manifest["materials"]["blank_schedule"]),
        "concentration": r(manifest["materials"]["concentration"]),
    }
    conditions = _parse_conditions(r(manifest["materials"]["conditions"]))
    revision = r(manifest["revision"]["concentration"])
    patches_text = r(manifest["local_patch"]["power"])
    patches = _parse_patches(patches_text)
    return {"manifest": manifest, "files": files, "conditions": conditions,
            "revision_text": revision, "patches": patches, "dir": str(case_dir)}


def _parse_conditions(text: str) -> dict[str, str]:
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row]
    out = {}
    for name, value in rows[1:]:
        if name.startswith("temperature"):
            out["temperature"] = value
        elif name.startswith("cell_volume"):
            out["cell_volume"] = value
    return out


def _parse_patches(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    return [{"time_s": float(row["time_s(s)"]), "power_w": float(row["power_w(W)"])}
            for row in reader]


def _scalar_csv(kind: str, dim: str, raw_value: str) -> str:
    unit = {"temperature": "C", "cell_volume": "uL"}[kind]
    header = {"temperature": f"value_{kind}({unit})",
              "cell_volume": f"value_{kind}({unit})"}[kind]
    return f"name,{header}\n{kind},{raw_value}\n"


def seed_case(wf: Workflow, case_dir: str | pathlib.Path, *,
              project_id: str = "demo-itc-case",
              name: str = "滴定热谱：ITC 注射积分、稀释热与绑定模型审议") -> dict:
    data = read_case(case_dir)
    store = wf.store
    store.create_project(project_id, name, seed={"case_dir": data["dir"],
                                                 "domain": data["manifest"]["domain"]})

    def imp(kind: str, text: str, origin: str):
        return wf.import_material(project_id, kind, text, origin)

    imp("power", data["files"]["power"], "power.csv")
    imp("schedule", data["files"]["schedule"], "schedule.csv")
    imp("concentration", data["files"]["concentration"], "concentration.csv")
    imp("temperature", _scalar_csv("temperature", "temperature",
                                    data["conditions"]["temperature"]),
        "conditions.csv#temperature_C")
    imp("cell_volume", _scalar_csv("cell_volume", "volume",
                                    data["conditions"]["cell_volume"]),
        "conditions.csv#cell_volume_uL")
    imp("baseline", data["files"]["baseline"], "baseline.csv")
    imp("blank", data["files"]["blank"], "blank.csv")
    imp("blank_schedule", data["files"]["blank_schedule"], "blank_schedule.csv")

    post = data["manifest"]["integration_post_window_s"]
    interval = data["manifest"]["injection_interval_s"]
    # first integration: overlapping peaks + gap + missing last blank
    wf.set_blank_rule(project_id, "by_index", branch="main",
                      rationale="空白运行与样品运行逐针对应（同体积、同时序）",
                      basis=["blank_schedule.csv", "schedule.csv"])
    integ = wf.integrate(project_id, post_s=post, max_gap_s=5.0,
                         blank_rule="by_index", note="基线版本 v1 + 初始浓度")
    wf.discard_shots(project_id, [1], rationale="首注为不完整预注射，积分保留但不进拟合",
                     basis=[integ.events[0]["event_id"]])
    fit = wf.fit(project_id, note="一位点/两位点/稀释候选同时保留")

    # parameter revision: concentration corrected -> fit stale, integrals intact
    pre_rev_conc = _material_head(wf, project_id, "concentration")
    rev = wf.import_material(project_id, "concentration", data["revision_text"],
                             "revision/concentration_v2.csv",
                             note="注射器浓度修订：仅使绑定模型过期")
    refit = wf.fit(project_id, note="浓度修订后的重新拟合（历史拟合保留为过期）")

    # conflicting human decisions are both retained
    accept = wf.manual_decision(
        project_id, "accept_best_model", "accepted",
        rationale="等优候选全部保留，接受一位点作为报告模型（参数更少）",
        basis=[fit.events[0]["event_id"], refit.events[0]["event_id"]])
    wf.manual_decision(
        project_id, "accept_best_model", "rejected",
        rationale="反对意见：两位点候选残差同样优，需要更多滴定点",
        basis=[fit.events[0]["event_id"]],
        conflicts_with=accept.events[0]["event_id"])

    # explicit endpoint decision for the gap-terminated 7th window
    wf.set_endpoints(project_id, 7, 6 * interval - 2.0, 6 * interval + 28.0,
                     rationale="第 7 针窗口在采集缺口前终止，积分窗不跨越缺口",
                     basis=[integ.events[0]["event_id"]])
    integ2 = wf.integrate(project_id, post_s=post, blank_rule="by_index",
                          note="采用第 7 针显式端点后重积")

    # ---- branch 1: local-only source change ----
    fork_after = rev.events[0]["event_id"]
    local_branch = wf.fork(project_id, "局部功率补丁分支",
                           from_event_id=fork_after)
    wf.patch_power(project_id, data["patches"], origin="patch/power_local.csv",
                   branch=local_branch,
                   rationale="仅修正第 4 针窗口内若干采样点，其他材料不变")
    wf.integrate(project_id, branch=local_branch, post_s=post,
                 blank_rule="by_index", note="局部补丁后的积分")

    # ---- branch 2: alternative dilution rule decision ----
    alt_branch = wf.fork(project_id, "体积比例空白规则分支",
                         from_event_id=integ.events[0]["event_id"])
    wf.set_blank_rule(project_id, "volume_proportional", branch=alt_branch,
                      rationale="改用体积比例对应规则，保留原始逐针规则分支")
    wf.integrate(project_id, branch=alt_branch, post_s=post,
                 blank_rule="volume_proportional", note="替代稀释热对应规则")

    return {"project_id": project_id, "data": data,
            "events": {"integration": integ.events[0]["event_id"],
                       "fit": fit.events[0]["event_id"],
                       "revision": rev.events[0]["event_id"],
                       "refit": refit.events[0]["event_id"],
                       "accept": accept.events[0]["event_id"],
                       "integration2": integ2.events[0]["event_id"]},
            "branches": {"main": "main", "local": local_branch,
                         "alt_blank": alt_branch}}


def _material_head(wf: Workflow, project_id: str, kind: str) -> str | None:
    state = wf._head_state(project_id, "main")
    return Workflow._head_event_id(state, kind)
