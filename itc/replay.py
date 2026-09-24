"""Replay the offline ITC case pack into a fully evidenced project state.

The replay is itself an executable specification: every required situation is
traversed with an explicit, machine-readable event and affected objects:

overlapping injection peaks (3&4), first injection discard (ordinal 1),
missing blank counterpart (ordinal 6 after blank first-shot discard),
concentration revision (0.40 -> 0.32 mM; models stale, integrals untouched),
multiple equipotent models (flat dilution branch), data gap + integration
endpoint (ordinal 5, gap 355-372 s), local-only power fragment patch
(490-560 s), and conflicting human decisions on the same branch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import casebuild
from .engine import (
    ProjectState, apply_event, branch_at, make_decision_event,
    make_material_event, new_project,
)
from .errors import FormatCorrupt, Diagnostic
from .materials import parse_power

TITRATION = "titration-2026-09-24"
BLANK = "blank-buffer-2026-09-24"
FLAT = "flatrun-2026-09-24"


@dataclass
class ReplayReport:
    state: ProjectState
    steps: list[dict]


def load_pack(case_dir: str | Path) -> dict:
    case_dir = Path(case_dir)
    pack = {}
    for path in sorted((case_dir / "materials").rglob("*.json")):
        rel = path.relative_to(case_dir / "materials")
        pack[str(rel)] = json.loads(path.read_text(encoding="utf-8"))
    pack["__replay__"] = json.loads(
        (case_dir / "replay" / "replay.json").read_text(encoding="utf-8"))
    return pack


def _material(state, kind, raw, *, run, source, branch="main", stream=None,
              label="", timestamp=""):
    event, rev = make_material_event(
        state, kind, raw, run_id=run, source=source, branch=branch,
        stream=stream, label=label, timestamp=timestamp)
    produced = apply_event(state, event)
    return event, rev, produced


def _decision(state, decision, *, branch="main", label="", timestamp="",
              conflict_with=None):
    event = make_decision_event(
        state, decision, branch=branch, label=label, timestamp=timestamp,
        conflict_with=conflict_with)
    apply_event(state, event)
    return event


def build_project(case_dir: str | Path, *, budget: int = 65536) -> ReplayReport:
    pack = load_pack(case_dir)
    state = new_project("demo-itc-case",
                        "滴定热谱：ITC 注射积分、稀释热与绑定模型审议")
    steps: list[dict] = []

    def note(code, label, evidence, branch="main"):
        steps.append({"code": code, "label": label, "branch": branch,
                      "evidence": evidence})

    m = pack
    # --- initial versioned import (all seven material kinds) ------------- #
    order = [
        ("power", "titration/power.v1.json", TITRATION),
        ("injections", "titration/injections.v1.json", TITRATION),
        ("temperature", "titration/temperature.v1.json", TITRATION),
        ("cell_volume", "titration/cell_volume.v1.json", TITRATION),
        ("concentration", "titration/concentration.v1.json", TITRATION),
        ("baseline", "titration/baseline.v1.json", TITRATION),
        ("blank_link", "titration/blank_link.v1.json", TITRATION),
        ("power", "blank/power.v1.json", BLANK),
        ("injections", "blank/injections.v1.json", BLANK),
        ("baseline", "blank/baseline.v1.json", BLANK),
    ]
    for kind, rel, run in order:
        ev, rev, produced = _material(
            state, kind, m[f"{rel}"], run=run,
            source=f"case/materials/{rel}",
            label=f"import {kind} ({run})", timestamp="2026-09-24T09:00:00+09:00")
        note("import", f"import {kind} {run}",
             {"event_id": ev.event_id, "revision_id": rev.revision_id})

    note("initial_review",
         "核对功率采样/注射时序/浓度/温度/池体积/基线/空白曲线（分开版本化）",
         {"materials": {k: v.ref().revision_id for k, v in
                        state.heads_for(TITRATION).items()}})

    # initial computed evidence: overlaps, first discard, gap, missing blank
    note("overlap", "注射 3 与 4 注射后峰重叠；积分窗止于下一注射时刻",
         {"ordinals": [3, 4]})
    note("first_discard", "首注 ordinal=1 舍弃（保留积分证据，不进模型）",
         {"ordinal": 1})
    note("gap_endpoint", "ordinal=5 窗口遇 355-372s 数据缺口，端点取 before_gap",
         {"ordinal": 5, "gap_s": [355.0, 372.0]})
    note("blank_missing", "空白对应规则丢弃空白白首注后，ordinal=6 缺失对应",
         {"ordinal": 6, "rule": "ordinal+discard_first"})

    # --- human endpoint decision for the truncated window ----------------- #
    _decision(state, {
        "action": "endpoint_decision", "run_id": TITRATION,
        "ordinal": 5, "window_end_s": 355.0,
        "choice": "before_gap",
        "reason": "积分窗不跨越数据缺口；缺口后峰尾不计入"},
        label="确认 ordinal=5 缺口前端点", timestamp="2026-09-24T09:20:00+09:00")
    note("decision_endpoint", "确认缺口前端点 before_gap", {"ordinal": 5})

    # --- parameter revision: syringe concentration 0.40 -> 0.32 mM -------- #
    ev, rev, _ = _material(
        state, "concentration", m["titration/concentration.v2.json"],
        run=TITRATION, source="case/materials/titration/concentration.v2.json",
        label="浓度修订 0.40→0.32 mM", timestamp="2026-09-24T09:30:00+09:00")
    note("concentration_revision",
         "配液复核：syringe 0.40→0.32 mM；模型过期，原积分不变",
         {"event_id": ev.event_id, "revision_id": rev.revision_id})

    # --- local-only source change: power fragment patch ------------------- #
    v2_power = _merge_power_patch(m["titration/power.v1.json"],
                                  m["power.patch.490-560.json"])
    ev, rev, produced = _material(
        state, "power", v2_power, run=TITRATION,
        source="case/materials/power.patch.490-560.json",
        label="局部功率片段重采 490-560s",
        timestamp="2026-09-24T09:40:00+09:00")
    note("local_patch", "只改动 490-560s 功率片段：局部依赖变化",
         {"event_id": ev.event_id, "revision_id": rev.revision_id,
          "affected_ordinals": [7, 8]})

    # --- conflicting human decisions (both retained) ---------------------- #
    _decision(state, {
        "action": "accept_model", "run_id": TITRATION,
        "model": "one_site",
        "reason": "单一位点 SSR 最低且参数可辨识"},
        label="接受一位点模型", timestamp="2026-09-24T09:50:00+09:00")
    conflict = _decision(state, {
        "action": "reject_model", "run_id": TITRATION,
        "model": "one_site",
        "reason": "审阅者认为 ordinal=5 截断引入系统偏差，拒绝同一位点模型"},
        label="冲突：拒绝一位点模型", timestamp="2026-09-24T09:55:00+09:00",
        conflict_with=None)
    note("conflicting_decisions",
         "互相冲突的人工决定均保留：接受与拒绝同一位点模型",
         {"model": "one_site", "events": ["accept", "reject"]})

    # parameter adjustment decision -> new event, recompute
    _decision(state, {
        "action": "parameter_override", "run_id": TITRATION,
        "model": "one_site",
        "params": {"N": 1.05},
        "reason": "结合独立分析将位点数上调至 1.05"},
        label="参数调整 N=1.05", timestamp="2026-09-24T10:00:00+09:00")

    # ------------------------------------------------------------------ #
    # Second branch: flat dilution-dominated run -> equipotent candidates
    # ------------------------------------------------------------------ #
    branch_at(state, "main", "boundary-flat",
              at_event=state.branches["main"][0])
    # bring the boundary branch through the initial import prefix quickly:
    # simplest replay is to rebuild its own stream on the branch starting at
    # the same root (flat power + same injections + blank link without shift)
    flat_power = casebuild._samples_payload(casebuild.make_flat_trace(),
                                           time_ref="t=0 at acquisition start")
    _material(state, "power", flat_power, run=FLAT, source="case://flatrun",
              branch="boundary-flat", label="平坦滴定功率（稀释主导）",
              timestamp="2026-09-24T10:10:00+09:00")
    _material(state, "injections",
              casebuild.injections_payload(casebuild.INJ_TIMES_S,
                                           casebuild.INJ_VOLUMES_UL),
              run=FLAT, source="case://flatrun", branch="boundary-flat",
              label="平坦滴定注射时序", timestamp="2026-09-24T10:11:00+09:00")
    _material(state, "baseline", casebuild.baseline_payload(1.0e-6),
              run=FLAT, source="case://flatrun", branch="boundary-flat",
              label="平坦滴定基线", timestamp="2026-09-24T10:12:00+09:00")
    _material(state, "temperature", casebuild.temperature_payload(25.0),
              run=FLAT, source="case://flatrun", branch="boundary-flat",
              label="平坦滴定温度", timestamp="2026-09-24T10:13:00+09:00")
    _material(state, "cell_volume", casebuild.cell_volume_payload(0.2034),
              run=FLAT, source="case://flatrun", branch="boundary-flat",
              label="平坦滴定池体积", timestamp="2026-09-24T10:14:00+09:00")
    _material(state, "concentration",
              casebuild.concentration_payload(0.32, 20.0), run=FLAT,
              source="case://flatrun", branch="boundary-flat",
              label="平坦滴定浓度", timestamp="2026-09-24T10:15:00+09:00")
    _material(state, "blank_link",
              casebuild.blank_link_payload(BLANK, rule="ordinal",
                                           discard_first=False,
                                           note="平坦滴定直接按序数对应"),
              run=FLAT, source="case://flatrun", branch="boundary-flat",
              label="平坦滴定空白对应（不丢弃首注）",
              timestamp="2026-09-24T10:16:00+09:00")
    note("equivalent_models",
         "平坦/稀释主导数据：稀释假设与绑定模型等优，保留全部合法候选",
         {"branch": "boundary-flat", "run": FLAT}, branch="boundary-flat")
    return ReplayReport(state, steps)


def _merge_power_patch(base_raw: dict, patch_raw: dict) -> dict:
    """Merge a local fragment patch into v1 power sampling (produces v2)."""
    lo, hi = patch_raw["replaces_interval_s"]
    kept = [row for row in base_raw["samples"]
            if not (float(row["t"]["value"]) >= lo
                    and float(row["t"]["value"]) <= hi)]
    merged = sorted(kept + patch_raw["samples"],
                    key=lambda r: float(r["t"]["value"]))
    return {"time_reference": base_raw.get(
        "time_reference", "t=0 at acquisition start"),
        "samples": merged, "gaps": base_raw.get("gaps", [])}
