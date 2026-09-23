"""Replay the offline narrative's boundary combinations.

The initial import registers every material (power / injection volume+time /
concentrations / temperature / cell volume / separately-versioned baseline and
blank).  Subsequent steps deliberately trigger:

  overlapping peaks, first-injection discard, missing blank correspondence,
  concentration revision, multiple equal-best models, data gaps, integration
  endpoints - plus one local-only source patch and one conflicting human
  decision on a forked branch.

Every action is an event; accepted and rejected candidates are both retained.
"""

from __future__ import annotations

import os
from typing import Optional

from .errors import FormatCorrupted
from .service import ProjectService

CASE_ROOT = os.path.join("cases", "titration-heat-review", "v1", "materials")
SERIES = {
    "main": ("sample", "Main titration"),
    "blank_dilution": ("blank", "Dilution blank"),
    "overlap": ("sample", "Overlapping peaks"),
    "gap": ("sample", "Data gap / endpoint"),
    "tie": ("sample", "Near-equivalent models"),
}
KINDS = ["power", "injections", "concentrations", "conditions", "baseline"]
KIND_FILE = {"power": "power.csv", "schedule": "injections.csv",
             "concentrations": "concentrations.csv", "conditions": "conditions.csv",
             "baseline": "baseline.csv"}
KIND_FOR_FILE = {v: k for k, v in KIND_FILE.items()}


def _read(series, fname):
    path = os.path.join(CASE_ROOT, series, fname)
    with open(path, encoding="utf-8") as f:
        return f.read()


def initial_import(svc: ProjectService, project_id: str = "titration-heat-review"):
    svc.create_project(project_id, "滴定热谱：ITC 注射积分、稀释热与绑定模型审议")
    for sid, (role, label) in SERIES.items():
        svc.add_series(project_id, sid, role, label)
        for fname in ["power.csv", "injections.csv", "concentrations.csv",
                      "conditions.csv", "baseline.csv"]:
            kind = KIND_FOR_FILE[fname]
            content = _read(sid, fname)
            svc.add_material(project_id, sid, kind, content,
                             path=f"v1/materials/{sid}/{fname}")
    svc.link_blank(project_id, "main", "blank_dilution", {"mode": "ordinal"})
    return project_id


def play(svc: ProjectService, project_id: str = "titration-heat-review"):
    initial_import(svc, project_id)
    events = {}

    # --- main branch narrative -------------------------------------------
    # First injection is systematically noisy: reviewer discards it.
    events["first_discard"] = svc.add_decision(
        project_id, "main", "first_injection_discard", {"count": 1},
        note="首注舍弃：首滴体积/混合误差，原积分保留、仅排除建模").get("id")

    # Reviewer adopts the one-site candidate (two-site and dilution stay on
    # record as un-adopted candidates).
    review = svc.review(project_id, "main")
    one = next(f for f in review["series"]["main"]["models"]["fits"]
               if f["model"] == "one_site")
    events["accept_one_site"] = svc.add_decision(
        project_id, "main", "model_accepted",
        {"model": "one_site", "fit_id": one["id"],
         "conc_digest": review["series"]["main"]["material_digests"]["concentrations"],
         "params_snapshot": one["params"]},
        note="接受一位点模型；两位点与稀释假设保留为未采纳候选").get("id")

    # Concentration revision (ligand stock re-titrated). This makes the
    # accepted model STALE but must not alter the original power integrals.
    revised = _read("main", "concentrations.csv").replace("# version: 1", "# version: 2")
    revised = revised.replace("ligand,1000,µM", "ligand,1080,µM")
    events["concentration_revision"] = svc.add_material(
        project_id, "main", "concentrations", revised,
        path="v1/materials/main/concentrations.csv#rev2")["event"]["id"]

    # Reviewer re-integrates with a wider window and compares again.
    events["window_adjust"] = svc.add_decision(
        project_id, "main", "integration_window",
        {"pre_s": 12.0, "post_s": 250.0},
        note="参数调整：加宽积分窗复核端点").get("id")

    # --- overlap branch ---------------------------------------------------
    svc.fork(project_id, "overlap", "main", reason="检验重叠峰积分窗")
    events["overlap_window"] = svc.add_decision(
        project_id, "overlap", "integration_window",
        {"pre_s": 10.0, "post_s": 240.0}, branch="overlap",
        note="注射后峰可重叠：窗在下一针前裁剪，不强行切开").get("id")

    # --- gap branch: data gap + endpoint + missing blank correspondence ---
    svc.fork(project_id, "gap", "main", reason="检验数据缺口、端点与空白对应缺失")
    # Link gap series to the blank using a volume rule it cannot satisfy for
    # one injection (blank is 5 µL x20 while gap's schedule differs) ->
    # per-injection BLANK_CORRESPONDENCE_MISSING diagnostics.
    # gap series shares blank but schedule length differs -> some ordinal
    # counterparts are missing, reported per injection (never zero-filled).
    svc.store.append(project_id, "gap", "blank_linked",
                     {"sample_series_id": "gap",
                      "blank_series_id": "blank_dilution",
                      "rule": {"mode": "ordinal"},
                      "rationale_codes": ["BLANK_CORRESPONDENCE_EXPLICIT",
                                          "BLANK_CORRESPONDENCE_MISSING_POSSIBLE"]})
    events["gap_first_discard"] = svc.add_decision(
        project_id, "gap", "first_injection_discard", {"count": 1},
        branch="gap").get("id")

    # --- tie branch: keep all legal candidates (no forced unique choice) ---
    svc.fork(project_id, "tie", "main", reason="保留多个等优模型候选")
    events["tie_note"] = svc.store.append(
        project_id, "tie", "note",
        {"text": "等价输入稳定排序；无法唯一决定时保留全部合法候选"}).get("id")

    # --- conflict branch: mutually conflicting human decisions ------------
    svc.fork(project_id, "conflict", "main", reason="互相冲突的人工决定")
    rev = svc.review(project_id, "conflict")
    tie_models = [f["model"] for f in rev["series"]["main"]["models"]["fits"]]
    for m in ("dilution", "two_site"):
        if m in tie_models:
            fit = next(f for f in rev["series"]["main"]["models"]["fits"]
                       if f["model"] == m)
            svc.add_decision(
                project_id, "conflict", "model_accepted",
                {"model": m, "fit_id": fit["id"], "params_snapshot": fit["params"]},
                branch="conflict", note=f"冲突决定：{m}")
    events["conflict"] = True
    return events


def local_source_patch(svc: ProjectService, project_id: str, branch: str = "main"):
    """A source change that affects only local dependencies: patch the blank
    power curve slightly. Integrals of the blank change; sample binding models
    (derived from sample power) are unaffected through their own material ids.
    """
    text = _read("blank_dilution", "power.csv")
    patched = text.replace("# power_unit: µcal/s",
                           "# power_unit: µcal/s\n# note: local patch v1.1")
    return svc.add_material(project_id, "blank_dilution", "power", patched,
                            branch=branch,
                            path="v1/materials/blank_dilution/power.csv#patch1")
