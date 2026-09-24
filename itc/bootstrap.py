"""Load the bundled offline case and replay its review narrative.

The replay *actually passes through* every required situation (overlapping
peaks, first-injection discard, missing blank correspondence, concentration
revision, tied models, data gap + integration endpoints) and leaves behind
the input fragment, judgement state and affected object for each.
"""

from __future__ import annotations

import json
import os

from .materials import (
    parse_baseline,
    parse_blank,
    parse_conditions,
    parse_power,
    parse_schedule,
)
from .rules import RULE_KIND_BLANK, RULE_KIND_INTEGRATION, RULE_KIND_MODEL


def case_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "case")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_case_files(root=None, revision=None, schedule_revision=None):
    root = root or case_root()
    def p(*parts):
        return os.path.join(root, *parts)
    power, d1 = parse_power(_read(p("v1", "power.csv")), origin_label="v1/power.csv")
    sched_path = schedule_revision or p("v1", "schedule.csv")
    schedule, d2 = parse_schedule(_read(sched_path), origin_label=os.path.relpath(sched_path, root))
    cond_path = revision or p("v1", "conditions.json")
    conditions, d3 = parse_conditions(_read(cond_path), origin_label=os.path.relpath(cond_path, root))
    baseline, d4 = parse_baseline(_read(p("v1", "baseline.csv")), origin_label="v1/baseline.csv")
    blank, d5 = parse_blank(_read(p("v1", "blank.csv")), origin_label="v1/blank.csv")
    diagnostics = d1 + d2 + d3 + d4 + d5
    return {
        "power": power, "schedule": schedule, "conditions": conditions,
        "baseline": baseline, "blank": blank,
    }, diagnostics


def bootstrap(store, root=None):
    """Run the narrative into a fresh store. Returns a replay report."""
    root = root or case_root()
    files, import_diags = load_case_files(root)
    report = {"steps": [], "import_diagnostics": import_diags}

    # Step 0: initial import of the five independently-versioned materials.
    for kind in ("power", "schedule", "conditions", "baseline", "blank"):
        r = store.add_material(files[kind], branch="main",
                               source=f"case:{files[kind].payload['origin_label']}",
                               caused_by="case.step0.import")
        report["steps"].append({"step": 0, "action": f"import:{kind}",
                                "event_seq": r["event"]["seq"],
                                "version": r["material_version"],
                                "diagnostics": r["diagnostics"]})

    # Step 1: integration rules formalized (discard first injection) ->
    # overlap flags, gap clipping and first-injection discard become state.
    r = store.set_rules(RULE_KIND_INTEGRATION, {
        "start_offset_s": 4.0, "end_offset_s": 150.0, "settle_s": 60.0,
        "baseline_mode": "curve_anchors",
        "discard_first_injection": True,
    }, reason="v1 review protocol: discard first injection",
        caused_by="case.step1.integration_rules")
    report["steps"].append({"step": 1, "action": "set_integration_rules",
                            "event_seq": r["event"]["seq"],
                            "diagnostics": r["diagnostics"]})

    # Step 2: blank correspondence rule stated explicitly (by_index). The
    # missing #18 correspondence is a structured diagnostic from here on.
    r = store.set_rules(RULE_KIND_BLANK, {
        "correspondence": "by_index", "required": True, "apply": True,
    }, reason="pair sample and blank injections by ordinal index",
        caused_by="case.step2.blank_rules")
    report["steps"].append({"step": 2, "action": "set_blank_rules",
                            "event_seq": r["event"]["seq"],
                            "diagnostics": r["diagnostics"]})

    # Step 3: parameter revision -- syringe concentration 0.600 -> 0.500 mM.
    # Model becomes stale; raw integrals are untouched.
    revised_cond, cond_diags = parse_conditions(
        _read(os.path.join(root, "revisions", "conditions_r1.json")),
        origin_label="revisions/conditions_r1.json")
    r = store.add_material(revised_cond, source="case:revisions/conditions_r1.json",
                           caused_by="case.step3.concentration_revision")
    report["steps"].append({"step": 3, "action": "concentration_revision",
                            "event_seq": r["event"]["seq"],
                            "version": r["material_version"],
                            "diagnostics": r["diagnostics"]})

    # Step 4: model rules formalized and candidates compared.
    r = store.set_rules(RULE_KIND_MODEL, {
        "tie_relative_tolerance": 5e-4,
        "evaluation_budget": 4000,
        "discard_first_injection": True,
    }, reason="compare one-site/two-site/dilution under BIC+tie rule",
        caused_by="case.step4.model_rules")
    report["steps"].append({"step": 4, "action": "set_model_rules",
                            "event_seq": r["event"]["seq"],
                            "diagnostics": r["diagnostics"]})

    view = store.view("main")
    fit = view.results.get("fit")
    tied = sorted(fit["tied_families"]) if fit else []

    # Step 5: reviewer accepts the one-site candidate (recorded as an event;
    # when several legal candidates are tied all remain present).
    decision_target = "fit:" + (fit["fit_id"] if fit else "missing")
    store.decision(decision_target,
                   "accept_one_site" if "one_site" in tied else "note",
                   reason="one-site selected among tied legal candidates",
                   candidate_ref="one_site",
                   caused_by="case.step5.accept_one_site")
    report["steps"].append({"step": 5, "action": "review_decision_main",
                            "tied_families": tied})

    # Step 6: local source-data fix on a separate branch (injection 12
    # volume typo). Only locally-dependent results change.
    store.create_branch("local_fix", from_branch="main",
                        label="fix injection 12 volume recording")
    revised_sched, _ = parse_schedule(
        _read(os.path.join(root, "revisions", "schedule_r2.csv")),
        origin_label="revisions/schedule_r2.csv")
    r = store.add_material(revised_sched, branch="local_fix",
                           source="case:revisions/schedule_r2.csv",
                           caused_by="case.step6.local_schedule_fix")
    report["steps"].append({"step": 6, "action": "local_schedule_fix_branch",
                            "event_seq": r["event"]["seq"],
                            "diagnostics": r["diagnostics"]})

    # Step 7: conflicting manual decision on the branch -- reviewer rejects
    # dilution-only there, in conflict with main's acceptance.
    view_fix = store.view("local_fix")
    fit_fix = view_fix.results.get("fit")
    store.decision(
        "fit:" + (fit_fix["fit_id"] if fit_fix else "missing"),
        "reject_dilution", branch="local_fix",
        reason="conflicting decision: dilution-only hypothesis rejected after local fix",
        candidate_ref="dilution", caused_by="case.step7.conflict_decision")
    report["steps"].append({"step": 7, "action": "conflicting_decision_branch"})

    report["final"] = {
        "tied_families_main": tied,
        "head_main": store.head_seq("main"),
        "head_local_fix": store.head_seq("local_fix"),
    }
    return report
