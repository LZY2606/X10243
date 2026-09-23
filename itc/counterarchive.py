"""Replay harness for the counterexample archive.

For every case: seed the baseline into a fresh store, apply exactly one
direct-source change, re-run the pipeline, and measure the blast radius.
Baseline and post-change results are both retained in the store (accepted
and unadopted results coexist); judgment rules are emitted as
machine-readable events.
"""
from __future__ import annotations

import json
import os

from .service import Service
from .store import Store

ARCHIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "counterexamples", "archive.json")


def load_archive(path=None):
    with open(path or ARCHIVE) as f:
        return json.load(f)


def _seed(service, materials):
    for kind in ("power_sampling", "injection_schedule", "concentrations",
                 "temperature", "cell_volume", "baseline", "blank_curve"):
        service.import_material(kind, materials[kind])


def _pipeline(service, params=None):
    integ = service.run_integration(params=params)
    blank = service.run_blank_correction()
    fit = service.run_fit()
    return {"integration": integ, "blank": blank, "fit": fit}


def _windows(run):
    if not run["integration"]["result"]:
        return {}
    return {r["index"]: r["window_s"]
            for r in run["integration"]["result"]["injections"]}


def replay_case(case, baseline_materials):
    """Returns an impact report for one archive case."""
    base_store, case_store = Store(), Store()
    base_svc, case_svc = Service(base_store), Service(case_store)
    _seed(base_svc, baseline_materials)
    _seed(case_svc, baseline_materials)
    change = case["change"]
    params = change.get("params")
    base_run = _pipeline(base_svc, params=params)
    _pipeline(case_svc, params=params)   # pre-change computations
    if "decision" in change:
        d = change["decision"]
        case_svc.decide(d["target"], d["action"], d.get("value"),
                        "counterexample decision")
    elif change.get("special") == "flat_observed":
        # make every observed heat identical -> multiple equal-best models
        case_svc.import_material(
            "power_sampling",
            {"time": baseline_materials["power_sampling"]["time"],
             "power": {"values": [0.5] * len(
                 baseline_materials["power_sampling"]["time"]["values"]),
                       "unit": "uJ/s"},
             "gaps": []})
        case_svc.import_material(
            "blank_curve", {"heats": {"values": [0.0] * 8, "unit": "uJ"}})
    else:
        case_svc.import_material(change["kind"], change["payload"])
    case_run = _pipeline(case_svc, params=params)
    report = {"case_id": case["id"], "expected": case["expected"]}
    bw, cw = _windows(base_run), _windows(case_run)
    if cw:
        report["windows_changed"] = sorted(
            i for i in cw if bw.get(i) != cw[i])
        report["windows_unchanged"] = sorted(
            i for i in cw if bw.get(i) == cw[i])
        report["windows"] = cw
    if base_run["integration"]["result"] and case_run["integration"]["result"]:
        bh = [r["heat_uJ"] for r in
              base_run["integration"]["result"]["injections"]]
        ch = [r["heat_uJ"] for r in
              case_run["integration"]["result"]["injections"]]
        report["integration_unchanged"] = (
            change.get("kind") not in ("power_sampling",
                                       "injection_schedule", "baseline")
            and bh == ch)
    report["judgment_rules"] = sorted({
        json.loads(e["payload"]).get("rule")
        for e in case_svc.store.events()
        if json.loads(e["payload"]).get("rule")} - {None})
    report["blank_status"] = {
        r["index"]: r.get("blank_status")
        for r in (case_run["blank"]["result"] or {}).get("injections", [])}
    fit = case_run["fit"]
    if fit["result"]:
        report["fit_status"] = fit["status"]
        report["candidates"] = [
            {"model": c["model"], "theta": c["theta"], "rank": c["rank"]}
            for c in fit["result"].get("candidates", [])]
        report["fit_observed"] = fit["result"].get("observed_uJ")
    # staleness of the pre-change computations, evaluated in the case store
    report["stale_uids"] = case_svc.store.stale_uids()
    report["integration_stale"] = any(
        c["kind"] == "integration" and case_svc._is_stale(c)
        for c in case_svc.store.computations())
    report["fit_stale"] = any(
        c["kind"] == "fit" and case_svc._is_stale(c)
        for c in case_svc.store.computations())
    # stability: identical inputs -> identical candidate order
    if case["id"] == "equal_best_models":
        again = _pipeline(case_svc)["fit"]
        report["stable_order"] = (
            [c["model"] for c in again["result"]["candidates"]]
            == [c["model"] for c in fit["result"]["candidates"]])
    base_store.close()
    report["store"] = case_store  # caller closes
    return report


def replay_all(path=None):
    archive = load_archive(path)
    reports = []
    for case in archive["cases"]:
        reports.append(replay_case(case, archive["baseline"]["materials"]))
    return archive, reports
