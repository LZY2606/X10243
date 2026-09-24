"""Loader for the offline case package. Executes the case script through
the normal service layer so every step produces real events and results.
"""

import json
import os

DEFAULT_CASE = os.path.join(os.path.dirname(__file__), "..", "casepack",
                            "itc_titration_review_case.json")


def load_case(service, path=None):
    path = path or DEFAULT_CASE
    with open(path, encoding="utf-8") as fh:
        case = json.load(fh)
    summary = {"case_id": case["case_id"], "steps": []}
    service.import_materials(case["materials"])
    fit_result_id = None
    for step in case["script"]:
        op = step["op"]
        branch = step.get("branch", "main")
        if op == "integrate":
            rid, ev = service.run_integration(params=step.get("params"))
        elif op == "blank_subtract":
            rid, ev = service.run_blank_subtraction(rule=step.get("rule"))
        elif op == "fit":
            rid, ev = service.run_fits(params=step.get("params"))
            fit_result_id = rid
        elif op == "decide":
            ev = service.decide(fit_result_id, step["candidate"], step["action"],
                                note=step.get("note"), branch=branch)
            rid = fit_result_id
        elif op == "revise":
            _, ev = service.revise_material(step["category"], step["payload"],
                                            note=step.get("note"))
            rid = None
        elif op == "branch":
            ev = service.open_branch(step["name"])
            rid = None
        else:
            raise ValueError(f"unknown case op: {op}")
        summary["steps"].append({"op": op, "event_seq": ev["seq"],
                                 "result_id": rid})
    return summary
