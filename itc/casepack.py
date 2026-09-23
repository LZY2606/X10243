"""Offline case pack: "滴定热谱：ITC 注射积分、稀释热与绑定模型审议".

The pack versions power sampling, injection volumes/times, concentrations,
temperature, cell volume, baseline and blank curve separately, then replays:
overlapping injection peaks, first-injection discard, a missing blank
counterpart, a concentration revision, equipotent model candidates, a data
gap at integration endpoints, a localized blank correction, and conflicting
human decisions on two branches."""

import json
import math
import os

import numpy as np

from . import models, pipeline

N_INJECTIONS = 18
FIRST_TIME = 120.0
SPACING = 90.0
WINDOW = 90.0
TAU = 35.0
GAP = (1000.0, 1050.0)
T_END = 1800.0
V0_UL = 1400.0
CELL_MM = 0.06
SYRINGE_MM_V1 = 0.8
SYRINGE_MM_V2 = 0.84
TRUE_N, TRUE_LOGK, TRUE_DH = 0.95, math.log10(3e6), -7000.0


def _injection_times():
    return [FIRST_TIME + SPACING * i for i in range(N_INJECTIONS)]


def _volumes():
    return [2.0] + [10.0] * (N_INJECTIONS - 1)


def _blank_heats():
    return [-(2.0 + 0.08 * i) for i in range(N_INJECTIONS)]


def _true_heats():
    cond = {"V0": V0_UL * 1e-6, "M0": CELL_MM * 1e-3,
            "Xs": SYRINGE_MM_V1 * 1e-3,
            "dv": [v * 1e-6 for v in _volumes()]}
    return models.predict("one_site", [TRUE_N, TRUE_LOGK, TRUE_DH], cond)


def build_power_curve():
    rng = np.random.default_rng(20260919)
    times = _injection_times()
    true_h = _true_heats()
    blank_h = _blank_heats()
    heat_noise = rng.normal(0.0, 0.35, N_INJECTIONS)
    total = [t + b + n for t, b, n in zip(true_h, blank_h, heat_noise)]
    segments = []
    for lo, hi in ((0.0, GAP[0]), (GAP[1], T_END)):
        t = np.arange(lo, hi, 1.0)
        p = 0.8 - 2e-4 * t + rng.normal(0.0, 0.02, t.size)
        for t_i, q_i in zip(times, total):
            mask = t >= t_i
            p[mask] += (q_i / TAU) * np.exp(-(t[mask] - t_i) / TAU)
        segments.append({"t": [round(float(v), 3) for v in t],
                         "p": [round(float(v), 5) for v in p]})
    return {"time_unit": "s", "power_unit": "ucal/s",
            "time_base": "seconds_since_run_start",
            "note": "raw detector power; gap 1000-1050 s is a detector reset",
            "segments": segments}


def build_materials_v1():
    times = _injection_times()
    volumes = _volumes()
    blank_entries = [{"injection_index": i, "time": times[i], "volume": volumes[i],
                      "heat": round(_blank_heats()[i], 4), "heat_unit": "ucal"}
                     for i in range(N_INJECTIONS) if i != 4]
    return {
        "power": build_power_curve(),
        "injections": {"time_unit": "s", "volume_unit": "uL",
                       "time_base": "seconds_since_run_start",
                       "entries": [{"index": i, "time": times[i], "volume": volumes[i]}
                                   for i in range(N_INJECTIONS)]},
        "concentrations": {"cell": {"value": CELL_MM, "unit": "mM"},
                           "syringe": {"value": SYRINGE_MM_V1, "unit": "mM"},
                           "impurity": {"value": 0.3, "unit": "mg/mL"}},
        "temperature": {"value": 25.0, "unit": "°C"},
        "cell_volume": {"value": V0_UL, "unit": "uL"},
        "baseline": {"kind": "poly", "coeffs": [-2e-4, 0.8],
                     "power_unit": "ucal/s", "time_unit": "s"},
        "blank": {"heat_unit": "ucal", "rule_hint": "by_index",
                  "note": "ligand-into-buffer titration; injection 4 missing "
                          "in v1 (vial misread)",
                  "entries": blank_entries},
    }


def build_scenario():
    v1 = build_materials_v1()
    conc_v2 = dict(v1["concentrations"])
    conc_v2["syringe"] = {"value": SYRINGE_MM_V2, "unit": "mM"}
    blank_v2 = dict(v1["blank"])
    blank_v2["note"] = "re-measured; injection 4 recovered"
    blank_v2["entries"] = sorted(
        v1["blank"]["entries"] + [{"injection_index": 4, "time": _injection_times()[4],
                                   "volume": _volumes()[4],
                                   "heat": round(_blank_heats()[4], 4),
                                   "heat_unit": "ucal"}],
        key=lambda e: e["injection_index"])
    return {
        "name": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "materials_v1": v1,
        "steps": [
            {"op": "import_materials"},
            {"op": "integrate",
             "params": {"window_seconds": WINDOW, "discard_first": True}},
            {"op": "blank_correct", "params": {"rule": "by_index"}},
            {"op": "fit", "params": {"budget": 20000}},
            {"op": "revise", "kind": "concentrations", "payload": conc_v2,
             "reason": "syringe re-weighed: 0.80 -> 0.84 mM"},
            {"op": "revise", "kind": "blank", "payload": blank_v2,
             "reason": "blank vial 4 recovered; only local deps affected"},
            {"op": "blank_correct", "params": {"rule": "by_index"}},
            {"op": "fit", "params": {"budget": 20000}},
            {"op": "fit", "params": {"budget": 20000, "equipotent_rel_tol": 0.05,
                                     "only_injections": [14, 15, 16, 17]},
             "label": "late-window dilution check"},
            {"op": "decide", "branch": "main", "target": "best:one_site",
             "action": "accept",
             "rationale": "one-site isotherm consistent with blanks"},
            {"op": "decide", "branch": "review-b", "target": "best:one_site",
             "action": "reject",
             "rationale": "reviewer B: late-window heats look dilution-only"},
            {"op": "decide", "branch": "review-b", "target": "best:dilution",
             "action": "accept",
             "rationale": "conflicting call kept for audit"},
        ],
    }


def _resolve_target(store, token):
    family = token.split(":", 1)[1]
    fits = [r for r in store.results() if r["kind"] == "model_fit" and r["status"] == "ok"]
    if not fits:
        raise RuntimeError("no fit result to resolve decision target")
    seq = {e["id"]: e["seq"] for e in store.events()}
    latest = max(fits, key=lambda r: seq[r["event_id"]])
    for cand in latest["payload"]["candidates"]:
        if cand["family"] == family:
            return cand["id"]
    raise RuntimeError(f"no candidate of family {family}")


def replay(store, scenario):
    parents = []
    for step in scenario["steps"]:
        op = step["op"]
        if op == "import_materials":
            with store.txn() as (cur, txn_id):
                event = store.add_event(cur, txn_id, "import:materials",
                                        {"kinds": sorted(scenario["materials_v1"])},
                                        parents)
                for kind, payload in sorted(scenario["materials_v1"].items()):
                    store.put_material(cur, txn_id, kind, payload, event["id"])
            parents = [event["id"]]
        elif op == "revise":
            with store.txn() as (cur, txn_id):
                event = store.add_event(cur, txn_id, f"revise:{step['kind']}",
                                        {"kind": step["kind"],
                                         "reason": step.get("reason", "")}, parents)
                store.put_material(cur, txn_id, step["kind"], step["payload"],
                                   event["id"])
            parents = [event["id"]]
        elif op == "integrate":
            _, event = pipeline.run_integration(store, step["params"], parents)
            parents = [event["id"]]
        elif op == "blank_correct":
            _, event = pipeline.run_blank_correction(store, step["params"], parents)
            parents = [event["id"]]
        elif op == "fit":
            _, event = pipeline.run_fit(store, step["params"], parents)
            parents = [event["id"]]
        elif op == "decide":
            target = _resolve_target(store, step["target"])
            with store.txn() as (cur, txn_id):
                event = store.add_event(cur, txn_id, f"decision:{step['action']}",
                                        {"branch": step["branch"], "target": target,
                                         "rationale": step["rationale"]}, parents)
                store.put_decision(cur, txn_id, event["id"], step["branch"],
                                   target, step["action"], step["rationale"])
            parents = [event["id"]]
        else:
            raise ValueError(f"unknown scenario op {op}")
    return store


def load_casepack(store, path="casepack"):
    with open(os.path.join(path, "scenario.json"), encoding="utf-8") as fh:
        scenario = json.load(fh)
    return replay(store, scenario)


def main():
    scenario = build_scenario()
    os.makedirs("casepack", exist_ok=True)
    with open("casepack/scenario.json", "w", encoding="utf-8") as fh:
        json.dump(scenario, fh, ensure_ascii=False, indent=1)
    print("wrote casepack/scenario.json")


if __name__ == "__main__":
    main()
