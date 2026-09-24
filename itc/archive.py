"""Counterexample archive builder.

Replays the case-pack narrative into a store: versioned baseline of all
materials, then single-source changes whose impact must stay in the correct
scope. Both adopted and rejected results are retained. Key judgments are
recorded as machine-readable rule events:

  * integration_window_no_gap_crossing
  * dilution_blank_correspondence_rule_required
  * concentration_revision_stales_models_not_integrals
  * post_injection_peaks_may_overlap
  * first_injection_discard
"""
import csv
import json
import os

from . import pipeline
from .ids import digest

RULE_GAP = "integration_window_no_gap_crossing"
RULE_BLANK = "dilution_blank_correspondence_rule_required"
RULE_CONC = "concentration_revision_stales_models_not_integrals"
RULE_OVERLAP = "post_injection_peaks_may_overlap"
RULE_FIRST = "first_injection_discard"


def load_casepack(pack_dir):
    def read_csv(name):
        with open(os.path.join(pack_dir, name)) as fh:
            rows = list(csv.reader(fh))
        return {"format": "csv", "columns": rows[0],
                "rows": [list(r) for r in rows[1:]]}

    def read_json(name):
        with open(os.path.join(pack_dir, name)) as fh:
            return json.load(fh)

    power = read_csv("power_curve.csv")
    power["units"] = {"time": "s", "power": "ucal/s"}
    blank = read_csv("blank_power.csv")
    blank["units"] = {"time": "s", "power": "ucal/s"}
    sched = read_csv("injections.csv")
    sched["units"] = {"time": "s", "volume": "uL", "duration": "s"}
    sched_v2 = read_csv("injections_v2_overlap.csv")
    sched_v2["units"] = {"time": "s", "volume": "uL", "duration": "s"}
    return {
        "power_curve": power,
        "blank_curve": blank,
        "injection_schedule": sched,
        "injection_schedule_v2": sched_v2,
        "concentrations_v1": read_json("concentrations_v1.json"),
        "concentrations_v2": read_json("concentrations_v2.json"),
        "temperature": read_json("temperature.json"),
        "cell_volume": read_json("cell_volume.json"),
        "baseline": read_json("baseline.json"),
        "manifest": read_json("manifest.json"),
    }


def _integrals_digest(store, result_id):
    cmp_ = store.computation(result_id)
    payload = json.loads(cmp_["payload"])
    return digest([(i["index"], i["heat_uJ"]) for i in payload["injections"]])


def build_counterexample_archive(store, pack_dir, budget=None):
    pack = load_casepack(pack_dir)
    log = {"events": [], "computations": {}, "decisions": []}

    # --- 1. versioned baseline: every material kind imported separately ---
    versions = {}
    for kind, content in (
            ("power_curve", pack["power_curve"]),
            ("injection_schedule", pack["injection_schedule"]),
            ("concentrations", pack["concentrations_v1"]),
            ("temperature", pack["temperature"]),
            ("cell_volume", pack["cell_volume"]),
            ("baseline", pack["baseline"]),
            ("blank_curve", pack["blank_curve"])):
        m = store.add_material(kind, content)
        versions[kind] = m["version"]
        ev = store.add_event("import_material", {
            "material_id": m["material_id"], "digest": m["digest"],
            "note": "raw representation preserved; normalization is derived"},
            parents=[m["material_id"]])
        log["events"].append(ev["event_id"])
    store.add_event("baseline_established", {
        "versions": versions,
        "note": "power sampling, injection volumes/times, concentrations, "
                "temperature, cell volume, baseline and blank curve are "
                "versioned separately"})

    # --- 2. integration (sample + blank) and dilution subtraction --------
    integ, diags, _ = pipeline.run_integration(store, budget=budget)
    log["computations"]["integration"] = integ
    gap_diags = [d for d in diags if d["code"] == "window_crosses_gap"]
    store.add_event("integrate", {
        "sample": integ["sample"]["result_id"],
        "blank": integ["blank"]["result_id"],
        "subtraction": integ["subtraction"]["result_id"],
        "diagnostics": [d["code"] for d in diags]},
        parents=[integ["sample"]["result_id"], integ["blank"]["result_id"]])
    for d in gap_diags:
        store.add_event("rule_verdict", {
            "rule": RULE_GAP, "verdict": "enforced", "evidence": d,
            "note": "integration window truncated at gap edge; windows never "
                    "cross data gaps"}, parents=[integ["sample"]["result_id"]])
    blank_missing = [d for d in diags if d["code"] == "blank_missing"]
    store.add_event("rule_verdict", {
        "rule": RULE_BLANK, "verdict": "enforced",
        "evidence": {"rule_used": integ["rule"],
                     "missing": [d["path"] for d in blank_missing]},
        "note": "dilution-heat subtraction states its correspondence rule; "
                "injections without a blank counterpart keep raw heats and "
                "are excluded from the corrected series"},
        parents=[integ["subtraction"]["result_id"]])

    # --- 3. first model fit ----------------------------------------------
    fit1, fit1_diags, _ = pipeline.run_fit(store, budget=budget)
    log["computations"]["fit_v1"] = fit1
    fit1_payload = json.loads(store.computation(fit1["result_id"])["payload"])
    equally = [c["model"] for c in fit1_payload["candidates"] if c["equally_good"]]
    store.add_event("fit_models", {
        "result_id": fit1["result_id"],
        "candidates": [(c["model"], c["rss"]) for c in fit1_payload["candidates"]],
        "equally_good": equally,
        "note": "all legal candidates retained when the fit cannot uniquely "
                "decide"}, parents=[fit1["result_id"]])

    # --- 4. decision: discard the first injection, then refit -------------
    dec_first = store.add_decision(
        "inj_00", "discard", "first injection (0.4 uL) is routinely discarded "
        "from fitting; its integral is retained", {"rule": RULE_FIRST})
    log["decisions"].append(dec_first["decision_id"])
    store.add_event("rule_verdict", {
        "rule": RULE_FIRST, "verdict": "applied",
        "evidence": {"decision": dec_first["decision_id"], "target": "inj_00"},
        "note": "discard affects fitting membership only, never the stored "
                "integral"}, parents=[dec_first["decision_id"]])
    fit2, _, _ = pipeline.run_fit(store, excluded=(0,),
                                  decision_parents=(dec_first["decision_id"],),
                                  budget=budget)
    log["computations"]["fit_no_first"] = fit2

    # --- 5. concentration revision: models stale, integrals unchanged -----
    integ_digest_before = _integrals_digest(store, integ["sample"]["result_id"])
    m2 = store.add_material("concentrations", pack["concentrations_v2"])
    current_fits = [c for c in store.computations(kind="model_fit")]
    staled = [c["result_id"] for c in current_fits
              if json.loads(c["parents"]).get("concentrations") != m2["version"]]
    integ_digest_after = _integrals_digest(store, integ["sample"]["result_id"])
    store.add_event("revise_material", {
        "material_id": m2["material_id"], "digest": m2["digest"],
        "reason": "syringe concentration re-assayed 3.00 -> 2.90 mM"},
        parents=[m2["material_id"]])
    store.add_event("rule_verdict", {
        "rule": RULE_CONC, "verdict": "enforced",
        "evidence": {
            "staled_models": staled,
            "integrals_unchanged": integ_digest_before == integ_digest_after,
            "integral_digest_before": integ_digest_before,
            "integral_digest_after": integ_digest_after},
        "note": "concentration revision expires model fits but must not "
                "alter stored integrals"}, parents=staled)
    fit3, _, _ = pipeline.run_fit(store, excluded=(0,),
                                  decision_parents=(dec_first["decision_id"],),
                                  budget=budget)
    log["computations"]["fit_conc_v2"] = fit3

    # --- 6. conflicting human decisions on two branches -------------------
    fit3_payload = json.loads(store.computation(fit3["result_id"])["payload"])
    one_site = next(c for c in fit3_payload["candidates"] if c["model"] == "one_site")
    dec_accept = store.add_decision(
        fit3["result_id"], "accept",
        "reviewer A: one-site model accepted on main",
        {"model": "one_site", "rss": one_site["rss"]}, branch="main")
    store.add_event("branch_open", {"branch": "review",
                                    "reason": "independent reviewer B"})
    dec_reject = store.add_decision(
        fit3["result_id"], "reject",
        "reviewer B: residuals structure unacceptable; prefers two_site",
        {"model": "one_site"}, branch="review")
    log["decisions"] += [dec_accept["decision_id"], dec_reject["decision_id"]]
    store.add_event("decision_conflict", {
        "target": fit3["result_id"],
        "decisions": [dec_accept["decision_id"], dec_reject["decision_id"]],
        "note": "conflicting human decisions are both retained with their "
                "branch identity"}, parents=[dec_accept["decision_id"],
                                             dec_reject["decision_id"]])

    # --- 7. second branch: single-source change with local impact ---------
    store.add_event("branch_open", {"branch": "overlap",
                                    "reason": "probe overlapping peaks and "
                                              "gap/endpoint rules"})
    m3 = store.add_material("injection_schedule",
                            pack["injection_schedule_v2"], branch="overlap")
    store.add_event("revise_material", {
        "material_id": m3["material_id"],
        "reason": "injections 18/19 moved closer together (single source "
                  "change)", "branch": "overlap"}, parents=[m3["material_id"]],
        branch="overlap")
    integ2, diags2, _ = pipeline.run_integration(store, branch="overlap",
                                                 budget=budget)
    log["computations"]["integration_overlap"] = integ2
    before = {i["index"]: i["heat_uJ"] for i in json.loads(
        store.computation(integ["sample"]["result_id"])["payload"])["injections"]}
    after = {i["index"]: i["heat_uJ"] for i in json.loads(
        store.computation(integ2["sample"]["result_id"])["payload"])["injections"]}
    changed = sorted(i for i in before if before[i] != after.get(i))
    overlap_flags = sorted(i["index"] for i in json.loads(
        store.computation(integ2["sample"]["result_id"])["payload"])["injections"]
        if "overlap" in i["flags"])
    store.add_event("impact_scope", {
        "source": m3["material_id"], "changed_injections": changed,
        "unchanged_injections": sorted(i for i in before if i not in changed),
        "note": "a single-source change must only affect its dependents"},
        parents=[integ["sample"]["result_id"], integ2["sample"]["result_id"]],
        branch="overlap")
    store.add_event("rule_verdict", {
        "rule": RULE_OVERLAP, "verdict": "allowed",
        "evidence": {"overlap_injections": overlap_flags},
        "note": "post-injection peaks may overlap; overlap is flagged, "
                "never an error"}, parents=[integ2["sample"]["result_id"]],
        branch="overlap")

    log["summary"] = {
        "events": store.max_seq(),
        "branches": store.branches(),
        "changed_injections_overlap_branch": changed,
        "equally_good_models_v1": equally,
        "staled_models": staled,
    }
    return log
