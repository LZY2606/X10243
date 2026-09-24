"""Counterexample archive around ITC / injection integration / binding models.

A baseline is frozen from the separately versioned materials; each entry then
mutates exactly one direct source and checks that the impact stays inside the
correct scope. Adopted and rejected results are both kept; the judgements for
  - overlapping post-injection peaks / windows never crossing data gaps
  - dilution subtraction must name its correspondence rule
  - concentration revision stales models but not the original integrals
are emitted as machine-readable entries.
"""
from __future__ import annotations

import copy

from . import casepack
from .domain import MATERIAL_KINDS, RESULT_OK
from .service import Session
from .store import Store


def _baseline(path):
    store = Store(path)
    session = Session(store)
    mats = casepack.build_materials()
    for kind in MATERIAL_KINDS:
        session.import_material(kind, mats[kind], note="counterexample baseline")
    r_int = session.run_integration()
    r_dil = session.run_dilution(rule="by_index")
    r_fit = session.run_fit()
    return store, session, {"integration": r_int, "dilution": r_dil, "fit": r_fit}


def _stale_kinds(session):
    view = session.view()
    return sorted({r["kind"] for r in view["results"] if r["freshness"] == "stale"})


def _check_overlap(tmp):
    store, session, base = _baseline(tmp / "overlap.db")
    mats = casepack.build_materials()
    moved = copy.deepcopy(mats["injection_schedule"])
    moved["injections"][6]["time"]["value"] = 505.0  # I07 closer to I06
    session.import_material("injection_schedule", moved, note="move I07 closer")
    stale = _stale_kinds(session)
    rid = session.run_integration()
    res = store.result(rid)
    i06 = res["payload"]["injections"][5]
    overlapped = any(d["code"] == "overlapping-peaks" for d in i06["diagnostics"])
    ok = stale == ["dilution", "fit", "integration"] and overlapped
    return {"name": "overlapping-injection-peaks",
            "mutation": "injection_schedule: I07 moved next to I06",
            "affected": stale, "expected": ["dilution", "fit", "integration"],
            "details": {"overlap_diagnostic": overlapped}, "pass": ok}


def _check_first_injection(tmp):
    store, session, base = _baseline(tmp / "first.db")
    rid_keep = session.run_integration(config={"discard_first": False})
    kept = store.result(rid_keep)["payload"]["injections"][0]
    old = store.result(base["integration"])["payload"]["injections"][0]
    ok = (kept["status"] == "ok" and old["status"] == "discarded"
          and store.result(base["integration"]) is not None)
    return {"name": "first-injection-discard",
            "mutation": "integration config discard_first toggled",
            "affected": ["integration"], "expected": ["integration"],
            "details": {"adopted_status": kept["status"], "rejected_status": old["status"],
                        "both_results_kept": True},
            "pass": ok}


def _check_missing_blank(tmp):
    store, session, base = _baseline(tmp / "blank.db")
    mats = casepack.build_materials()
    fewer = copy.deepcopy(mats["blank_trace"])
    fewer["schedule"]["injections"] = fewer["schedule"]["injections"][:3]
    session.import_material("blank_trace", fewer, note="blank run truncated")
    stale = _stale_kinds(session)
    rid = session.run_dilution(rule="by_index")
    res = store.result(rid)
    missing = [e["injection_id"] for e in res["payload"]["injections"]
               if e["status"] == "missing_blank"]
    rule_stated = res["payload"].get("rule") == "by_index"
    ok = stale == ["dilution", "fit"] and len(missing) >= 4 and rule_stated
    return {"name": "blank-correspondence-missing",
            "mutation": "blank_trace: only 3 blank injections remain",
            "affected": stale, "expected": ["dilution", "fit"],
            "details": {"missing_blank": missing, "rule": res["payload"].get("rule"),
                        "integration_untouched": "integration" not in stale},
            "pass": ok}


def _check_concentration_revision(tmp):
    store, session, base = _baseline(tmp / "conc.db")
    old_int = store.result(base["integration"])["payload"]["injections"][2]["heat_uJ"]
    session.import_material("concentrations", casepack.revised_concentrations(),
                            note="syringe re-weighed")
    stale = _stale_kinds(session)
    new_int = store.result(base["integration"])["payload"]["injections"][2]["heat_uJ"]
    ok = stale == ["fit"] and old_int == new_int
    return {"name": "concentration-revision",
            "mutation": "concentrations: syringe 1.0 -> 1.15 mM",
            "affected": stale, "expected": ["fit"],
            "details": {"integrals_unchanged": old_int == new_int,
                        "models_staled": stale},
            "pass": ok}


def _check_equipotent(tmp):
    store, session, base = _baseline(tmp / "tie.db")
    fit = store.result(base["fit"])
    cands = fit["payload"]["candidates"]
    all_kept = len(fit["payload"]["models"]) == 3
    ok = cands == ["one_site", "two_site"] and all_kept
    return {"name": "multiple-equipotent-models",
            "mutation": "none (baseline fit)",
            "affected": ["fit"], "expected": ["fit"],
            "details": {"candidates": cands, "all_models_kept": all_kept,
                        "unique_decision": fit["payload"]["unique_decision"]},
            "pass": ok}


def _check_gap_endpoint(tmp):
    store, session, base = _baseline(tmp / "gap.db")
    mats = casepack.build_materials()
    moved = copy.deepcopy(mats["power_trace"])
    moved["samples"] = [s for s in moved["samples"]
                        if not (700.0 <= s["t"] < 720.0)]
    session.import_material("power_trace", moved, note="gap moved onto I09")
    stale = _stale_kinds(session)
    rid = session.run_integration()
    res = store.result(rid)
    old = {e["injection_id"]: e for e in store.result(base["integration"])["payload"]["injections"]}
    new = {e["injection_id"]: e for e in res["payload"]["injections"]}
    untouched = all(old[k]["heat_uJ"] == new[k]["heat_uJ"] for k in ("I02", "I03", "I04"))
    i09_changed = old["I09"]["heat_uJ"] != new["I09"]["heat_uJ"] or new["I09"]["status"] != "ok"
    no_crossing = all(
        not (w and any(g[0] > w[0] and g[0] < w[1] for g in res["payload"]["gaps"]))
        for w in (e["window"] for e in res["payload"]["injections"]))
    ok = stale == ["dilution", "fit", "integration"] and untouched and i09_changed and no_crossing
    return {"name": "data-gap-and-integration-endpoints",
            "mutation": "power_trace: data gap moved onto I09",
            "affected": stale, "expected": ["dilution", "fit", "integration"],
            "details": {"untouched_heats_identical": untouched,
                        "i09_impacted": i09_changed,
                        "no_window_crosses_gap": no_crossing},
            "pass": ok}


CHECKS = [_check_overlap, _check_first_injection, _check_missing_blank,
          _check_concentration_revision, _check_equipotent, _check_gap_endpoint]


def run_archive(tmp_path):
    """Execute every counterexample; returns a machine-readable report."""
    import pathlib
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    report = [check(tmp_path) for check in CHECKS]
    return {"archive": "itc-counterexamples/1", "checks": report,
            "all_pass": all(c["pass"] for c in report)}
