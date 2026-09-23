"""Loader + narrative replay for the offline case pack in ``casepack/``.

The pack is domain material (ITC titration), not a generic table: raw
power samples, injection schedule, blank heats and instrument metadata,
plus a narrative that exercises a parameter revision, a local-dependency
source change and conflicting human decisions.
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "casepack")


def _read_csv(name):
    with open(os.path.join(HERE, name)) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    header, data = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in data]


def load_casepack():
    power_rows = _read_csv("power_curve.csv")
    inj_rows = _read_csv("injections.csv")
    blank_rows = _read_csv("blank_heats.csv")
    with open(os.path.join(HERE, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(HERE, "narrative.json")) as f:
        narrative = json.load(f)
    materials = {
        "power_sampling": {
            "time": {"values": [float(r["time_s"]) for r in power_rows],
                     "unit": "s", "basis": "instrument_clock"},
            "power": {"values": [float(r["power_uJ_per_s"])
                                 for r in power_rows], "unit": "uJ/s"},
            "gaps": [{"start": {"value": meta["gap"]["start_s"],
                                "unit": "s"},
                      "end": {"value": meta["gap"]["end_s"], "unit": "s"},
                      "reason": meta["gap"]["reason"]}],
        },
        "injection_schedule": {
            "injections": [
                {"time": {"value": float(r["time_s"]), "unit": "s",
                          "basis": "instrument_clock"},
                 "volume": {"value": float(r["volume_uL"]), "unit": "uL"}}
                for r in inj_rows]},
        "concentrations": meta["concentrations"],
        "temperature": meta["temperature"],
        "cell_volume": meta["cell_volume"],
        "baseline": meta["baseline"],
        "blank_curve": {
            "heats": {"values": [float(r["heat_uJ"]) for r in blank_rows],
                      "unit": "uJ"}},
    }
    return {"materials": materials, "meta": meta, "narrative": narrative}


def seed(service):
    """Initial import: every material kind versioned separately."""
    pack = load_casepack()
    versions = {}
    for kind in ("power_sampling", "injection_schedule", "concentrations",
                 "temperature", "cell_volume", "baseline", "blank_curve"):
        versions[kind] = service.import_material(kind, pack["materials"][kind])
    return versions


def replay_narrative(service):
    """Replay the bundled narrative on top of the seeded state.

    Steps (each leaves machine-readable events):
      1. main-chain review: integrate -> blank-correct -> fit
      2. first injection discarded (small-volume first injection)
      3. parameter revision: concentrations re-weighed -> fits stale,
         integrations untouched
      4. local source change: temperature re-read -> only fits stale
      5. conflicting human decisions on the preferred model
      6. branch-b: re-integrate with a longer post window and keep every
         equal-best model candidate
    """
    log = {}
    log["integration"] = service.run_integration()
    log["blank"] = service.run_blank_correction()
    log["fit"] = service.run_fit()
    log["discard_first"] = service.decide(
        "injection:0", "discard", {"reason": "2 uL priming injection"},
        "首注体积仅 2 µL，属预充注射，予以舍弃")
    log["fit_after_discard"] = service.run_fit()
    log["concentration_revision"] = service.import_material(
        "concentrations",
        {"cell": {"value": 0.1, "unit": "mM"},
         "syringe": {"value": 1.1, "unit": "mM"}})
    log["temperature_reread"] = service.import_material(
        "temperature", {"value": 298.1, "unit": "K"})
    log["decision_one_site"] = service.decide(
        "model:preferred", "accept", {"model": "one_site"},
        "一位点模型残差已白化，采纳")
    log["decision_two_site"] = service.decide(
        "model:preferred", "accept", {"model": "two_site"},
        "第二位审阅者偏好两位点模型（冲突决定，两者均保留）")
    log["branch_b_integration"] = service.run_integration(
        params={"post_window_s": 90.0}, branch="branch-b")
    log["branch_b_blank"] = service.run_blank_correction(branch="branch-b")
    log["branch_b_fit"] = service.run_fit(branch="branch-b")
    return log
