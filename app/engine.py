"""Server-side business-rule engine.

Everything the browser can do is also callable here as plain functions over
storage connections, so the same rules are exercised by pytest.

Layering
--------
immutable materials   : versioned rows in ``materials``
decisions             : append-only rows in ``events`` (adopted and rejected)
derived results       : versioned, parent-linked rows in ``results``

Replaying a branch rebuilds the exact visible conclusions at any event: the
latest computation never overwrites history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import storage
from .binding import fit_models
from .errors import IncompleteInput
from .integration import (apply_dilution, correspondence_with_volumes,
                          integrate_injections)


def _parsed(row):
    return json.loads(row["parsed_json"])


def _latest_map(conn, project_id):
    out = {}
    for row in storage.all_materials(conn, project_id):
        out[(row["kind"], row["role"])] = row
    return out


def _conditions(parsed):
    cond = {}
    for f in parsed.get("fields_", []):
        cond[f["name"]] = f
    return cond


@dataclass
class ReplayView:
    project_id: str
    branch_id: str
    head_event_id: str | None
    seq: int
    materials: dict
    events: list
    integrations: list[dict] = field(default_factory=list)
    fit: dict | None = None
    diagnostics: list[dict] = field(default_factory=list)
    decisions: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "project_id": self.project_id,
            "branch_id": self.branch_id,
            "head_event_id": self.head_event_id,
            "seq": self.seq,
            "material_versions": {
                f"{k[0]}:{k[1]}": {"material_id": v["material_id"],
                                   "version": v["version"]}
                for k, v in self.materials.items()},
            "events": self.events,
            "integrations": self.integrations,
            "fit": self.fit,
            "diagnostics": self.diagnostics,
            "decisions": self.decisions,
        }


def compute_integrations(materials: dict) -> dict[str, Any]:
    """Pure derivation from current material versions + branch decisions."""
    power = materials[("power", "main")]
    inj = materials[("injections", "main")]
    baseline = materials.get(("baseline", "main"))
    blank_power = materials.get(("blank_power", "blank"))
    blank_inj = materials.get(("blank_injections", "blank"))
    conc = materials.get(("concentrations", "main"))
    conds = _conditions(_parsed(materials[("conditions", "main")]))

    pp = _parsed(power)
    ii = _parsed(inj)
    bp = _parsed(baseline) if baseline else None
    diagnostics = list(pp.get("diagnostics", [])) + list(ii.get("diagnostics", []))

    raw = integrate_injections(
        pp["samples"], ii["injections"],
        bp["samples"] if bp else None)
    raw_by_index = {r.index: r for r in raw}

    mapping = None
    blank_results = []
    if blank_power and blank_inj:
        bps = _parsed(blank_power)
        bis = _parsed(blank_inj)
        blank_results = integrate_injections(bps["samples"], bis["injections"], None)
    return {"raw_results": raw, "raw_by_index": raw_by_index,
            "blank_results": blank_results, "blank_injections":
            _parsed(blank_inj)["injections"] if blank_inj else None,
            "materials": materials, "conc": _parsed(conc) if conc else None,
            "conds": conds, "diagnostics": diagnostics}


def derive_view(conn, project_id, branch_id, upto_event_id=None,
                decision_overrides=None) -> ReplayView:
    events = [dict(e) for e in storage.branch_events(conn, branch_id)]
    upto = None
    if upto_event_id:
        for i, e in enumerate(events):
            if e["event_id"] == upto_event_id:
                upto = i
                break
        events = events[:upto + 1]

    # material versions visible at this point: import events name explicit ids
    materials: dict[tuple, Any] = {}
    decisions: dict[str, Any] = {
        "discarded_indices": [],
        "blank_rule": None,
        "volume_tolerance_ul": 0.5,
        "accepted_model": None,
        "manual_adj": {},
        "rejected": [],
    }
    event_dicts = []
    for e in events:
        payload = json.loads(e["payload_json"])
        event_dicts.append({**e, "payload": payload})
        et = e["type"]
        if et in ("material_import", "material_revision", "initial_import"):
            for ref in payload.get("materials", []):
                row = storage.material_by_id(conn, ref["material_id"])
                materials[(row["kind"], row["role"])] = row
        elif et == "decision":
            d = payload.get("decision", payload)
            name = d.get("name")
            value = d.get("value")
            adopted = bool(e["adopted"])
            if not adopted:
                decisions["rejected"].append(d)
            elif name == "discard_injections":
                decisions["discarded_indices"] = list(value)
            elif name == "blank_correspondence_rule":
                decisions["blank_rule"] = value
                decisions["volume_tolerance_ul"] = d.get("tolerance_ul", 0.5)
            elif name == "accept_model":
                decisions["accepted_model"] = value
            elif name == "manual_adjustment":
                decisions["manual_adj"][d["injection_index"]] = value

    if decision_overrides:
        decisions.update(decision_overrides)

    view = ReplayView(project_id, branch_id,
                      events[-1]["event_id"] if events else None,
                      len(events), materials, event_dicts,
                      decisions=decisions)
    if ("power", "main") not in materials or ("injections", "main") not in materials:
        return view

    ctx = compute_integrations(materials)
    diagnostics = list(ctx["diagnostics"])
    blank_rule = decisions["blank_rule"]
    if ctx["blank_results"] and blank_rule:
        mapping = correspondence_with_volumes(
            _parsed(materials[("injections", "main")])["injections"],
            ctx["blank_injections"], blank_rule,
            decisions["volume_tolerance_ul"])
        entries = apply_dilution(ctx["raw_results"], ctx["blank_results"], mapping)
    else:
        entries = []
        for r in ctx["raw_results"]:
            entry = r.to_dict()
            if ctx["blank_results"] and not blank_rule:
                entry["blank_correspondence"] = {
                    "matched": False, "rule": None,
                    "reason": "no_rule_declared"}
                entry["net_q_ucal"] = None
                entry["dilution_status"] = "unsubtracted_rule_not_declared"
                entry["diagnostics"].append({
                    "code": "blank_correspondence_rule_missing",
                    "index": r.index,
                    "detail": "a blank exists but no correspondence rule has "
                              "been declared"})
            else:
                entry["blank_correspondence"] = {"matched": False, "rule": None,
                                                 "reason": "no_blank_imported"}
                entry["net_q_ucal"] = r.q_ucal if r.q_ucal is not None else None
                entry["dilution_status"] = "no_blank_available"
                if r.q_ucal is not None:
                    entry["diagnostics"].append({
                        "code": "blank_correspondence_missing",
                        "index": r.index, "rule": None,
                        "reason": "no_blank_imported",
                        "detail": "net heat equals raw heat; dilution not "
                                  "corrected"})
            entries.append(entry)

    # concentrations / conditions
    conc = ctx["conc"]
    conds = ctx["conds"]
    cell_vol = conds.get("cell_volume", {}).get("value")
    syringe = conc.lookup("syringe", "ligand") if conc else None
    macro = conc.lookup("cell", "macromolecule") if conc else None
    syringe_m = syringe["value_m"] if syringe else None
    macro_m = macro["value_m"] if macro else None

    injections = _parsed(materials[("injections", "main")])["injections"]
    discarded = set(decisions["discarded_indices"])
    for entry in entries:
        idx = entry["index"]
        inj = injections[idx - 1]
        entry["discarded"] = idx in discarded
        vol_ul = inj.get("volume_ul")
        entry["volume_ul"] = vol_ul
        entry["ligand_amount_umol"] = (vol_ul * 1e-6 * syringe_m * 1e6
                                       if vol_ul and syringe_m else None)
        entry["manual_adjustment"] = decisions["manual_adj"].get(idx)

    # fit: requires net heats; concentration revisions invalidate stored fits
    # but the historical view recomputes against the *currently visible*
    # concentration version while tagging provenance.
    fit = None
    nets = [e["net_q_ucal"] for e in entries]
    vols = [e["volume_ul"] for e in entries]
    fit_diag = []
    try:
        if syringe_m is not None and macro_m is not None and cell_vol:
            fit = fit_models(nets, vols, syringe_m, macro_m, cell_vol,
                             discarded_indices={i - 1 for i in discarded},
                             budget=20000)
            fit["input_materials"] = {
                "concentrations": materials[("concentrations", "main")]["material_id"],
                "conditions": materials[("conditions", "main")]["material_id"],
                "injections": materials[("injections", "main")]["material_id"],
                "power": materials[("power", "main")]["material_id"]}
            fit["accepted_model"] = decisions["accepted_model"]
    except IncompleteInput as exc:
        fit_diag.append(exc.to_dict())

    for e in entries:
        for d in e["diagnostics"]:
            diagnostics.append({**d, "injection": e["index"]})
    view.integrations = entries
    view.fit = fit
    view.diagnostics = diagnostics + fit_diag
    return view
