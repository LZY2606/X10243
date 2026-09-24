"""Service layer: orchestrates materials, computations, events and replay.

Every computation records an event with input digest, parameter version and
parent dependencies. Decisions (accept / reject / adjust) are events too.
Replaying to any old event restores the dependency versions and visible
conclusions of that moment; nothing is overwritten by newer computations.
"""

from . import fit as fit_mod
from . import integrate as integ_mod
from . import units
from .store import digest

CATEGORY_ORDER = [
    "power_curve", "injection_schedule", "concentration", "temperature",
    "cell_volume", "baseline", "blank_curve",
]

FIT_MATERIAL_DEPS = ("concentration", "cell_volume", "temperature")


def _visible_branches(branch):
    return ("main",) if branch == "main" else ("main", branch)


class Service:
    def __init__(self, store):
        self.store = store

    # ---------------- materials ----------------
    def import_materials(self, materials, branch="main"):
        ids = {}
        for cat in CATEGORY_ORDER:
            if cat in materials:
                ids[cat] = self.store.add_material(cat, materials[cat], branch)
        ev = self.store.add_event("import", {
            "materials": {k: {"id": v["id"], "version": v["version"]}
                          for k, v in ids.items()},
        }, branch)
        return ids, ev

    def revise_material(self, category, payload, branch="main", note=None):
        m = self.store.add_material(category, payload, branch)
        ev = self.store.add_event("revise", {
            "category": category, "id": m["id"], "version": m["version"],
            "note": note,
        }, branch)
        stale = []
        for r in self.store.results(branches=_visible_branches(branch)):
            if r["kind"] != "fit":
                continue
            deps = r["params"].get("deps", {})
            if category in deps and deps[category]["id"] != m["id"]:
                stale.append(r["id"])
        if stale:
            self.store.add_event("mark_stale", {
                "result_ids": stale,
                "reason": f"{category}_revised",
                "category": category,
            }, branch)
        return m, ev

    # ---------------- computations ----------------
    def _record(self, kind, payload, deps, params, branch, parents=None):
        parents = parents or []
        full_params = {"deps": deps, "params": params, "parents": parents}
        inputs_digest = digest({"kind": kind, "deps": deps,
                                "params": params, "parents": parents})
        rid = f"{kind}-{inputs_digest[:12]}"
        ev = self.store.add_event(kind, {
            "result_id": rid,
            "inputs_digest": inputs_digest,
            "deps": deps,
            "params": params,
            "parents": parents,
            "status": payload.get("status"),
        }, branch)
        self.store.add_result(rid, kind, payload.get("status", "ok"),
                              inputs_digest, full_params, payload,
                              payload.get("diagnostics", []), ev["seq"], branch)
        return rid, ev

    def _latest(self, branch, as_of=None):
        return self.store.latest_materials(_visible_branches(branch), as_of)

    def run_integration(self, branch="main", params=None):
        params = dict(params or {})
        mats = self._latest(branch)
        need = ("power_curve", "injection_schedule", "baseline")
        missing = [c for c in need if c not in mats]
        if missing:
            payload = {"status": "incomplete",
                       "diagnostics": [{"code": "missing_material",
                                        "categories": missing}]}
            return self._record("integration", payload, {}, params, branch)
        res = integ_mod.integrate_injections(
            mats["power_curve"]["payload"],
            mats["injection_schedule"]["payload"],
            mats["baseline"]["payload"], params)
        deps = {c: {"id": mats[c]["id"], "version": mats[c]["version"]} for c in need}
        return self._record("integration", res, deps, params, branch)

    def run_blank_subtraction(self, branch="main", rule=None, integ_params=None):
        rule = rule or {"kind": "by_index"}
        mats = self._latest(branch)
        integrations = [r for r in self.store.results(_visible_branches(branch))
                        if r["kind"] == "integration"]
        if not integrations or "blank_curve" not in mats:
            payload = {"status": "incomplete",
                       "diagnostics": [{"code": "missing_material",
                                        "categories": ["integration", "blank_curve"]}]}
            return self._record("blank_subtraction", payload, {}, {"rule": rule}, branch)
        sample = integrations[-1]["payload"]
        blank_mat = mats["blank_curve"]["payload"]
        blank_res = integ_mod.integrate_injections(
            blank_mat["power"], mats["injection_schedule"]["payload"],
            blank_mat["baseline"], integ_params or {})
        res = integ_mod.subtract_blank(sample, blank_res, rule)
        deps = {"blank_curve": {"id": mats["blank_curve"]["id"],
                                "version": mats["blank_curve"]["version"]},
                "injection_schedule": {"id": mats["injection_schedule"]["id"],
                                       "version": mats["injection_schedule"]["version"]}}
        return self._record("blank_subtraction", res, deps, {"rule": rule}, branch,
                            parents=[integrations[-1]["id"]])

    def run_fits(self, branch="main", params=None):
        params = dict(params or {})
        mats = self._latest(branch)
        results = self.store.results(_visible_branches(branch))
        blanks = [r for r in results if r["kind"] == "blank_subtraction"]
        integs = [r for r in results if r["kind"] == "integration"]
        source = blanks[-1] if blanks else (integs[-1] if integs else None)
        missing = [c for c in FIT_MATERIAL_DEPS if c not in mats]
        if source is None or missing:
            payload = {"status": "incomplete",
                       "diagnostics": [{"code": "missing_material",
                                        "categories": missing or ["integration"]}],
                       "candidates": []}
            return self._record("fit", payload, {}, params, branch)
        heats_in = []
        for h in source["payload"]["heats"]:
            if h["excluded"] or h["status"] != "ok" or h["heat"] is None:
                continue
            q = h.get("heat_net") if h.get("blank_corrected") else h["heat"]
            if q["unit"] != "J":
                continue
            heats_in.append({"index": h["index"], "heat": q["value"]})
        dv_all = [h["volume_L"] for h in source["payload"]["heats"]]
        conc_cell, cdiag = units.normalize(
            mats["concentration"]["payload"]["cell"]["value"],
            mats["concentration"]["payload"]["cell"]["unit"], "M")
        conc_syr, sdiag = units.normalize(
            mats["concentration"]["payload"]["syringe"]["value"],
            mats["concentration"]["payload"]["syringe"]["unit"], "M")
        vol, vdiag = units.normalize(
            mats["cell_volume"]["payload"]["value"],
            mats["cell_volume"]["payload"]["unit"], "L")
        if cdiag or sdiag or vdiag:
            diags = [d for d in (cdiag, sdiag, vdiag) if d]
            payload = {"status": "semantic_unsolvable", "diagnostics": diags,
                       "candidates": []}
            deps = {c: {"id": mats[c]["id"], "version": mats[c]["version"]}
                    for c in FIT_MATERIAL_DEPS}
            return self._record("fit", payload, deps, params, branch,
                                parents=[source["id"]])
        res = fit_mod.fit_models(heats_in, dv_all, conc_cell, conc_syr, vol, params)
        deps = {c: {"id": mats[c]["id"], "version": mats[c]["version"]}
                for c in FIT_MATERIAL_DEPS}
        return self._record("fit", res, deps, params, branch, parents=[source["id"]])

    # ---------------- decisions ----------------
    def decide(self, result_id, candidate, action, note=None, branch="main",
               params=None):
        if action not in ("accept", "reject", "adjust"):
            raise ValueError(f"unknown decision action: {action}")
        ev = self.store.add_event("decision", {
            "result_id": result_id,
            "candidate": candidate,
            "action": action,
            "note": note,
            "params": params or {},
        }, branch)
        return ev

    def open_branch(self, name, note=None):
        return self.store.add_event("branch", {"name": name, "note": note}, name)

    # ---------------- state / replay ----------------
    def state(self, branch="main", as_of=None):
        branches = _visible_branches(branch)
        mats = self.store.latest_materials(branches, as_of)
        results = self.store.results(branches, as_of)
        events = self.store.events(branches, as_of)
        stale_ids, decisions = set(), []
        for e in events:
            if e["type"] == "mark_stale":
                stale_ids.update(e["payload"]["result_ids"])
            elif e["type"] == "decision":
                decisions.append(e["payload"])
        for r in results:
            if r["id"] in stale_ids:
                r["status"] = "stale"
            r["decisions"] = [d for d in decisions if d["result_id"] == r["id"]]
        branch_names = sorted({e["branch"] for e in self.store.events()} | {"main"})
        return {
            "branch": branch,
            "as_of": as_of,
            "branches": branch_names,
            "materials": [mats[c] for c in CATEGORY_ORDER if c in mats],
            "results": results,
            "events": events,
        }

    # ---------------- transfer ----------------
    def export(self):
        return self.store.export_data()

    def import_(self, data):
        return self.store.import_data(data)
