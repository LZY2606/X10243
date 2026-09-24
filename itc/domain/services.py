"""Application service: orchestrates material import, decisions, derived
computations, review actions and branches. Business rules live here and in
domain.*, never in the browser."""
from __future__ import annotations

import time

from . import pipeline
from . import integration as I
from .blank import MATCH_BY_ORDINAL, BlankParams
from .events import (BLANK_INTEGRATION_COMPUTED, DECISION_BLANK_RULE,
                     DECISION_FIRST_INJECTION, DECISION_INTEGRATION_PARAMS,
                     DECISION_MODEL_PREFERENCE, INTEGRATION_COMPUTED,
                     MATERIAL_COMMITTED, MODELS_FITTED, NET_HEATS_COMPUTED,
                     PARAMS_COMMITTED, REVIEW_ACCEPT, REVIEW_ADJUST,
                     REVIEW_REJECT, IdFactory, digest, short)
from .state import DEFAULT_BRANCH, Project, replay


def now_us() -> int:
    return time.time_ns() // 1000


class Service:
    def __init__(self, project: Project | None = None, store=None):
        self.project = project or Project()
        self.project.init_branch(DEFAULT_BRANCH)
        self.store = store

    # ---- persistence --------------------------------------------------------
    def save(self, description="save"):
        if self.store is not None:
            self.store.save_project(self.project, description)

    # ---- material import ----------------------------------------------------
    def import_material(self, kind, role, raw_text, *, version=None,
                        branch=DEFAULT_BRANCH, note=""):
        version = version or self._next_material_version(kind, role, branch)
        record = pipeline.parse_material(kind, role, version, raw_text)
        material_id = IdFactory.material_id(kind, role, raw_text)
        record["material_id"] = material_id
        payload = {"record": record}
        ev = self.project.append(
            branch, MATERIAL_COMMITTED, payload, note=note,
            epoch_us=now_us())
        self.save(f"import {kind}/{role} {version}")
        return ev, record

    def _next_material_version(self, kind, role, branch):
        pj = replay(self.project, branch)
        n = sum(1 for h in pj.material_history
                if h["key"] == [kind, role])
        return f"v{n + 1}"

    def commit_params(self, family, version, params_dict, *,
                      branch=DEFAULT_BRANCH, note=""):
        payload = {"family": family, "version": version,
                   "params": params_dict}
        ev = self.project.append(branch, PARAMS_COMMITTED, payload, note=note,
                                 epoch_us=now_us())
        self.save(f"params {family} {version}")
        return ev

    # ---- decisions ----------------------------------------------------------
    def decide_first_injection(self, discard: bool, *, branch=DEFAULT_BRANCH,
                               note=""):
        payload = {"discard": bool(discard), "scope": "first_injection"}
        ev = self.project.append(branch, DECISION_FIRST_INJECTION, payload,
                                 note=note, epoch_us=now_us())
        self.save("decision first injection")
        return ev

    def decide_blank_rule(self, rule=MATCH_BY_ORDINAL, scale_by_volume=True, *,
                          branch=DEFAULT_BRANCH, note=""):
        payload = {"rule": rule, "scale_by_volume": bool(scale_by_volume)}
        ev = self.project.append(branch, DECISION_BLANK_RULE, payload,
                                 note=note, epoch_us=now_us())
        self.save("decision blank rule")
        return ev

    def decide_model_preference(self, model_id, *, branch=DEFAULT_BRANCH,
                                note=""):
        payload = {"preferred_model": model_id,
                   "binding": "preference does not discard other candidates"}
        ev = self.project.append(branch, DECISION_MODEL_PREFERENCE, payload,
                                 note=note, epoch_us=now_us())
        self.save("decision model preference")
        return ev

    def decide_integration_params(self, overrides: dict, version: str, *,
                                  branch=DEFAULT_BRANCH, note=""):
        payload = {"version": version, "overrides": overrides}
        ev = self.project.append(branch, DECISION_INTEGRATION_PARAMS, payload,
                                 note=note, epoch_us=now_us())
        self.save("decision integration params")
        return ev

    # ---- integration --------------------------------------------------------
    def _integration_params(self, pj, branch):
        base = dict(pre_injection_baseline_s=10.0, max_window_s=180.0,
                    return_tolerance=4.0, min_points=3,
                    guard_before_next_s=1.0, baseline_rule="REFERENCE_CURVE")
        version = "integ-default-v1"
        pe = pj.params_versions.get("integration")
        decision = pj.decisions.get(DECISION_INTEGRATION_PARAMS)
        if pe:
            base.update(pe["params"]); version = pe["version"]
        if decision:
            base.update(decision["overrides"])
            version = decision["version"]
        return I.IntegrationParams(version=version, **base)

    def compute_integrations(self, *, branch=DEFAULT_BRANCH, note=""):
        pj = replay(self.project, branch)
        power = pj.material("power", "sample")
        schedule = pj.material("schedule", "sample")
        baseline = pj.material("baseline", "sample")
        if power is None or schedule is None:
            raise SemanticError("power and sample schedule are required")
        params = self._integration_params(pj, branch)
        results, injections, set_id = pipeline.run_integration(
            power, schedule, baseline, params)
        payload = {
            "set_id": set_id,
            "params_version": params.version,
            "results": [r.to_dict() for r in results],
            "injections": [pipeline.injection_dict(x) for x in injections],
            "power_material_id": power["material_id"],
            "baseline_material_id": (baseline["material_id"]
                                     if baseline else None),
            "schedule_material_id": schedule["material_id"],
        }
        payload["_inputs"] = self._integral_inputs(payload)
        ev = self.project.append(
            branch, INTEGRATION_COMPUTED, payload,
            inputs_digest=digest(payload["_inputs"]),
            parent_dependencies=[power["material_id"], schedule["material_id"]]
            + ([baseline["material_id"]] if baseline else []),
            note=note, epoch_us=now_us())
        self.save("compute integrations")
        return ev

    def _integral_inputs(self, payload):
        return {"set_id": payload["set_id"],
                "params_version": payload["params_version"],
                "power": payload["power_material_id"],
                "baseline": payload["baseline_material_id"],
                "schedule": payload["schedule_material_id"],
                "heats": [(r["injection_index"], r["valid"],
                           None if r["heat_ucal"] is None
                           else round(r["heat_ucal"], 9),
                           r["endpoint_reason"])
                          for r in payload["results"]]}

    # ---- blank + net --------------------------------------------------------
    def compute_blank_integrations(self, *, branch=DEFAULT_BRANCH, note=""):
        pj = replay(self.project, branch)
        blank_power = pj.material("blank", "control")
        blank_schedule = pj.material("schedule", "blank")
        if blank_power is None or blank_schedule is None:
            raise SemanticError("blank power and blank schedule are required")
        params = self._integration_params(pj, branch)
        results, injections, set_id = pipeline.run_blank_integration(
            {"power": blank_power, "schedule": blank_schedule,
             "baseline": pj.material("baseline", "blank")}, params)
        payload = {
            "set_id": set_id,
            "params_version": params.version,
            "results": [r.to_dict() for r in results],
            "injections": [pipeline.injection_dict(x) for x in injections],
            "blank_power_material_id": blank_power["material_id"],
            "blank_schedule_material_id": blank_schedule["material_id"],
        }
        payload["_inputs"] = {
            "set_id": set_id, "params_version": params.version,
            "blank_power": blank_power["material_id"],
            "blank_schedule": blank_schedule["material_id"],
            "heats": [(r["injection_index"], r["valid"],
                       None if r["heat_ucal"] is None
                       else round(r["heat_ucal"], 9), r["endpoint_reason"])
                      for r in payload["results"]]}
        ev = self.project.append(
            branch, BLANK_INTEGRATION_COMPUTED, payload,
            inputs_digest=digest(payload["_inputs"]),
            parent_dependencies=[blank_power["material_id"],
                                 blank_schedule["material_id"]],
            note=note, epoch_us=now_us())
        self.save("compute blank integrations")
        return ev

    def compute_net_heats(self, *, branch=DEFAULT_BRANCH, note=""):
        pj = replay(self.project, branch)
        sample_ev = pj.derived.get(INTEGRATION_COMPUTED)
        if sample_ev is None:
            raise SemanticError("sample integrations required first")
        from .integration import IntegrationResult
        from .parser import Injection
        sample_results = [_hydrate_result(r) for r in sample_ev["results"]]
        sample_injections = [_hydrate_injection(x)
                             for x in sample_ev["injections"]]
        blank_rule = pj.decisions.get(DECISION_BLANK_RULE)
        rule = blank_rule["rule"] if blank_rule else MATCH_BY_ORDINAL
        scale = blank_rule["scale_by_volume"] if blank_rule else True
        bp_version = blank_rule["event_id"] if blank_rule else "blank-default-v1"
        blank_ev = pj.derived.get(BLANK_INTEGRATION_COMPUTED)
        blank_results, blank_injections = [], []
        blank_set_id = None
        if blank_ev:
            blank_results = [_hydrate_result(r) for r in blank_ev["results"]]
            blank_injections = [_hydrate_injection(x)
                                for x in blank_ev["injections"]]
            blank_set_id = blank_ev["set_id"]
        bp = BlankParams(version=bp_version, rule=rule, scale_by_volume=scale)
        nets = pipeline.run_net(sample_results, blank_results,
                                sample_injections, blank_injections, bp)
        set_id = IdFactory.net_id(sample_ev["set_id"], bp_version, blank_set_id)
        for n in nets:
            n.stable_id = f"{set_id}:inj{n.injection_index}"
        payload = {
            "set_id": set_id,
            "blank_rule": rule,
            "scale_by_volume": scale,
            "blank_params_version": bp_version,
            "blank_set_id": blank_set_id,
            "sample_set_id": sample_ev["set_id"],
            "results": [n.to_dict() for n in nets],
        }
        payload["_inputs"] = {
            "set_id": set_id, "rule": rule, "scale": scale,
            "sample_set_id": sample_ev["set_id"],
            "blank_set_id": blank_set_id,
            "values": [(n.injection_index, n.valid,
                        None if n.net_ucal is None
                        or (isinstance(n.net_ucal, float)
                            and n.net_ucal != n.net_ucal)
                        else round(n.net_ucal, 9),
                        n.matched_blank_index) for n in nets]}
        deps = [sample_ev["set_id"]] + ([blank_set_id] if blank_set_id else [])
        ev = self.project.append(
            branch, NET_HEATS_COMPUTED, payload,
            inputs_digest=digest(payload["_inputs"]),
            parent_dependencies=deps, note=note, epoch_us=now_us())
        self.save("compute net heats")
        return ev

    # ---- model fits ---------------------------------------------------------
    def fit_models(self, *, branch=DEFAULT_BRANCH, equiopt_tol=0.10,
                   max_evals=12000, note=""):
        pj = replay(self.project, branch)
        net_ev = pj.derived.get(NET_HEATS_COMPUTED)
        cond = pj.material("conditions", "sample")
        conc = pj.material("concentration", "sample")
        if net_ev is None:
            raise SemanticError("net heats required first")
        if cond is None or conc is None:
            raise SemanticError("conditions and concentration materials "
                                "required for model fitting")
        discard_decision = pj.decisions.get(DECISION_FIRST_INJECTION)
        discard = bool(discard_decision["discard"]) if discard_decision else False
        net_by_idx = {r["injection_index"]: r for r in net_ev["results"]}
        inj_order = sorted(net_by_idx)
        heats = [net_by_idx[i]["net_ucal"] for i in inj_order]
        sched = pj.material("schedule", "sample")
        vol_by_idx = {x["index"]: x["volume_uL"]
                      for x in sched["summary"]["injections"]}
        volumes = [vol_by_idx.get(i) for i in inj_order]
        out = pipeline.run_fits(heats, volumes, cond, conc, discard,
                                equiopt_tol, max_evals, net_ev["set_id"],
                                net_ev["blank_params_version"])
        fits = out.pop("fits")
        for f in fits:
            f["stable_id"] = IdFactory.fit_id(
                f["model_id"], net_ev["set_id"], conc["material_id"],
                f["param_version"])
        payload = {"set_id": net_ev["set_id"],
                   "net_set_id": net_ev["set_id"],
                   "concentration_version": conc["version"],
                   "concentration_material_id": conc["material_id"],
                   "discard_first": discard,
                   "equiopt_tol": equiopt_tol, "status": out["status"],
                   "fits": fits, "diagnostics": out.get("diagnostics", []),
                   "used_indices_heat": out.get("used_indices_heat", []),
                   "cell_volume_uL": out.get("cell_volume_uL"),
                   "macromolecule_uM": out.get("macromolecule_uM"),
                   "ligand_uM": out.get("ligand_uM")}
        payload["_inputs"] = {
            "net_set_id": net_ev["set_id"],
            "concentration_material_id": conc["material_id"],
            "discard": discard, "tol": equiopt_tol,
            "fits": [(f["model_id"], f["status"],
                      None if f["sse"] is None else round(f["sse"], 9),
                      f["equi_optimal"]) for f in fits]}
        deps = [net_ev["set_id"], conc["material_id"],
                cond["material_id"]]
        ev = self.project.append(
            branch, MODELS_FITTED, payload,
            inputs_digest=digest(payload["_inputs"]),
            parent_dependencies=deps, note=note, epoch_us=now_us())
        self.save("fit models")
        return ev

    # ---- review actions -----------------------------------------------------
    def review(self, action, target_type, target_id, *, branch=DEFAULT_BRANCH,
               adjustment=None, note=""):
        type_map = {"accept": REVIEW_ACCEPT, "reject": REVIEW_REJECT,
                    "adjust": REVIEW_ADJUST}
        payload = {"action": action, "target_type": target_type,
                   "target_id": target_id,
                   "as_of_dependency_hashes": self._dependency_snapshot(branch),
                   "adjustment": adjustment or {}}
        ev = self.project.append(branch, type_map[action], payload, note=note,
                                 epoch_us=now_us())
        self.save(f"review {action} {target_type}")
        return ev

    def _dependency_snapshot(self, branch):
        pj = replay(self.project, branch)
        snap = {}
        for key, rec in pj.materials.items():
            snap[f"material:{key[0]}/{key[1]}"] = rec["material_id"]
        for t, d in pj.derived.items():
            snap[f"derived:{t}"] = {"set_id": d.get("set_id"),
                                    "inputs": d.get("_inputs")}
        return snap

    def create_branch(self, new_branch, base_branch=DEFAULT_BRANCH,
                      base_event_id=None, note=""):
        pj = replay(self.project, base_branch)
        if base_event_id is None:
            base_event_id = pj.events[-1].event_id if pj.events else None
        ev = self.project.create_branch(new_branch, base_branch,
                                        base_event_id, note)
        self.save("create branch")
        return ev

    # ---- views --------------------------------------------------------------
    def projection(self, branch=DEFAULT_BRANCH, as_of_event_id=None,
                   verify=False):
        return replay(self.project, branch, as_of_event_id, verify=verify)

    def staleness(self, branch=DEFAULT_BRANCH) -> dict:
        pj = replay(self.project, branch)
        stale = {}
        integ = pj.derived.get(INTEGRATION_COMPUTED)
        if integ and (integ["power_material_id"]
                      != pj.material("power", "sample")["material_id"]
                      or integ["schedule_material_id"]
                      != pj.material("schedule", "sample")["material_id"]):
            stale["integration"] = "STALE_POWER_OR_SCHEDULE_REVISED"
        fits = pj.derived.get(MODELS_FITTED)
        if fits and pj.material("concentration", "sample") and \
                fits["concentration_material_id"] != \
                pj.material("concentration", "sample")["material_id"]:
            stale["models"] = "STALE_CONCENTRATION_REVISED"
        if integ and not stale.get("integration"):
            stale["integration"] = "CURRENT"
        if fits and "models" not in stale:
            stale["models"] = "CURRENT"
        return stale


def _hydrate_result(d: dict):
    import math
    from .integration import IntegrationResult
    return IntegrationResult(
        injection_index=d["injection_index"],
        start_time_s=d["start_time_s"], end_time_s=d["end_time_s"],
        n_points=d["n_points"],
        heat_ucal=(math.nan if d["heat_ucal"] is None else d["heat_ucal"]),
        peak_power_ucal_s=(math.nan if d["peak_power_ucal_s"] is None
                           else d["peak_power_ucal_s"]),
        endpoint_reason=d["endpoint_reason"],
        overlap_next=d["overlap_next"], crosses_gap=d["crosses_gap"],
        valid=d["valid"], diagnostics=d.get("diagnostics", []),
        stable_id=d.get("stable_id", ""))


def _hydrate_injection(d: dict):
    import math
    from .parser import Injection
    return Injection(index=d["index"],
                     nominal_time_s=(math.nan if d["nominal_time_s"] is None
                                     else d["nominal_time_s"]),
                     raw_time=d["raw_time"], raw_time_unit=d["raw_time_unit"],
                     volume_uL=(math.nan if d["volume_uL"] is None
                                else d["volume_uL"]),
                     raw_volume=d["raw_volume"],
                     raw_volume_unit=d["raw_volume_unit"],
                     flags=d.get("flags", []))


class SemanticError(Exception):
    pass
