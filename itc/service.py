"""Application services: orchestrate import -> decisions -> integration -> fit.

Every operation appends events through a write batch and records input
summaries, parameter versions and parent dependencies, so every derived result
is replayable and auditable.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from . import binding, importer, integration
from .materials import MaterialVersion
from .storage import Store, sid
from .view import FIT_INPUT_KINDS, build_view

PARSERS = {
    "power": lambda t, o, v: importer.import_series("power", t, o, v),
    "baseline": lambda t, o, v: importer.import_series("baseline", t, o, v),
    "blank": lambda t, o, v: importer.import_series("blank", t, o, v),
    "schedule": importer.import_schedule,
    "blank_schedule": importer.import_schedule,
    "temperature": lambda t, o, v: importer.import_scalar("temperature", "temperature", t, o, v),
    "cell_volume": lambda t, o, v: importer.import_scalar("cell_volume", "volume", t, o, v),
    "concentration": importer.import_concentration,
}


@dataclass
class OperationResult:
    events: list[dict]
    diagnostics: list[dict]


class Workflow:
    def __init__(self, store: Store):
        self.store = store

    # ---- view ------------------------------------------------------------
    def view(self, project_id: str, branch: str = "main", *, upto: str | None = None) -> dict:
        branches = {b["branch_id"]: b for b in self.store.list_branches(project_id)}
        if branch not in branches:
            raise KeyError(f"branch {branch} not found")
        events = self.store.branch_events(project_id, branch, upto=upto)
        return build_view(events, project_id=project_id, branch_id=branch,
                          branch_name=branches[branch]["name"], upto=upto)

    # ---- project / materials --------------------------------------------
    def create_project(self, name: str, project_id: str | None = None,
                       seed: dict | None = None) -> str:
        project_id = project_id or sid("prj_", name)
        self.store.create_project(project_id, name, seed=seed)
        return project_id

    def import_material(self, project_id: str, kind: str, text: str, origin: str,
                        *, branch: str = "main", note: str = "") -> OperationResult:
        state = self._head_state(project_id, branch)
        version = len(state.materials.get(kind, [])) + 1
        material, diagnostics = PARSERS[kind](text, origin, version)
        # blank schedule is stored as a schedule-kind material with an extra
        # role so the blank curve stays a separately versioned material
        if kind == "blank_schedule":
            material = MaterialVersion(
                kind="blank_schedule", version=version, origin=origin,
                injections=material.injections)
        self.store.begin_batch(project_id)
        ids = []
        ev = self.store.append_event(project_id, branch, "material_edit", {
            "material": material.to_dict(), "diagnostics": diagnostics,
            "note": note})
        ids.append(ev)
        self.store.commit_batch()
        return OperationResult(ids, diagnostics)

    def _head_state(self, project_id: str, branch: str):
        from .projector import replay

        events = self.store.branch_events(project_id, branch)
        return replay(events, project_id=project_id)

    def revise_concentration(self, project_id: str, *, syringe_m: float | None = None,
                             cell_m: float | None = None, origin: str,
                             raw_syringe: str | None = None, raw_cell: str | None = None,
                             branch: str = "main", rationale: str = "",
                             basis: list | None = None) -> OperationResult:
        """Concentration revision appends a new material version only.

        Existing integrals keep their parents (power/baseline/schedule) and
        therefore stay numerically intact; binding fits become stale.
        """
        state = self._head_state(project_id, branch)
        current = state.material("concentration")
        if current is None:
            from .errors import SemanticImpossible

            raise SemanticImpossible("cannot revise concentration before first import")
        syringe_m = current.extra.get("syringe") if syringe_m is None else syringe_m
        cell_m = current.extra.get("cell") if cell_m is None else cell_m
        version = len(state.materials["concentration"]) + 1
        material = MaterialVersion(
            kind="concentration", version=version, origin=origin,
            quantity=syringe_m, unit="M",
            raw=raw_syringe or (f"{syringe_m} M"),
            extra={"syringe": syringe_m, "cell": cell_m,
                   "raw_syringe": raw_syringe, "raw_cell": raw_cell, "unit": "M"})
        self.store.begin_batch(project_id)
        ids = [self.store.append_event(project_id, branch, "material_edit", {
            "material": material.to_dict(),
            "change": {"kind": "concentration_revision",
                       "from_version": current.version, "to_version": version},
            "impact": ["fitting"], "diagnostics": []})]
        self.store.commit_batch()
        return OperationResult(ids, [])

    # ---- decisions -------------------------------------------------------
    def discard_shots(self, project_id: str, shots: list[int], *, branch: str = "main",
                      rationale: str, basis: list | None = None) -> OperationResult:
        return self._decision(project_id, branch, "discard_shots", {
            "shots": sorted(shots), "rationale": rationale,
            "basis": basis or [], "scope": {"applies_to": "fitting"}})

    def set_blank_rule(self, project_id: str, rule: str, *, branch: str = "main",
                       rationale: str, basis: list | None = None) -> OperationResult:
        if rule not in ("by_index", "volume_proportional", "none"):
            from .errors import SemanticImpossible

            raise SemanticImpossible(f"unknown blank rule {rule!r}")
        return self._decision(project_id, branch, "blank_rule", {
            "rule": rule, "rationale": rationale, "basis": basis or [],
            "scope": {"applies_to": "integration"}})

    def set_endpoints(self, project_id: str, shot_index: int, start_s: float,
                      end_s: float, *, branch: str = "main", rationale: str,
                      basis: list | None = None) -> OperationResult:
        return self._decision(project_id, branch, "integration_endpoints", {
            "shot_index": shot_index, "start_s": start_s, "end_s": end_s,
            "time_basis": "experiment_seconds", "rationale": rationale,
            "basis": basis or [],
            "scope": {"applies_to": "integration", "shot": shot_index}})

    def manual_decision(self, project_id: str, decision: str, outcome: str,
                        *, branch: str = "main", rationale: str,
                        basis: list | None = None, conflicts_with: str | None = None,
                        parameters: dict | None = None) -> OperationResult:
        """Accept/reject/adjust decision. A conflicting decision stays on the
        record instead of being erased or overwritten."""
        return self._decision(project_id, branch, "manual_decision", {
            "decision": decision, "outcome": outcome, "rationale": rationale,
            "basis": basis or [], "conflicts_with": conflicts_with,
            "parameters": parameters or {},
            "scope": {"applies_to": "review"}})

    def review_note(self, project_id: str, text: str, *, branch: str = "main",
                    basis: list | None = None) -> OperationResult:
        return self._decision(project_id, branch, "review_note",
                              {"text": text, "basis": basis or []})

    def _decision(self, project_id: str, branch: str, etype: str,
                  payload: dict) -> OperationResult:
        self.store.begin_batch(project_id)
        ids = [self.store.append_event(project_id, branch, etype, payload)]
        self.store.commit_batch()
        return OperationResult(ids, [])

    # ---- derived computations -------------------------------------------
    def integrate(self, project_id: str, *, branch: str = "main",
                  pre_s: float = 2.0, post_s: float = 120.0,
                  max_gap_s: float = 5.0, blank_rule: str = "by_index",
                  max_points: int = 200000, note: str = "") -> OperationResult:
        state = self._head_state(project_id, branch)
        power = state.material("power")
        schedule = state.material("schedule")
        baseline = state.material("baseline")
        if power is None or schedule is None:
            from .errors import SemanticImpossible

            raise SemanticImpossible(
                "integration needs power curve and injection schedule",
                diagnostics=[{"code": "missing_input",
                              "missing": [k for k, v in (("power", power),
                                                          ("schedule", schedule))
                                          if v is None]}])
        injections = sorted(schedule.injections, key=lambda i: i.index)
        times = {i.index: i.time_s for i in injections}
        next_after = {injections[k].index: (injections[k + 1].time_s
                                            if k + 1 < len(injections) else None)
                      for k in range(len(injections))}
        endpoints = self._current(state).get("endpoints", {})
        shots, diagnostics = [], []
        for inj in injections:
            override = None
            dep = endpoints.get(inj.index)
            if dep:
                p = dep["payload"]
                override = (p["start_s"], p["end_s"])
            try:
                shot = integration.integrate_shot(
                    inj.index, inj.time_s, next_after[inj.index], power, baseline,
                    pre_s=pre_s, post_s=post_s, max_gap_s=max_gap_s,
                    endpoint_override=override, max_points=max_points)
                if dep:
                    shot["endpoint_decision"] = dep["event_id"]
                shots.append(shot)
            except Exception as exc:
                diagnostics.append(getattr(exc, "to_dict", lambda: {
                    "kind": exc.__class__.__name__, "message": str(exc)})())
        blank = state.material("blank")
        blank_schedule = state.material("blank_schedule")
        blanks = integration.blank_integrals(
            blank, blank_schedule.injections if blank_schedule else (),
            pre_s=pre_s, post_s=post_s, baseline=baseline,
            max_gap_s=max_gap_s)
        sample_vol = injections[0].volume_l if injections else None
        blank_vol = None
        if blank_schedule and blank_schedule.injections:
            blank_vol = sorted(blank_schedule.injections, key=lambda i: i.index)[0].volume_l
        shots = integration.apply_blank_rule(
            shots, blanks, blank_rule, volume_l=sample_vol or 0.0,
            blank_volume_l=blank_vol)
        integration.heat_per_mole(shots, schedule.injections)
        parents = {
            "power": self._head_event_id(state, "power"),
            "baseline": self._head_event_id(state, "baseline"),
            "schedule": self._head_event_id(state, "schedule"),
            "blank": self._head_event_id(state, "blank"),
            "blank_schedule": self._head_event_id(state, "blank_schedule"),
        }
        decision_parents = {d["type"]: d["event_id"] for d in state.decisions
                            if d["type"] in ("blank_rule", "integration_endpoints")}
        settings = {"pre_s": pre_s, "post_s": post_s, "max_gap_s": max_gap_s,
                    "blank_rule": blank_rule, "max_points": max_points,
                    "decision_parents": decision_parents,
                    "time_basis": "experiment_seconds",
                    "power_unit": "W", "integral_unit": "J"}
        summary = {"n_shots": len(injections), "n_integrated": len(shots),
                   "n_diagnostics": len(diagnostics),
                   "power_origin": power.origin, "schedule_origin": schedule.origin,
                   "baseline_origin": baseline.origin if baseline else None}
        self.store.begin_batch(project_id)
        ids = [self.store.append_event(project_id, branch, "integrated", {
            "settings": settings, "input_summary": summary, "parents": parents,
            "shots": shots, "diagnostics": diagnostics, "note": note})]
        self.store.commit_batch()
        return OperationResult(ids, diagnostics)

    @staticmethod
    def _current(state):
        from .view import _current_decisions

        return _current_decisions(state)

    @staticmethod
    def _head_event_id(state, kind: str) -> str | None:
        ids = state.material_event_ids.get(kind)
        return ids[-1] if ids else None

    def fit(self, project_id: str, *, branch: str = "main",
            models: tuple = ("one_site", "two_site", "dilution_only"),
            max_iter: int = 60, point_budget: int = 4_000_000,
            integration_id: str | None = None, note: str = "") -> OperationResult:
        state = self._head_state(project_id, branch)
        required = ("concentration", "temperature", "cell_volume", "schedule")
        missing = [k for k in required if state.material(k) is None]
        if missing:
            from .errors import SemanticImpossible

            raise SemanticImpossible(
                "binding fit missing required materials",
                diagnostics=[{"code": "missing_input", "missing": missing,
                              "positions": [{"material": k} for k in missing]}])
        if not state.integrations:
            from .errors import SemanticImpossible

            raise SemanticImpossible(
                "binding fit needs an integration first",
                diagnostics=[{"code": "missing_input", "missing": ["integration"]}])
        target = next((i for i in reversed(state.integrations)
                       if i.event_id == integration_id), None) \
            or state.integrations[-1]
        decisions = self._current(state)
        discarded = set(decisions.get("discard_shots", {}).get("payload", {}).get("shots", []))
        shots = [s for s in target.shots
                 if s["index"] not in discarded
                 and not any(d.get("code") == "blank_correspondence_missing"
                             for d in s.get("diagnostics", []))
                 and s.get("blank_status") in (None, "applied")
                 and not s.get("gap_terminated")]
        if len(shots) < 4:
            from .errors import SemanticImpossible

            raise SemanticImpossible(
                "fewer than four usable shots after exclusions",
                diagnostics=[{"code": "insufficient_shots",
                              "usable": len(shots),
                              "positions": [{"shot": s["index"]}
                                            for s in target.shots if s not in shots]}])
        schedule = state.material("schedule")
        volumes = {i.index: i.volume_l for i in schedule.injections}
        conc = state.material("concentration").extra
        temp_k = state.material("temperature").quantity
        cell_v = state.material("cell_volume").quantity
        all_indices = sorted(volumes)
        vol_all = np.array([volumes[i] for i in all_indices], dtype=float)
        indices = [s["index"] for s in shots]
        y = np.array([s["net_j"] for s in shots], dtype=float)
        result = binding.fit_candidates(
            y, indices, vol_all, cell_volume=cell_v,
            macromolecule=conc["cell"], ligand_syringe=conc["syringe"],
            temperature_k=temp_k, models=models, max_iter=max_iter,
            point_budget=point_budget)
        parents = {k: self._head_event_id(state, k) for k in
                   ("concentration", "temperature", "cell_volume", "schedule")}
        decision_parents = {f"discard:{s}": self._decision_event(state, "discard_shots")
                            for s in ()}
        disc = decisions.get("discard_shots")
        if disc:
            decision_parents["discard_shots"] = disc["event_id"]
        blank = decisions.get("blank_rule")
        if blank:
            decision_parents["blank_rule"] = blank["event_id"]
        settings = {"models": list(models), "max_iter": max_iter,
                    "point_budget": point_budget,
                    "discarded_shots": sorted(discarded),
                    "decision_parents": decision_parents,
                    "units": {"heat": "J", "volume": "L", "concentration": "M",
                              "temperature": "K", "enthalpy": "J/mol"}}
        summary = {"n_used_shots": len(shots), "used_shots": indices,
                   "syringe_m": conc["syringe"], "cell_m": conc["cell"],
                   "temperature_k": temp_k, "cell_volume_l": cell_v,
                   "integration_id": target.event_id}
        payload = {"integration_id": target.event_id, "settings": settings,
                   "input_summary": summary, "parents": parents,
                   "candidates": result["candidates"],
                   "diagnostics": result["diagnostics"],
                   "status": result["status"], "note": note}
        self.store.begin_batch(project_id)
        ids = [self.store.append_event(project_id, branch, "fitted", payload)]
        self.store.commit_batch()
        return OperationResult(ids, result["diagnostics"])

    @staticmethod
    def _decision_event(state, etype: str) -> str | None:
        for d in reversed(state.decisions):
            if d["type"] == etype:
                return d["event_id"]
        return None

    def patch_power(self, project_id: str, patches: list[dict], *, origin: str,
                    branch: str = "main", rationale: str = "",
                    basis: list | None = None) -> OperationResult:
        """Apply a local patch to power samples only.

        Only windows whose [start,end] intersects a patched time are affected;
        schedule and concentrations are untouched. Unconvertible patch values
        invalidate the affected points with diagnostics rather than NaNs.
        """
        state = self._head_state(project_id, branch)
        current = state.material("power")
        if current is None:
            from .errors import SemanticImpossible

            raise SemanticImpossible("cannot patch power before import")
        rows = current.series.all_rows()
        diagnostics = []
        affected_times = sorted(float(p["time_s"]) for p in patches)
        for p in patches:
            hit = False
            for row in rows:
                if abs(row[0] - float(p["time_s"])) < 1e-9:
                    try:
                        val = float(p["power_w"])
                    except (TypeError, ValueError):
                        diagnostics.append({"code": "unconvertible_value",
                                            "origin": origin, "raw": str(p.get("power_w")),
                                            "dimension": "power",
                                            "at": {"time_s": p["time_s"]}})
                        row[1] = float("nan")
                        row[3] = str(p.get("power_w"))
                        row[4] = False
                        hit = True
                        break
                    row[1] = val
                    row[3] = str(p["power_w"])
                    row[4] = True
                    hit = True
                    break
            if not hit:
                diagnostics.append({"code": "patch_target_missing", "origin": origin,
                                    "at": {"time_s": p["time_s"]}})
        series = current.series
        object.__setattr__(series, "points", rows)
        version = len(state.materials["power"]) + 1
        material = MaterialVersion(
            kind="power", version=version, origin=origin, series=series)
        self.store.begin_batch(project_id)
        ids = [self.store.append_event(project_id, branch, "material_edit", {
            "material": material.to_dict(),
            "change": {"kind": "local_power_patch",
                       "affected_times_s": affected_times,
                       "from_version": current.version},
            "impact": ["integrations:windows_intersecting_patch"],
            "diagnostics": diagnostics, "rationale": rationale,
            "basis": basis or []})]
        self.store.commit_batch()
        return OperationResult(ids, diagnostics)

    # ---- branches --------------------------------------------------------
    def fork(self, project_id: str, name: str, *, from_event_id: str,
             branch_id: str | None = None) -> str:
        return self.store.create_branch(project_id, name,
                                        from_event_id=from_event_id,
                                        branch_id=branch_id)

    def branches(self, project_id: str) -> list[dict]:
        return self.store.list_branches(project_id)

    def events_at(self, project_id: str, branch: str, upto: str) -> dict:
        """View reconstructed as-of an old event (history is not overwritten)."""
        return self.view(project_id, branch, upto=upto)
