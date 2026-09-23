"""Orchestration: versioned material import, computations with recorded
input digests / parameter versions / parent dependencies, decisions,
staleness, and as-of views.  All business rules live here server-side.
"""
from __future__ import annotations

from . import integrate as integ
from . import fit as fitmod
from .store import Store, digest, canonical
from .units import normalize_series, normalize_quantity

STATUS_OK = "ok"
FAILURE_KINDS = ("budget_exhausted", "format_corrupt", "semantic_unsolvable")

# which material kinds each computation kind depends on
DEPENDS = {
    "integration": ("power_sampling", "injection_schedule", "baseline"),
    "blank_correction": ("blank_curve", "injection_schedule"),
    "fit": ("concentrations", "temperature", "cell_volume",
            "injection_schedule"),
}


class Service:
    def __init__(self, store: Store):
        self.store = store

    # ---------------- materials ----------------
    def import_material(self, kind, payload, branch="main"):
        """Version a material.  Raw payload is stored verbatim; the derived
        (normalized) form is computed once and stored alongside; values
        that fail conversion stay in place with diagnostics."""
        diagnostics = []
        derived = self._normalize(kind, payload, diagnostics)
        with self.store.transaction(f"import {kind}") as conn:
            uid, version, seq = self.store.add_material(
                kind, payload, derived, branch=branch, conn=conn)
            self.store.add_diagnostics(diagnostics, conn=conn, seq=seq)
            staled = self._stale_computations(kind)
            if staled:
                self.store._event(conn, "computations_staled", {
                    "reason": f"{kind} revised to v{version}",
                    "computation_uids": staled,
                    "rule": "revision_stales_dependents_preserves_parents",
                }, branch)
        return {"uid": uid, "version": version, "seq": seq,
                "diagnostics": diagnostics, "staled": staled}

    def _normalize(self, kind, payload, diags):
        if kind == "power_sampling":
            return {
                "time_s": normalize_series("time", payload["time"]["values"],
                                           payload["time"]["unit"],
                                           "power.time", diags),
                "power_uJ_s": normalize_series(
                    "power", payload["power"]["values"],
                    payload["power"]["unit"], "power.power", diags),
                "gaps": [
                    {"start_s": normalize_quantity(
                        "time", g["start"], f"gaps[{i}].start", diags),
                     "end_s": normalize_quantity(
                         "time", g["end"], f"gaps[{i}].end", diags),
                     "reason": g.get("reason", "")}
                    for i, g in enumerate(payload.get("gaps", []))],
                "time_basis": payload["time"].get("basis", "unspecified"),
            }
        if kind == "injection_schedule":
            out = []
            for i, inj in enumerate(payload["injections"]):
                out.append({
                    "time_s": normalize_quantity(
                        "time", inj["time"], f"injections[{i}].time", diags),
                    "volume_uL": normalize_quantity(
                        "volume", inj["volume"], f"injections[{i}].volume",
                        diags),
                })
            return {"injections": out}
        if kind == "concentrations":
            return {
                "cell_mM": normalize_quantity(
                    "concentration", payload["cell"], "concentrations.cell",
                    diags),
                "syringe_mM": normalize_quantity(
                    "concentration", payload["syringe"],
                    "concentrations.syringe", diags),
            }
        if kind == "temperature":
            return {"K": normalize_quantity("temperature", payload,
                                            "temperature", diags)}
        if kind == "cell_volume":
            return {"uL": normalize_quantity("volume", payload,
                                             "cell_volume", diags)}
        if kind == "baseline":
            coeffs = payload.get("coefficients", [])
            return {"kind": payload.get("kind", "constant"),
                    "coefficients": normalize_series(
                        "power", coeffs, payload.get("unit", "uJ/s"),
                        "baseline.coefficients", diags),
                    "t0": payload.get("t0", 0.0)}
        if kind == "blank_curve":
            return {
                "heats_uJ": normalize_series(
                    "heat", payload["heats"]["values"],
                    payload["heats"]["unit"], "blank.heats", diags),
            }
        raise ValueError(f"unknown material kind {kind!r}")

    # ---------------- computations ----------------
    def _parents(self, kinds):
        parents = []
        for kind in kinds:
            m = self.store.latest_material(kind)
            if m is None:
                return None, kind
            parents.append([kind, m["version"], m["uid"]])
        return parents, None

    def _stale_computations(self, changed_kind):
        stale = []
        for c in self.store.computations():
            if any(p[0] == changed_kind for p in c["parents"]):
                stale.append(c["uid"])
        return stale

    def run_integration(self, params=None, branch="main"):
        parents, missing = self._parents(DEPENDS["integration"])
        if missing:
            return self._incomplete("integration", missing)
        power = self.store.latest_material("power_sampling")["derived"]
        sched = self.store.latest_material("injection_schedule")["derived"]
        baseline = self.store.latest_material("baseline")["derived"]
        diagnostics = []
        injections = [{"index": i, "time_s": j["time_s"],
                       "volume_uL": j["volume_uL"]}
                      for i, j in enumerate(sched["injections"])]
        results, judgements = integ.integrate_injections(
            {"time": power["time_s"], "power": power["power_uJ_s"],
             "gaps": [{"start_s": g["start_s"], "end_s": g["end_s"],
                       "reason": g["reason"]} for g in power["gaps"]]},
            injections, baseline, params, diagnostics)
        p = dict(integ.DEFAULT_PARAMS)
        if params:
            p.update(params)
        return self._record_computation(
            "integration", parents, integ.PARAM_VERSION, p,
            {"injections": results, "units": {"heat": "uJ", "time": "s"},
             "time_basis": power["time_basis"]},
            diagnostics, judgements, branch)

    def run_blank_correction(self, integration_uid=None, params=None,
                             branch="main"):
        parents, missing = self._parents(DEPENDS["blank_correction"])
        if missing:
            return self._incomplete("blank_correction", missing)
        integ_comp = self._latest_computation("integration", integration_uid)
        if integ_comp is None:
            return self._incomplete("blank_correction", "integration")
        blank = self.store.latest_material("blank_curve")["derived"]
        diagnostics = []
        corrected, judgements = integ.blank_correct(
            integ_comp["result"]["injections"], blank["heats_uJ"], params,
            diagnostics)
        parents = parents + [["computation", "integration",
                              integ_comp["uid"]]]
        p = dict(integ.DEFAULT_PARAMS)
        if params:
            p.update(params)
        return self._record_computation(
            "blank_correction", parents, integ.PARAM_VERSION, p,
            {"injections": corrected, "units": {"heat": "uJ"}},
            diagnostics, judgements, branch)

    def run_fit(self, blank_uid=None, models=None, budget=20000,
                discarded=None, branch="main"):
        parents, missing = self._parents(DEPENDS["fit"])
        if missing:
            return self._incomplete("fit", missing)
        blank_comp = self._latest_computation("blank_correction", blank_uid)
        if blank_comp is None:
            return self._incomplete("fit", "blank_correction")
        conc = self.store.latest_material("concentrations")["derived"]
        vol = self.store.latest_material("cell_volume")["derived"]
        sched = self.store.latest_material("injection_schedule")["derived"]
        temp = self.store.latest_material("temperature")["derived"]
        diagnostics = []
        for pos, val in (("concentrations.cell", conc["cell_mM"]),
                         ("concentrations.syringe", conc["syringe_mM"]),
                         ("cell_volume", vol["uL"]),
                         ("temperature", temp["K"])):
            if val is None:
                diagnostics.append({
                    "severity": "error", "position": pos,
                    "code": "input_incomplete",
                    "message": f"{pos} has no convertible value; fit "
                               "cannot proceed"})
        if diagnostics:
            return self._record_computation(
                "fit", parents, fitmod.PARAM_VERSION, {},
                None, diagnostics, [], branch, status="diagnostic")
        discarded = sorted(discarded if discarded is not None
                           else self._discarded_injections())
        injections = blank_comp["result"]["injections"]
        observed = []
        for r in injections:
            if r["index"] in discarded:
                observed.append(None)
            else:
                observed.append(r.get("corrected_heat_uJ"))
        cond = fitmod.cell_conditions(
            [{"volume_uL": j["volume_uL"] or 0.0}
             for j in sched["injections"]],
            conc["cell_mM"], conc["syringe_mM"], vol["uL"])
        result = fitmod.fit_models(observed, cond, vol["uL"],
                                   models=models, budget=budget)
        diagnostics.extend(result.pop("diagnostics"))
        parents = (parents + [["computation", "blank_correction",
                               blank_comp["uid"]],
                              ["decisions", self._decision_seq(), ""]])
        result["discarded_injections"] = discarded
        result["temperature_K"] = temp["K"]
        result["conditions"] = cond
        result["observed_uJ"] = observed
        return self._record_computation(
            "fit", parents, fitmod.PARAM_VERSION,
            {"models": models or ["one_site", "two_site", "dilution_const"],
             "budget": budget},
            result, diagnostics, [], branch, status=result["status"])

    # ---------------- decisions ----------------
    def decide(self, target, action, value=None, rationale="", branch="main"):
        """Record a review decision.  Conflicting decisions on the same
        target are both preserved; the earlier one is superseded, never
        deleted."""
        conflicts = [d for d in self.store.decisions(branch=branch)
                     if d["target"] == target and d["status"] == "adopted"]
        with self.store.transaction(f"decide {action} {target}") as conn:
            for d in conflicts:
                conn.execute(
                    "UPDATE decisions SET status='superseded' WHERE uid=?",
                    (d["uid"],))
            if conflicts:
                self.store._event(conn, "decision_conflict", {
                    "target": target,
                    "superseded": [d["uid"] for d in conflicts],
                    "rule": "conflicting_decisions_both_retained",
                }, branch)
            uid, seq = self.store.add_decision(
                target, action, value, rationale, "adopted",
                branch=branch, conn=conn)
            staled = []
            if target.startswith("injection:"):
                staled = [c["uid"] for c in self.store.computations()
                          if c["kind"] == "fit"]
                if staled:
                    self.store._event(conn, "computations_staled", {
                        "reason": f"decision on {target}",
                        "computation_uids": staled,
                        "rule": "decision_change_stales_fits_only",
                    }, branch)
        return {"uid": uid, "seq": seq, "superseded": [d["uid"] for d in
                                                       conflicts],
                "staled": staled}

    # ---------------- views ----------------
    def state(self, at_seq=None, branch=None):
        materials = []
        for m in self.store.materials(at_seq):
            materials.append({"kind": m["kind"], "version": m["version"],
                              "uid": m["uid"], "seq": m["seq"],
                              "payload": m["payload"], "derived": m["derived"]})
        comps = []
        for c in self.store.computations(at_seq=at_seq, branch=branch):
            comps.append({**c, "stale": self._is_stale(c, at_seq)})
        decisions = [d for d in self.store.decisions(at_seq=at_seq,
                                                     branch=branch)]
        return {"materials": materials, "computations": comps,
                "decisions": decisions,
                "events": self.store.events(at_seq=at_seq),
                "diagnostics": self.store.diagnostics(),
                "seq": self.store.max_seq() if at_seq is None else at_seq}

    def view_at_event(self, seq):
        """Reconstruct the visible state as of an event: dependency
        versions and conclusions of that moment, not latest results."""
        return self.state(at_seq=seq)

    def _is_stale(self, comp, at_seq=None):
        latest = {m["kind"]: m["version"]
                  for m in self.store.materials(at_seq)}
        dec_seq = self.store.max_seq() if at_seq is None else at_seq
        for p in comp["parents"]:
            if p[0] == "decisions":
                if self.store._decision_seq_after(p[1], dec_seq):
                    return True
            elif p[0] == "computation":
                pass
            elif p[0] in latest and latest[p[0]] > p[1]:
                return True
        return False

    # ---------------- helpers ----------------
    def _record_computation(self, kind, parents, param_version, params,
                            result, diagnostics, judgements, branch,
                            status="ok"):
        if diagnostics and result is None:
            status = "diagnostic"
        input_digest = digest({"parents": parents, "params": params,
                               "param_version": param_version})
        with self.store.transaction(f"compute {kind}") as conn:
            uid, seq = self.store.add_computation(
                kind, input_digest, param_version, parents, status, result,
                branch=branch, conn=conn)
            self.store.add_diagnostics(diagnostics, computation_uid=uid,
                                       conn=conn, seq=seq)
            for j in judgements:
                self.store._event(conn, "judgment",
                                  {**j, "computation_uid": uid}, branch)
        return {"uid": uid, "seq": seq, "status": status, "result": result,
                "diagnostics": diagnostics}

    def _incomplete(self, kind, missing):
        return {"uid": None, "seq": None, "status": "diagnostic",
                "result": None,
                "diagnostics": [{
                    "severity": "error", "position": missing,
                    "code": "input_incomplete",
                    "message": f"required input {missing!r} missing; "
                               "nothing computed"}]}

    def _latest_computation(self, kind, uid=None):
        comps = self.store.computations(kind=kind)
        if uid:
            comps = [c for c in comps if c["uid"] == uid]
        comps = [c for c in comps if c["status"] in
                 ("ok", "budget_exhausted")]
        return comps[-1] if comps else None

    def _discarded_injections(self):
        out = []
        for d in self.store.decisions():
            if (d["target"].startswith("injection:")
                    and d["action"] == "discard"
                    and d["status"] == "adopted"):
                out.append(int(d["target"].split(":")[1]))
        return out

    def _decision_seq(self):
        seqs = [d["seq"] for d in self.store.decisions()]
        return max(seqs) if seqs else 0
