"""业务编排：所有业务规则在服务端执行，UI 只是视图。

事件类型：
  case_created / material_imported / branch_forked
  integration_policy_set / integrated
  blank_integrated / dilution_applied
  fitted / model_decision / injection_decision / parameter_adjustment / manual_decision / conflict_noted
"""
from __future__ import annotations

import json
import time

from . import dilution as dilution_mod
from . import integration as integration_mod
from . import models as models_mod
from .errors import BudgetExhausted, FormatCorrupt, ItcError, Outcome, SemanticUnresolved
from .identity import canonical_json, content_hash, short_hash, stable_id
from .importer import import_material
from .store import Store

MATERIAL_KINDS = ("power", "injections", "concentration", "temperature", "cell_volume", "baseline", "blank")


class Service:
    def __init__(self, store: Store, clock=time.time):
        self.store = store
        self.clock = clock

    def now(self) -> float:
        return self.clock()

    # ---------------------------------------------------------------- cases
    def create_case(self, name: str, title: str = "", case_id: str | None = None) -> Outcome:
        cid = case_id or stable_id("case", {"name": name, "title": title})
        self.store.create_case(cid, name, title, created_at=self.now())
        self.store.append_event(cid, f"br_{cid}_main", "case_created",
                                {"name": name, "title": title, "case_id": cid},
                                event_id=stable_id("evt", {"case": cid, "type": "case_created"}),
                                ts=self.now())
        return Outcome.ok({"case_id": cid, "branch_id": f"br_{cid}_main"})

    def fork_branch(self, case_id: str, name: str, parent_branch_id: str,
                    reason: str = "") -> Outcome:
        fork_event_id = self.store.head(parent_branch_id)
        bid = self.store.create_branch(case_id, name, parent_branch_id, fork_event_id,
                                       created_at=self.now())
        evt = self.store.append_event(case_id, bid, "branch_forked",
                                      {"name": name, "parent_branch": parent_branch_id,
                                       "fork_event_id": fork_event_id, "reason": reason},
                                      event_id=stable_id("evt", {"branch": bid, "type": "branch_forked"}),
                                      ts=self.now())
        return Outcome.ok({"branch_id": bid, "event": evt})

    # ---------------------------------------------------------------- import
    def import_material_event(self, case_id: str, branch_id: str, kind: str, raw_text: str,
                              lineage: str | None = None, source_label: str = "uploaded") -> Outcome:
        if kind not in MATERIAL_KINDS:
            return Outcome.fail(FormatCorrupt(f"未知材料类型 {kind!r}"))
        lineage = lineage if lineage is not None else kind
        try:
            parsed = import_material(kind, raw_text)
        except ItcError as exc:
            return Outcome.fail(exc)
        digest = content_hash({"kind": kind, "lineage": lineage, "raw": raw_text})
        material_id, version, created = self.store.put_material(
            case_id, kind, lineage, branch_id, raw_text, parsed, digest, created_at=self.now())
        payload = {
            "material_id": material_id, "kind": kind, "lineage": lineage,
            "version": version, "content_hash": digest,
            "source_label": source_label,
            "diagnostics": parsed.get("diagnostics", []),
            "summary": _material_summary(kind, parsed),
        }
        if created:
            self.store.append_event(case_id, branch_id, "material_imported", payload,
                                    material_id=material_id,
                                    event_id=stable_id("evt", {"branch": branch_id, "mat": material_id,
                                                              "ver": version}),
                                    ts=self.now())
        return Outcome.ok({"material_id": material_id, "version": version, "created": created,
                           "payload": payload})

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, branch_id: str, at_event_id: str | None = None) -> dict:
        """重放事件得到结构化状态。at_event_id 给出时只重放到该事件（含），
        且仅保留该事件所在分支链可见的事件 —— 回到旧事件即恢复当时依赖版本与结论。"""
        events = self.store.timeline(branch_id)
        if at_event_id is not None:
            target = next((e for e in events if e["event_id"] == at_event_id), None)
            if target is None:
                raise SemanticUnresolved("目标事件在该分支不可见", )
            events = [e for e in events
                      if (e["timestamp"], e["branch_id"], e["seq"]) <=
                      (target["timestamp"], target["branch_id"], target["seq"])]

        case_id = None
        heads: dict[str, dict] = {}
        materials_all: list[dict] = []
        results: list[dict] = []
        decisions: list[dict] = []
        policy = None
        correspondence = None
        replayed_events: list[dict] = []

        for e in events:
            p = e["payload"]
            case_id = case_id or self._case_of_branch(branch_id, e["branch_id"])
            replayed_events.append(_event_view(e))
            if e["type"] == "material_imported":
                mat = self.store.get_material(e["material_id"])
                rec = {"kind": p["kind"], "lineage": p["lineage"],
                       "material_id": e["material_id"], "version": p["version"],
                       "event_id": e["event_id"], "event_seq": e["seq"],
                       "source_label": p.get("source_label", ""),
                       "summary": p.get("summary", {}),
                       "diagnostics": p.get("diagnostics", []),
                       "raw_text": mat["raw_text"] if mat else "",
                       "parsed": mat["parsed"] if mat else None}
                materials_all = [m for m in materials_all
                                 if not (m["kind"] == rec["kind"] and m["lineage"] == rec["lineage"])]
                materials_all.append(rec)
                heads[(rec["kind"], rec["lineage"])] = rec
            elif e["type"] == "integration_policy_set":
                policy = p["policy"]
            elif e["type"] == "integrated":
                r = self.store.get_result(e["result_id"])
                results.append({"slot": "integration", **self._result_view(r, e, heads)})
            elif e["type"] == "blank_integrated":
                r = self.store.get_result(e["result_id"])
                results.append({"slot": "blank_integration", **self._result_view(r, e, heads)})
            elif e["type"] == "dilution_applied":
                r = self.store.get_result(e["result_id"])
                results.append({"slot": "dilution", **self._result_view(r, e, heads)})
            elif e["type"] == "fitted":
                r = self.store.get_result(e["result_id"])
                results.append({"slot": "fit", **self._result_view(r, e, heads)})
            elif e["type"] in ("model_decision", "injection_decision",
                               "parameter_adjustment", "manual_decision", "conflict_noted"):
                decisions.append({"event_id": e["event_id"], "type": e["type"],
                                  "seq": e["seq"], "timestamp": e["timestamp"],
                                  "branch_id": e["branch_id"], "payload": p})

        materials = {}
        for m in materials_all:
            materials.setdefault(m["kind"], []).append(m)
        latest: dict[str, dict] = {}
        for m in materials_all:
            if m["kind"] not in latest or m["version"] > latest[m["kind"]]["version"]:
                latest[m["kind"]] = m

        chain_results = self._annotate_freshness(results, latest)
        return {
            "branch_id": branch_id,
            "case_id": case_id,
            "events": replayed_events,
            "materials": materials,
            "latest_materials": {k: _mat_ref(v) for k, v in latest.items()},
            "policy": policy,
            "correspondence": correspondence,
            "results": chain_results,
            "decisions": decisions,
            "head_event_id": events[-1]["event_id"] if events else None,
        }

    def _case_of_branch(self, viewer_branch: str, event_branch: str) -> str | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT case_id FROM branches WHERE branch_id=?",
                               (event_branch,)).fetchone()
            return row["case_id"] if row else None

    def _result_view(self, r: dict | None, e: dict, heads: dict) -> dict:
        if r is None:
            return {"result_id": e["result_id"], "missing": True,
                    "event_id": e["event_id"], "seq": e["seq"],
                    "branch_id": e["branch_id"], "content": None}
        return {"result_id": r["result_id"], "kind": r["kind"], "event_id": e["event_id"],
                "seq": e["seq"], "timestamp": e["timestamp"], "branch_id": e["branch_id"],
                "parents": r["parents"], "inputs_digest": r["inputs_digest"],
                "content": r["content"]}

    def _annotate_freshness(self, results: list[dict], latest: dict[str, dict]) -> list[dict]:
        last_int = [r for r in results if r["slot"] == "integration"]
        last_blank = [r for r in results if r["slot"] == "blank_integration"]
        last_dil = [r for r in results if r["slot"] == "dilution"]
        last_fit = [r for r in results if r["slot"] == "fit"]

        def event_id_of(r):
            return r["event_id"]

        for r in last_fit:
            parents = r.get("parents", [])
            r["stale"] = False
            r["stale_reasons"] = []
            parent_events = [p for p in parents if p.startswith("evt_")]
            for pev_id in parent_events:
                pev = self.store.event_by_id(pev_id)
                if pev is None:
                    continue
                p = pev["payload"]
                if pev["type"] == "material_imported" and p.get("kind") in ("concentration",):
                    cur = latest.get("concentration")
                    if cur and cur["event_id"] != pev_id:
                        r["stale"] = True
                        r["stale_reasons"].append({
                            "code": "concentration_revision",
                            "message": "浓度修订：此拟合基于旧浓度，模型已过期；原积分不因此改变",
                            "used_event": pev_id, "current_event": cur["event_id"]})
        for r in results:
            r.setdefault("stale", False)
            r.setdefault("stale_reasons", [])
        return results

    # ---------------------------------------------------------------- compute
    def _head_material(self, branch_id: str, kind: str, at: dict | None = None):
        snap = self.snapshot(branch_id)
        ref = (at or {}).get(kind) or snap["latest_materials"].get(kind)
        if ref is None:
            return None
        return self.store.get_material(ref["material_id"])

    def set_integration_policy(self, case_id: str, branch_id: str, policy: dict) -> Outcome:
        evt = self.store.append_event(case_id, branch_id, "integration_policy_set",
                                      {"policy": policy}, ts=self.now())
        return Outcome.ok({"event": evt})

    def run_integration(self, case_id: str, branch_id: str,
                        power_id: str | None = None, injections_id: str | None = None,
                        baseline_id: str | None = None, policy: dict | None = None) -> Outcome:
        try:
            power = self._require(branch_id, "power", power_id)
            injs = self._require(branch_id, "injections", injections_id)
            base = self.store.get_material(baseline_id) if baseline_id else \
                self._head_material(branch_id, "baseline")
            snap = self.snapshot(branch_id)
            eff_policy = {**(snap["policy"] or {}), **(policy or {})} if snap["policy"] else policy
            run = integration_mod.integrate(power["parsed"], injs["parsed"],
                                            base["parsed"] if base else None, eff_policy)
        except ItcError as exc:
            return Outcome.fail(exc)
        parents = [power["material_id"], injs["material_id"]]
        if base:
            parents.append(base["material_id"])
        parents.append("policy:" + short_hash(run["policy"]))
        inputs = _run_inputs(run, parents, {"power": power["material_id"],
                                            "injections": injs["material_id"],
                                            "baseline": base["material_id"] if base else None,
                                            "policy": run["policy"]})
        rid = stable_id("res_int", {"inputs": inputs})
        self.store.put_result(rid, case_id, "integration", branch_id, run,
                              content_hash(inputs), parents, created_at=self.now())
        parent_events = self._material_event_ids(branch_id, [power, injs] + ([base] if base else []))
        evt = self.store.append_event(case_id, branch_id, "integrated",
                                      {"result_id": rid, "inputs": inputs,
                                       "parent_events": parent_events,
                                       "finding_count": len(run["findings"]),
                                       "gaps": run["gaps"]},
                                      result_id=rid, ts=self.now())
        return Outcome.ok({"result_id": rid, "event": evt, "run": run})

    def run_blank(self, case_id: str, branch_id: str, blank_id: str | None = None,
                  policy: dict | None = None) -> Outcome:
        try:
            blank = self._require(branch_id, "blank", blank_id)
            run = dilution_mod.integrate_blank_series(blank["parsed"], policy)
        except ItcError as exc:
            return Outcome.fail(exc)
        parents = [blank["material_id"], "policy:" + short_hash(policy or {})]
        rid = stable_id("res_blank", {"blank": blank["material_id"], "policy": policy or {}})
        self.store.put_result(rid, case_id, "blank_integration", branch_id, run,
                              content_hash({"p": parents}), parents, created_at=self.now())
        pev = self._material_event_ids(branch_id, [blank])
        evt = self.store.append_event(case_id, branch_id, "blank_integrated",
                                      {"result_id": rid, "blank_material_id": blank["material_id"],
                                       "parent_events": pev,
                                       "gaps": run.get("gaps", [])},
                                      result_id=rid, ts=self.now())
        return Outcome.ok({"result_id": rid, "event": evt, "run": run})

    def run_dilution(self, case_id: str, branch_id: str, integration_result_id: str | None = None,
                     blank_result_id: str | None = None, correspondence: dict | None = None) -> Outcome:
        snap = self.snapshot(branch_id)
        ints = [r for r in snap["results"] if r["slot"] == "integration"]
        blanks = [r for r in snap["results"] if r["slot"] == "blank_integration"]
        if not ints:
            return Outcome.fail(SemanticUnresolved(
                "还没有积分结果，无法扣除稀释热",
                [_diag("missing_integration", "先完成 integrated", "/results/integration")]))
        int_r = self.store.get_result(integration_result_id) if integration_result_id else \
            self.store.get_result(ints[-1]["result_id"])
        blank_r = self.store.get_result(blank_result_id) if blank_result_id else \
            (self.store.get_result(blanks[-1]["result_id"]) if blanks else None)
        corr = correspondence or {"rule": "by_index", "time_tolerance_s": 30.0, "volume_scale": True}
        try:
            if blank_r is None:
                raise SemanticUnresolved(
                    "空白积分缺失：没有可用的空白对照，无法建立对应",
                    [_diag("blank_correspondence_missing",
                           "先导入空白并完成 blank_integrated，或显式决定不扣除",
                           "/materials/blank")])
            run = dilution_mod.apply_dilution(int_r["content"], blank_r["content"], corr)
        except ItcError as exc:
            return Outcome.fail(exc)
        parents = [int_r["result_id"], blank_r["result_id"],
                   "corr:" + short_hash(corr)]
        rid = stable_id("res_dil", {"int": int_r["result_id"],
                                    "blank": blank_r["result_id"], "corr": corr})
        self.store.put_result(rid, case_id, "dilution", branch_id, run,
                              content_hash({"p": parents}), parents, created_at=self.now())
        evt = self.store.append_event(case_id, branch_id, "dilution_applied",
                                      {"result_id": rid, "integration_result_id": int_r["result_id"],
                                       "blank_result_id": blank_r["result_id"],
                                       "correspondence": corr,
                                       "finding_count": len(run["findings"])},
                                      result_id=rid, ts=self.now())
        return Outcome.ok({"result_id": rid, "event": evt, "run": run})

    def run_fit(self, case_id: str, branch_id: str, dilution_result_id: str | None = None,
                budget: dict | None = None) -> Outcome:
        snap = self.snapshot(branch_id)
        dils = [r for r in snap["results"] if r["slot"] == "dilution"]
        if not dils:
            return Outcome.fail(SemanticUnresolved(
                "还没有稀释扣除结果，无法拟合模型",
                [_diag("missing_dilution", "先完成 dilution_applied", "/results/dilution")]))
        dil_r = self.store.get_result(dilution_result_id) if dilution_result_id else \
            self.store.get_result(dils[-1]["result_id"])
        int_r = self.store.get_result(dil_r["parents"][0])
        try:
            conc = self._require(branch_id, "concentration")
            cellv = self._require(branch_id, "cell_volume")
            temp_mat = self._head_material(branch_id, "temperature")
            run = models_mod.fit_models(dil_r["content"], int_r["content"],
                                        conc["parsed"], cellv["parsed"],
                                        temp_mat["parsed"] if temp_mat else None, budget)
        except ItcError as exc:
            return Outcome.fail(exc)
        parents = [dil_r["result_id"], int_r["result_id"], conc["material_id"],
                   cellv["material_id"]]
        conc_evt = self._material_event_ids(branch_id, [conc])
        run["inputs"]["parent_events"] = {"concentration": conc_evt[0] if conc_evt else None}
        rid = stable_id("res_fit", {"dil": dil_r["result_id"], "int": int_r["result_id"],
                                    "conc": conc["material_id"], "cellv": cellv["material_id"],
                                    "budget": budget or {}})
        self.store.put_result(rid, case_id, "fit", branch_id, run,
                              content_hash({"p": parents}), parents, created_at=self.now())
        evt = self.store.append_event(case_id, branch_id, "fitted",
                                      {"result_id": rid, "dilution_result_id": dil_r["result_id"],
                                       "integration_result_id": int_r["result_id"],
                                       "concentration_material_id": conc["material_id"],
                                       "concentration_event_id": conc_evt[0] if conc_evt else None,
                                       "cell_volume_material_id": cellv["material_id"],
                                       "preferred_family": run["preferred_family"],
                                       "equivalent_families": run["equivalent_families"],
                                       "ambiguous": run["ambiguous"],
                                       "finding_count": len(run["findings"])},
                                      result_id=rid, ts=self.now())
        return Outcome.ok({"result_id": rid, "event": evt, "run": run})

    # ---------------------------------------------------------------- decisions
    def decide_model(self, case_id: str, branch_id: str, fit_result_id: str,
                     decision: str, family: str, note: str = "") -> Outcome:
        if decision not in ("accepted", "rejected"):
            return Outcome.fail(FormatCorrupt("decision 必须是 accepted/rejected"))
        fit = self.store.get_result(fit_result_id)
        if fit is None:
            return Outcome.fail(SemanticUnresolved("拟合结果不存在",
                [_diag("missing_result", fit_result_id, "/results")]))
        families = [c["family"] for c in fit["content"]["candidates"]]
        if family not in families:
            return Outcome.fail(SemanticUnresolved(
                f"候选中没有 {family}", [_diag("unknown_candidate", str(families), "/models")]))
        evt = self.store.append_event(case_id, branch_id, "model_decision", {
            "fit_result_id": fit_result_id, "decision": decision, "family": family,
            "note": note, "equivalent_families": fit["content"]["equivalent_families"],
        }, ts=self.now())
        return Outcome.ok({"event": evt})

    def decide_injection(self, case_id: str, branch_id: str, injection_id: str,
                         decision: str, reason: str) -> Outcome:
        if decision not in ("kept", "discarded", "flagged"):
            return Outcome.fail(FormatCorrupt("注射决定必须是 kept/discarded/flagged"))
        evt = self.store.append_event(case_id, branch_id, "injection_decision", {
            "injection_id": injection_id, "decision": decision, "reason": reason,
        }, ts=self.now())
        return Outcome.ok({"event": evt})

    def adjust_parameters(self, case_id: str, branch_id: str, fit_result_id: str,
                          family: str, params: dict, note: str) -> Outcome:
        fit = self.store.get_result(fit_result_id)
        if fit is None:
            return Outcome.fail(SemanticUnresolved("拟合结果不存在",
                [_diag("missing_result", fit_result_id, "/results")]))
        rid = stable_id("res_manual", {"fit": fit_result_id, "family": family, "params": params})
        candidate = next((c for c in fit["content"]["candidates"] if c["family"] == family), None)
        content = {"kind": "manual_model", "base_fit_result_id": fit_result_id,
                   "family": family, "params": params,
                   "param_units": candidate["param_units"] if candidate else {},
                   "note": note, "origin": "manual_adjustment"}
        self.store.put_result(rid, case_id, "fit", branch_id, content,
                              content_hash(content), [fit_result_id], created_at=self.now())
        evt = self.store.append_event(case_id, branch_id, "parameter_adjustment", {
            "result_id": rid, "base_fit_result_id": fit_result_id, "family": family,
            "params": params, "note": note,
        }, result_id=rid, ts=self.now())
        return Outcome.ok({"result_id": rid, "event": evt})

    def manual_decision(self, case_id: str, branch_id: str, topic: str, decision: str,
                        rationale: str, conflicts_with: str | None = None) -> Outcome:
        evt_type = "conflict_noted" if conflicts_with else "manual_decision"
        evt = self.store.append_event(case_id, branch_id, evt_type, {
            "topic": topic, "decision": decision, "rationale": rationale,
            "conflicts_with": conflicts_with,
        }, ts=self.now())
        return Outcome.ok({"event": evt})

    # ---------------------------------------------------------------- helpers
    def _require(self, branch_id: str, kind: str, material_id: str | None = None) -> dict:
        if material_id:
            mat = self.store.get_material(material_id)
            if mat is None:
                raise SemanticUnresolved(f"{kind} 材料 {material_id} 不存在",
                    [_diag("missing_material", material_id, f"/materials/{kind}")])
            return mat
        snap = self.snapshot(branch_id)
        ref = snap["latest_materials"].get(kind)
        if ref is None:
            raise SemanticUnresolved(f"缺少必需材料 {kind}",
                [_diag("missing_material", f"分支 {branch_id} 尚未导入 {kind}",
                       f"/materials/{kind}")])
        return self.store.get_material(ref["material_id"])

    def _material_event_ids(self, branch_id: str, mats: list[dict]) -> list[str]:
        snap = self.snapshot(branch_id)
        out = []
        by_id = {m["material_id"]: ev for ev in snap["events"]
                 for m in [ev["payload"]] if False}
        for ev in snap["events"]:
            if ev["type"] == "material_imported" and ev["payload"].get("material_id"):
                by_id[ev["payload"]["material_id"]] = ev["event_id"]
        for m in mats:
            mid = by_id.get(m["material_id"])
            if mid:
                out.append(mid)
        return out


def _diag(code: str, message: str, loc: str, value=None):
    from .errors import Diagnostic
    return Diagnostic(code, message, loc, value)


def _mat_ref(mat: dict) -> dict:
    return {"material_id": mat["material_id"], "kind": mat["kind"],
            "lineage": mat["lineage"], "version": mat["version"],
            "event_id": mat["event_id"], "summary": mat.get("summary", {})}


def _event_view(e: dict) -> dict:
    return {"event_id": e["event_id"], "branch_id": e["branch_id"], "seq": e["seq"],
            "parent_event_id": e.get("parent_event_id"), "type": e["type"],
            "timestamp": e["timestamp"], "actor": e.get("actor", "reviewer"),
            "payload": e["payload"]}


def _material_summary(kind: str, parsed: dict) -> dict:
    if kind in ("power", "blank"):
        return {"samples": parsed.get("valid_sample_count"),
                "diagnostic_count": len(parsed.get("diagnostics", [])),
                "time_unit": parsed.get("meta", {}).get("time_unit_declared"),
                "power_unit": parsed.get("meta", {}).get("power_unit_declared")}
    if kind == "baseline":
        return {"mode": parsed.get("mode"),
                "samples": parsed.get("valid_sample_count"),
                "diagnostic_count": len(parsed.get("diagnostics", []))}
    if kind == "injections":
        return {"records": parsed.get("valid_count"),
                "diagnostic_count": len(parsed.get("diagnostics", []))}
    if kind == "concentration":
        return {"syringe": parsed["syringe"]["raw"], "cell": parsed["cell"]["raw"]}
    if kind == "temperature":
        return {"raw": parsed["temperature"]["raw"]}
    if kind == "cell_volume":
        return {"raw": parsed["cell_volume"]["raw"]}
    return {}


def _run_inputs(run: dict, parents: list[str], labels: dict) -> dict:
    return {"parents": parents, "labels": labels,
            "inputs_summary": run.get("inputs", {}),
            "policy": run.get("policy"),
            "gaps": run.get("gaps", [])}
