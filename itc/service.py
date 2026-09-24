"""服务编排层：业务规则的唯一入口（浏览器只调用它）。

事件溯源：每次动作产生机器可读取事件；结果记录输入摘要/参数版本/父级依赖。
浓度修订使模型过期但不改原积分；局部基线改动只影响落入局部区间的注射。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .blanks import apply_blank
from .errors import BudgetExhausted, FormatCorrupt, SemanticInvalid
from .integration import integrate_run
from .materials import (BASELINE, BLANK, CONCENTRATIONS, CONDITIONS, MATERIAL_KINDS,
                        POWER, SCHEDULE, Material, make_material, material_health,
                        parse_bundle)
from .models import fit_models
from .rules import (BlankRules, IntegrationRules, ModelRules, RULE_SET_BLANK,
                    RULE_SET_INTEGRATION, RULE_SET_MODEL)
from .store import Store
from .units import assert_semantics


class ProjectService:
    def __init__(self, store: Store, project_id: str = "alpha", branch: str = "main") -> None:
        self.store = store
        self.project_id = project_id
        self.branch = branch

    # ------------------------------------------------ 导入与材料修订
    def import_bundle(self, bundle: dict[str, Any], *, source_ref: str = "alpha",
                      revision: int = 1, branch: str | None = None,
                      note: str = "初始导入") -> list[Material]:
        branch = branch or self.branch
        try:
            materials = parse_bundle(bundle, source_ref=source_ref, revision=revision)
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatCorrupt(f"原始 ITC 包结构损坏：{exc}", {"source_ref": source_ref}) from exc
        diagnostics = material_health(materials)
        assert_semantics(diagnostics, "材料语义检查未通过")

        op = self.store.begin_op(self.project_id)
        payload = {
            "source_ref": source_ref,
            "revision": revision,
            "material_ids": {k: m.id for k, m in materials.items()},
            "units": {
                "time_basis": "s", "power_basis": "cal/s", "heat_basis": "cal",
                "conc_basis": "M", "volume_basis": "L", "temperature_basis": "K",
            },
            "tokens": {k: m.tokens for k, m in materials.items() if m.tokens},
            "note": note,
        }
        for m in materials.values():
            self.store.put_material(self.project_id, op, m.to_dict())
        event = self.store.append_event(self.project_id, op, branch, "materials_imported",
                                        payload, self._parent(branch))
        self.store.commit_op(op)
        self._after_event(event)
        return list(materials.values())

    def revise_material(self, kind: str, new_canonical_raw: Any, name: str, *,
                        branch: str | None = None, note: str = "") -> Material:
        """不可变材料修订：创建新版本（新身份），旧版本保留。"""
        branch = branch or self.branch
        if kind not in MATERIAL_KINDS:
            raise SemanticInvalid("未知材料类别", [])
        existing = [m for m in self.store.materials(self.project_id) if m["kind"] == kind]
        existing.sort(key=lambda m: m["revision"])
        rev = (existing[-1]["revision"] + 1) if existing else 1
        source_ref = existing[-1]["source_ref"] if existing else kind
        mat = make_material(kind, source_ref, rev, name, new_canonical_raw, new_canonical_raw)
        op = self.store.begin_op(self.project_id)
        self.store.put_material(self.project_id, op, mat.to_dict())
        affected = self._stale_dependents(kind, mat)
        event = self.store.append_event(self.project_id, op, branch, "material_revised", {
            "kind": kind,
            "revision": rev,
            "material_id": mat.id,
            "supersedes": existing[-1]["id"] if existing else None,
            "stale_effect": affected,
            "note": note,
        }, self._parent(branch))
        self.store.commit_op(op)
        self._after_event(event)
        return mat

    def _stale_dependents(self, kind: str, mat: Material) -> dict[str, Any]:
        """声明源改动的影响范围（结构化，可重放）。"""
        latest = self._latest_materials()
        if kind == CONCENTRATIONS:
            return {"integration": "unchanged", "model_fit": "stale",
                    "rule": "浓度修订使绑定模型过期，但不改变原积分",
                    "affected_results": [r["id"] for r in self.store.results(self.project_id)
                                         if r["kind"] == "model_fit"]}
        if kind in (POWER, SCHEDULE, BASELINE, BLANK):
            if kind == BASELINE:
                affected_injections = self._baseline_affected(mat, latest)
            else:
                affected_injections = "all"
            return {"integration": "stale", "model_fit": "stale",
                    "scope": kind,
                    "affected_injections": affected_injections}
        return {"integration": "stale", "model_fit": "stale", "scope": kind}

    def _baseline_affected(self, mat: Material, latest: dict[str, dict]) -> list[int]:
        """局部基线修订：只标记锚点时间落入局部修订区间的注射。"""
        anchors = mat.canonical["anchors"] if isinstance(mat.canonical, dict) else mat["canonical"]
        # 修订区间：非零偏移锚点覆盖的最小/最大时间
        if not latest or SCHEDULE not in latest:
            return "all"
        ts = [a["t"] for a in anchors]
        lo, hi = min(ts), max(ts)
        out = []
        for inj in latest[SCHEDULE]["canonical"]["injections"]:
            if lo <= inj["t"] <= hi:
                out.append(inj["index"])
        return out

    # ------------------------------------------------ 积分 / 空白
    def run_integration(self, rules: IntegrationRules, *, branch: str | None = None,
                        is_blank: bool = False, note: str = "") -> dict[str, Any]:
        branch = branch or self.branch
        materials = self._material_objects(self._latest_materials())
        outcome = integrate_run(materials, rules, is_blank=is_blank)
        op = self.store.begin_op(self.project_id)
        self.store.put_rule(self.project_id, op, RULE_SET_INTEGRATION, rules.version,
                            rules.id, {"id": rules.id, "rule_set": RULE_SET_INTEGRATION,
                                       "version": rules.version, "content": rules.content()})
        result = outcome.to_dict()
        self.store.put_result(self.project_id, op, result)
        kind_act = "blank_integrated" if is_blank else "injection_integrated"
        event = self.store.append_event(self.project_id, op, branch, kind_act, {
            "result_id": outcome.id,
            "rule_id": rules.id,
            "rule_version": rules.version,
            "parents": outcome.parent_ids,
            "input_summary": outcome.input_summary,
            "is_blank": is_blank,
            "diagnostic_codes": sorted({d["code"] for d in outcome.diagnostics} |
                                       {d["code"] for r in outcome.injections for d in r["diagnostics"]}),
            "note": note,
        }, self._parent(branch))
        self.store.commit_op(op)
        self._after_event(event)
        return result

    def correct_blank(self, blank_result_id: str, rules: BlankRules, *,
                      branch: str | None = None, note: str = "") -> dict[str, Any]:
        branch = branch or self.branch
        latest = self._latest_materials()
        materials = self._material_objects(latest)
        from .integration import IntegrationOutcome
        tit = self.store.result_by_id(self._head_integration(branch))
        blank = self.store.result_by_id(blank_result_id)
        if blank is None:
            raise SemanticInvalid("空白积分结果不存在", [])
        outcome_t = self._outcome_from_dict(tit)
        outcome_b = self._outcome_from_dict(blank)
        corrected = apply_blank(outcome_t, outcome_b, materials, rules)
        op = self.store.begin_op(self.project_id)
        self.store.put_rule(self.project_id, op, RULE_SET_BLANK, rules.version, rules.id,
                            {"id": rules.id, "rule_set": RULE_SET_BLANK, "version": rules.version,
                             "content": rules.content()})
        result = corrected.to_dict()
        self.store.put_result(self.project_id, op, result)
        event = self.store.append_event(self.project_id, op, branch, "blank_corrected", {
            "result_id": corrected.id,
            "titration_result_id": tit["id"],
            "blank_result_id": blank_result_id,
            "rule_id": rules.id,
            "rule_version": rules.version,
            "parents": corrected.parent_ids,
            "input_summary": corrected.input_summary,
            "correspondence": rules.correspondence,
            "missing_correspondence": [r["index"] for r in corrected.injections
                                       if any(d["code"] == "blank_correspondence_missing"
                                              for d in r["diagnostics"])],
            "note": note,
        }, self._parent(branch))
        self.store.commit_op(op)
        self._after_event(event)
        return result

    # ------------------------------------------------ 模型
    def fit(self, rules: ModelRules, integration_result_id: str | None = None, *,
            branch: str | None = None, budget_override: int | None = None,
            note: str = "") -> dict[str, Any]:
        branch = branch or self.branch
        integration = self.store.result_by_id(integration_result_id or self._head_corrected(branch))
        if integration is None:
            raise SemanticInvalid("没有可供拟合的稀释校正积分", [])
        materials = self._material_objects(self._latest_materials())
        try:
            outcome = fit_models(integration, materials, rules, budget_override=budget_override)
        except BudgetExhausted as exc:
            op = self.store.begin_op(self.project_id)
            event = self.store.append_event(self.project_id, op, branch, "fit_budget_exhausted", {
                "rule_version": rules.version,
                "budget": exc.budget, "used": exc.used,
                "integration_result_id": integration["id"],
                "note": note or exc.message,
            }, self._parent(branch))
            self.store.commit_op(op)
            self._after_event(event)
            raise
        op = self.store.begin_op(self.project_id)
        self.store.put_rule(self.project_id, op, RULE_SET_MODEL, rules.version, rules.id,
                            {"id": rules.id, "rule_set": RULE_SET_MODEL, "version": rules.version,
                             "content": rules.content()})
        result = outcome.to_dict()
        self.store.put_result(self.project_id, op, result)
        event = self.store.append_event(self.project_id, op, branch, "models_fitted", {
            "result_id": outcome.id,
            "rule_id": rules.id,
            "rule_version": rules.version,
            "parents": outcome.parent_ids,
            "input_summary": outcome.input_summary,
            "candidate_models": sorted({c["model"] for c in outcome.candidates}),
            "tied_models": [c["model"] for c in outcome.selected],
            "n_candidates": len(outcome.candidates),
            "note": note,
        }, self._parent(branch))
        self.store.commit_op(op)
        self._after_event(event)
        return result

    # ------------------------------------------------ 决定 / 分支
    def decide(self, action_label: str, decision: str, target_id: str, *,
               branch: str | None = None, rationale: str = "",
               params: dict[str, Any] | None = None, conflict_with: str | None = None) -> dict[str, Any]:
        """接受 / 否定 / 参数调整：都产生新事件与决定，绝不覆盖旧结果。"""
        branch = branch or self.branch
        if decision not in ("accepted", "rejected", "parameter_adjusted", "conflict"):
            raise SemanticInvalid("未知决定类型", [])
        op = self.store.begin_op(self.project_id)
        event = self.store.append_event(self.project_id, op, branch, f"decision_{action_label}", {
            "decision": decision,
            "target_id": target_id,
            "rationale": rationale,
            "params": params or {},
            "conflict_with": conflict_with,
        }, self._parent(branch))
        self.store.put_decision(self.project_id, op, event, {
            "decision": decision, "target_id": target_id, "rationale": rationale,
            "params": params or {}, "conflict_with": conflict_with,
        })
        self.store.commit_op(op)
        self._after_event(event)
        return event

    def create_branch(self, name: str, *, from_branch: str | None = None) -> str:
        from_branch = from_branch or self.branch
        op = self.store.begin_op(self.project_id)
        parent = self.store.conn.execute(
            "SELECT head_event_id FROM branches WHERE project_id=? AND name=?",
            (self.project_id, from_branch)).fetchone()
        parent_id = parent["head_event_id"] if parent else None
        self.store.conn.execute(
            "INSERT OR REPLACE INTO branches (project_id, name, head_event_id, op_id) VALUES (?,?,?,?)",
            (self.project_id, name, parent_id, op))
        event = self.store.append_event(self.project_id, op, name, "branch_created",
                                        {"from_branch": from_branch, "parent_event_id": parent_id}, None)
        self.store.commit_op(op)
        self._after_event(event)
        return name

    # ------------------------------------------------ 物化辅助
    def _parent(self, branch: str) -> str | None:
        row = self.store.conn.execute(
            "SELECT head_event_id FROM branches WHERE project_id=? AND name=?",
            (self.project_id, branch)).fetchone()
        return row["head_event_id"] if row else None

    def _after_event(self, event: dict[str, Any]) -> None:
        # 过期传播在“读取当前状态”时计算，保证历史事件重放不被最新结果覆盖
        return None

    def _latest_materials(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for m in self.store.materials(self.project_id):
            if m["kind"] not in out or m["revision"] > out[m["kind"]]["revision"]:
                out[m["kind"]] = m
        return out

    @staticmethod
    def _material_objects(latest: dict[str, dict]) -> dict[str, Material]:
        objs: dict[str, Material] = {}
        for kind, d in latest.items():
            objs[kind] = Material(
                kind=d["kind"], source_ref=d["source_ref"], revision=d["revision"],
                name=d["name"], raw=d["raw"], canonical=d["canonical"],
                tokens=d.get("tokens", []), id=d["id"])
        return objs

    @staticmethod
    def _outcome_from_dict(d: dict[str, Any]):
        from .integration import IntegrationOutcome
        return IntegrationOutcome(
            id=d["id"], injections=d["injections"], diagnostics=d["diagnostics"],
            input_summary=d["input_summary"], parent_ids=d["parent_ids"],
            rule_version=d["rule_version"], rule_id=d["rule_id"],
            accepted=d.get("accepted", False))

    def _head_integration(self, branch: str) -> str | None:
        for ev in reversed(self.store.events(self.project_id, branch)):
            if ev["action"] in ("injection_integrated", "blank_corrected"):
                return ev["payload"]["result_id"]
        return None

    def _head_corrected(self, branch: str) -> str | None:
        for ev in reversed(self.store.events(self.project_id, branch)):
            if ev["action"] == "blank_corrected":
                return ev["payload"]["result_id"]
            if ev["action"] == "injection_integrated":
                return ev["payload"]["result_id"]
        return None

    def _head_fit(self, branch: str) -> str | None:
        for ev in reversed(self.store.events(self.project_id, branch)):
            if ev["action"] == "models_fitted":
                return ev["payload"]["result_id"]
        return None

    # ------------------------------------------------ 当前视图（含过期传播）
    def current_view(self, branch: str | None = None) -> dict[str, Any]:
        branch = branch or self.branch
        latest = self._latest_materials()
        int_id = self._head_integration(branch)
        fit_id = self._head_fit(branch)
        results = {r["id"]: r for r in self.store.results(self.project_id)}
        integration = results.get(int_id) if int_id else None
        fit = results.get(fit_id) if fit_id else None

        # 过期传播：基于最新材料与结果父级依赖
        fit_status = None
        if fit is not None:
            parent_set = set(fit["parent_ids"])
            conc_id = latest.get(CONCENTRATIONS, {}).get("id")
            integration_parent = fit["input_summary"].get("integration_result")
            stale_reasons = []
            if conc_id and conc_id not in parent_set:
                stale_reasons.append("concentration_revised")
            if integration_parent and integration_parent != int_id and integration is not None:
                stale_reasons.append("upstream_integration_superseded")
            fit = dict(fit)
            fit["stale"] = bool(stale_reasons)
            fit["stale_reason"] = stale_reasons
            fit_status = "stale" if stale_reasons else "current"

        decisions = {d["event_id"]: d for d in self.store.decisions(self.project_id)}
        return {
            "project_id": self.project_id,
            "branch": branch,
            "branches": [b["name"] for b in self.store.branches(self.project_id)],
            "materials": [latest[k] for k in MATERIAL_KINDS if k in latest],
            "material_versions": {
                k: [{"id": m["id"], "revision": m["revision"], "name": m["name"]}
                    for m in sorted((x for x in self.store.materials(self.project_id) if x["kind"] == k),
                                    key=lambda x: x["revision"])]
                for k in MATERIAL_KINDS
            },
            "integration": integration,
            "integration_status": "current" if integration else None,
            "fit": fit,
            "fit_status": fit_status,
            "events": self.store.events(self.project_id, branch),
            "decisions": list(decisions.values()),
            "head_event_id": self._parent(branch),
        }

    # ------------------------------------------------ 历史重放
    def state_at_event(self, event_id: str, branch: str) -> dict[str, Any]:
        """回到任意旧事件：恢复当时的依赖版本与可见结论，不用最新结果覆盖。"""
        events = self.store.events(self.project_id, branch)
        upto = None
        for ev in events:
            if ev["id"] == event_id:
                upto = ev
                break
        if upto is None:
            raise SemanticInvalid("该分支上找不到事件", [])
        history = events[:upto["seq"]]
        results = {r["id"]: r for r in self.store.results(self.project_id)}
        materials_all = self.store.materials(self.project_id)

        # 当时可见材料：每个 kind 截至该事件的最新修订
        visible_mats: dict[str, dict] = {}
        int_id = None
        fit_id = None
        for ev in history:
            if ev["action"] == "materials_imported":
                for kind, mid in ev["payload"]["material_ids"].items():
                    m = next((x for x in materials_all if x["id"] == mid), None)
                    if m:
                        visible_mats[kind] = m
            elif ev["action"] == "material_revised":
                m = next((x for x in materials_all if x["id"] == ev["payload"]["material_id"]), None)
                if m:
                    visible_mats[m["kind"]] = m
            elif ev["action"] in ("injection_integrated", "blank_corrected"):
                int_id = ev["payload"]["result_id"]
            elif ev["action"] == "models_fitted":
                fit_id = ev["payload"]["result_id"]

        return {
            "project_id": self.project_id,
            "branch": branch,
            "at_event": {"id": upto["id"], "seq": upto["seq"], "action": upto["action"]},
            "materials": [visible_mats[k] for k in MATERIAL_KINDS if k in visible_mats],
            "integration": results.get(int_id) if int_id else None,
            "fit": results.get(fit_id) if fit_id else None,
            "events_through": history,
        }

    # ------------------------------------------------ 导出 / 导入
    def export(self) -> dict[str, Any]:
        return self.store.export_project(self.project_id)

    @classmethod
    def import_snapshot(cls, store: Store, snapshot: dict[str, Any]) -> "ProjectService":
        pid = store.import_project(snapshot)
        svc = cls(store, project_id=pid)
        return svc
