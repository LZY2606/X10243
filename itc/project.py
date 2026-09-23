"""项目服务层：不可变材料版本 + 可修改决定分层；事件流可重放、可回溯。

不变量：
  * 材料（功率/时序/浓度/温度/池体积/基线/空白）只追加新版本，不改旧版。
  * 所有计算结果记录 input_digest、param_version 与父级依赖 deps。
  * 浓度修订使绑定拟合过期（stale），但积分不依赖浓度 -> 原积分不变。
  * 决定（接受/否定/调整参数）是独立事件；冲突决定落到不同分支，全部保留。
  * 回溯到任意旧事件：按该 seq 可见的材料版本与结论重建视图。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from . import binding, integrate
from .errors import Budget, Diagnostic, Outcome

MATERIAL_KINDS = [
    "power", "schedule", "concentrations", "temperature",
    "cell_volume", "baseline", "blank",
]
DEFAULT_BRANCH = "main"


def digest(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ref(material_id: str, version: int) -> str:
    return f"material:{material_id}@v{version}"


class Project:
    def __init__(self, store):
        self.store = store

    # ---- 材料版本 ------------------------------------------------------

    def import_material(self, kind: str, payload: dict[str, Any],
                        raw: str = "", branch: str = DEFAULT_BRANCH
                        ) -> tuple[str, int, int]:
        if kind not in MATERIAL_KINDS:
            raise ValueError(f"未知材料类型 {kind}")
        latest = self.store.latest_material(kind)
        version = 1 if latest is None else latest["version"] + 1
        with self.store.write_group(f"import-{kind}") as g:
            g.insert("materials", {
                "material_id": kind, "version": version, "kind": kind,
                "payload": json.dumps(payload, ensure_ascii=False,
                                      sort_keys=True),
                "raw": raw, "created_seq": 0})
            seq = self.store.append_event(
                g, branch, "material.import",
                {"kind": kind, "ref": ref(kind, version),
                 "digest": digest(payload),
                 "basis": payload.get("basis", {})},
                [self.store.head(branch)])
            # created_seq 回填
            self.store.conn.execute(
                "UPDATE materials SET created_seq=? WHERE material_id=? AND version=?",
                (seq, kind, version))
        return kind, version, seq

    def revise_material(self, kind: str, payload: dict[str, Any],
                        raw: str = "", reason: str = "",
                        branch: str = DEFAULT_BRANCH) -> tuple[str, int, int]:
        kind, version, seq = self.import_material(kind, payload, raw, branch)
        with self.store.write_group("revise-note") as g:
            self.store.append_event(
                g, branch, "material.revise",
                {"kind": kind, "ref": ref(kind, version),
                 "previous_ref": ref(kind, version - 1) if version > 1 else None,
                 "reason": reason, "digest": digest(payload)},
                [self.store.head(branch)])
        return kind, version, seq

    def current_at(self, kind: str, at_seq: int | None = None,
                   branch: str = DEFAULT_BRANCH) -> tuple[int, dict[str, Any]] | None:
        visible = {r["seq"] for r in self.visible_events(branch, at_seq)}
        rows = [r for r in self.store.all_materials()
                if r["material_id"] == kind and r["created_seq"] in visible]
        rows.sort(key=lambda r: r["version"])
        if not rows:
            return None
        r = rows[-1]
        return r["version"], json.loads(r["payload"])

    def visible_events(self, branch: str, at_seq: int | None = None) -> list:
        # 分支可见事件：父分支上 parent_seq 之前的事件 + 本分支事件
        brow = next((b for b in self.store.branches()
                     if b["branch_id"] == branch), None)
        rows = self.store.events()
        if brow is None:
            return [r for r in rows][:0]
        if brow["parent_branch"]:
            allowed = [r for r in rows
                       if r["branch"] == brow["parent_branch"]
                       and r["seq"] <= brow["parent_seq"]]
            allowed += [r for r in rows if r["branch"] == branch]
        else:
            allowed = [r for r in rows if r["branch"] == branch]
        allowed.sort(key=lambda r: r["seq"])
        if at_seq is not None:
            allowed = [r for r in allowed if r["seq"] <= at_seq]
        return allowed

    # ---- 积分 ----------------------------------------------------------

    def run_integration(self, params: integrate.IntegrationParams | None = None,
                        budget: Budget | None = None,
                        branch: str = DEFAULT_BRANCH) -> Outcome:
        params = params or integrate.IntegrationParams()
        budget = budget or Budget()
        need = ["power", "schedule", "baseline"]
        cur = {k: self.current_at(k, branch=branch) for k in need}
        if any(v is None for v in cur.values()):
            missing = [k for k, v in cur.items() if v is None]
            return Outcome.fail("semantic_unsolvable", [Diagnostic(
                "missing_material",
                f"积分缺少必需材料 {missing}，拒绝以默认值继续",
                f"integration.{missing}")])
        blank_cur = self.current_at("blank", branch=branch)
        deps = {
            "power": ["power", cur["power"][0]],
            "schedule": ["schedule", cur["schedule"][0]],
            "baseline": ["baseline", cur["baseline"][0]],
            "blank": ["blank", blank_cur[0]] if blank_cur else None,
        }
        out = integrate.integrate(
            cur["power"][1], cur["schedule"][1], cur["baseline"][1],
            blank_cur[1] if blank_cur else None,
            ref("blank", blank_cur[0]) if blank_cur else None,
            params, budget)
        if not out.ok:
            return out
        payload = out.value
        input_digest = digest({
            "deps": deps, "params": params.to_dict(),
            "inputs": self._input_summary(deps),
        })
        result_id = f"integration:{input_digest[:14]}"
        if self.store.get_result(result_id) is None:
            with self.store.write_group("integrate") as g:
                g.insert("results", {
                    "result_id": result_id, "kind": "integration",
                    "payload": json.dumps(payload, ensure_ascii=False,
                                          sort_keys=True),
                    "deps": json.dumps(deps, ensure_ascii=False, sort_keys=True),
                    "input_digest": input_digest,
                    "param_version": params.param_version,
                    "created_seq": 0, "branch": branch})
                seq = self.store.append_event(
                    g, branch, "integrate",
                    {"result_id": result_id, "input_digest": input_digest,
                     "param_version": params.param_version, "deps": deps,
                     "input_summary": self._input_summary(deps),
                     "rule_events": payload["rule_events"]},
                    [self.store.head(branch)])
                self.store.conn.execute(
                    "UPDATE results SET created_seq=? WHERE result_id=?",
                    (seq, result_id))
        return Outcome.good({"result_id": result_id, "payload": payload},
                            out.diagnostics)

    def _input_summary(self, deps: dict[str, Any]) -> dict[str, Any]:
        """计算结果记录输入摘要（行数、起止、单位基准）。"""
        summary: dict[str, Any] = {}
        for name, spec in deps.items():
            if not spec:
                summary[name] = None
                continue
            mat = self.store.material_at(spec[0], spec[1])
            p = json.loads(mat["payload"])
            item = {"ref": ref(spec[0], spec[1]), "basis": p.get("basis", {})}
            if "t" in p and p.get("t") is not None:
                item["n_samples"] = len(p["t"])
                item["t_range_s"] = [p["t"][0], p["t"][-1]]
            if "injections" in p:
                item["n_injections"] = len(p["injections"])
            summary[name] = item
        return summary

    # ---- 拟合 ----------------------------------------------------------

    def _conditions(self, integration_payload: dict[str, Any],
                    at_branch: str) -> tuple[binding.Conditions | None,
                                             dict[str, Any]]:
        need = ["concentrations", "cell_volume", "temperature"]
        cur = {k: self.current_at(k, branch=at_branch) for k in need}
        if any(v is None for v in cur.values()):
            return None, {k: None if v is None else v[0]
                          for k, v in cur.items()}
        vols = [float(r["volume_uL"])
                for r in integration_payload["injections"]]
        cond = binding.Conditions(
            cell_mM=cur["concentrations"][1]["cell_mM"],
            syringe_mM=cur["concentrations"][1]["syringe_mM"],
            V0_uL=cur["cell_volume"][1]["uL"],
            T_K=cur["temperature"][1]["K"],
            volumes_uL=vols)
        deps_versions = {k: v[0] for k, v in cur.items()}
        return cond, deps_versions

    def fit_models(self, kinds: list[str] | None = None,
                   budget: Budget | None = None,
                   integration_id: str | None = None,
                   branch: str = DEFAULT_BRANCH) -> Outcome:
        budget = budget or Budget()
        kinds = kinds or ["one_site", "two_site", "dilution"]
        integ_row = self._latest_integration(branch) if integration_id is None \
            else self.store.get_result(integration_id)
        if integ_row is None:
            return Outcome.fail("semantic_unsolvable", [Diagnostic(
                "no_integration", "尚无积分结果，无法拟合", "fit")])
        integ = json.loads(integ_row["payload"])
        cond, cond_vers = self._conditions(integ, branch)
        if cond is None:
            return Outcome.fail("semantic_unsolvable", [Diagnostic(
                "missing_material",
                f"拟合缺少条件材料 {[k for k,v in cond_vers.items() if v is None]}",
                "fit.conditions")])
        deps = {
            "integration": [integ_row["result_id"], None],
            "concentrations": ["concentrations", cond_vers["concentrations"]],
            "cell_volume": ["cell_volume", cond_vers["cell_volume"]],
            "temperature": ["temperature", cond_vers["temperature"]],
        }
        fits, diagnostics = [], []
        for kind in kinds:
            out = binding.fit_one(kind, integ, cond, budget)
            if not out.ok:
                # 单个候选失败也要留下状态，但不阻断其它候选；整体语义无解时整体失败
                diagnostics.extend(out.diagnostics)
                fits.append({"model": kind, "status": out.status,
                             "failure": out.diagnostics[0].to_dict()
                             if out.diagnostics else None})
                continue
            fits.append(out.value)
        real = [f for f in fits if f.get("kind") == "fit"]
        if not real:
            return Outcome.fail(
                "semantic_unsolvable" if not any(
                    f.get("status") == "budget_exhausted" for f in fits)
                else "budget_exhausted", diagnostics)
        binding.mark_equivalent(real)
        input_digest = digest({"deps": deps, "fits": [
            {k: f[k] for k in ("model", "params", "chi2")} for f in real]})
        result_id = f"fit-set:{input_digest[:14]}"
        if self.store.get_result(result_id) is None:
            with self.store.write_group("fit") as g:
                g.insert("results", {
                    "result_id": result_id, "kind": "fit-set",
                    "payload": json.dumps(
                        {"fits": fits}, ensure_ascii=False, sort_keys=True),
                    "deps": json.dumps(deps, ensure_ascii=False, sort_keys=True),
                    "input_digest": input_digest,
                    "param_version": binding.PARAM_VERSION,
                    "created_seq": 0, "branch": branch})
                seq = self.store.append_event(
                    g, branch, "fit",
                    {"result_id": result_id, "input_digest": input_digest,
                     "param_version": binding.PARAM_VERSION, "deps": deps,
                     "models": [f["model"] for f in fits],
                     "equivalent_groups": [
                         f.get("equivalent_group") for f in real
                         if f.get("equivalent")]},
                    [self.store.head(branch)])
                self.store.conn.execute(
                    "UPDATE results SET created_seq=? WHERE result_id=?",
                    (seq, result_id))
        return Outcome.good({"result_id": result_id, "fits": fits},
                            diagnostics)

    def _latest_integration(self, branch: str):
        rows = [r for r in self.store.all_results()
                if r["kind"] == "integration" and r["branch"] == branch]
        return rows[-1] if rows else None

    # ---- 决定与冲突分支 ------------------------------------------------

    def decide(self, target: str, verdict: str, reason: str = "",
               reviewer: str = "reviewer",
               branch: str = DEFAULT_BRANCH) -> dict[str, Any]:
        """记录接受/否定/调整参数。同分支对同一目标出现互斥结论 -> 自动开分支。"""
        assert verdict in ("accept", "reject", "adjust"), verdict
        prior = self._prior_decisions(branch, target)
        conflict = any(p["verdict"] != verdict for p in prior)
        target_branch = branch
        forked = False
        if conflict:
            target_branch = f"review-{uuid.uuid4().hex[:6]}"
            parent_seq = self.store.head(branch)
            self.store.create_branch(
                target_branch, target_branch, branch, parent_seq)
            forked = True
        with self.store.write_group("decide") as g:
            seq = self.store.append_event(
                g, target_branch, "decide",
                {"target": target, "verdict": verdict, "reason": reason,
                 "reviewer": reviewer,
                 "conflict_with": [p["seq"] for p in prior] if conflict else [],
                 "rule": "decisions_append_only_conflicting_kept_on_branch"},
                [self.store.head(target_branch)])
        return {"seq": seq, "branch": target_branch, "forked": forked,
                "conflict": conflict}

    def _prior_decisions(self, branch: str, target: str) -> list[dict[str, Any]]:
        out = []
        for r in self.visible_events(branch):
            if r["kind"] != "decide":
                continue
            payload = json.loads(r["payload"])
            if payload.get("target") == target:
                out.append({"seq": r["seq"], "verdict": payload["verdict"]})
        return out

    def checkout(self, branch: str, seq: int) -> None:
        """显式移动分支 HEAD（事件不删除，只改可见终点）。"""
        self.store.update_head(branch, seq)

    # ---- 视图（重放到指定事件） ----------------------------------------

    def view(self, branch: str = DEFAULT_BRANCH,
             at_seq: int | None = None) -> dict[str, Any]:
        events = self.visible_events(branch, at_seq)
        visible_seqs = {r["seq"] for r in events}
        head_seq = events[-1]["seq"] if events else 0

        materials = []
        current_refs: dict[str, str] = {}
        for r in self.store.all_materials():
            if r["created_seq"] not in visible_seqs:
                continue
            payload = json.loads(r["payload"])
            is_current = self.current_at(r["material_id"], at_seq, branch)
            cur_ver = is_current[0] if is_current else None
            current = cur_ver == r["version"]
            if current:
                current_refs[r["material_id"]] = ref(r["material_id"], r["version"])
            materials.append({
                "kind": r["material_id"], "version": r["version"],
                "ref": ref(r["material_id"], r["version"]),
                "created_seq": r["created_seq"], "current": current,
                "basis": payload.get("basis", {}),
                "summary": self._material_summary(r["material_id"], payload),
                "diagnostics": payload.get("diagnostics", []),
            })
        materials.sort(key=lambda m: (m["kind"], m["version"]))

        # 积分结果：只显示可见事件里创建的
        integration = None
        fit_sets = []
        for r in self.store.all_results():
            if r["created_seq"] not in visible_seqs:
                continue
            payload = json.loads(r["payload"])
            deps = json.loads(r["deps"])
            if r["kind"] == "integration":
                stale = not self._integration_current(deps, current_refs)
                if integration is None or r["created_seq"] > \
                        self.store.get_result(
                            integration["result_id"])["created_seq"]:
                    integration = self._fit_view(r, payload, deps, stale)
            elif r["kind"] == "fit-set":
                stale, why = self._fitset_staleness(r, payload, deps, branch,
                                                    at_seq, visible_seqs)
                fit_sets.append({
                    "result_id": r["result_id"], "created_seq": r["created_seq"],
                    "fits": payload["fits"], "stale": stale, "stale_reasons": why,
                    "deps": deps})
        if integration is not None:
            integration["stale_reasons"] = []

        decisions = []
        for r in events:
            if r["kind"] == "decide":
                p = json.loads(r["payload"])
                decisions.append({"seq": r["seq"], **p, "branch": r["branch"]})

        timeline = [{
            "seq": r["seq"], "kind": r["kind"], "branch": r["branch"],
            "payload": json.loads(r["payload"]),
        } for r in events]

        diagnostics = []
        if integration:
            for row in integration["payload"]["injections"]:
                for d in row.get("diagnostics", []):
                    diagnostics.append(d)
        return {
            "project": self.store.get_meta("project_name", "ITC 滴定热谱审议案例"),
            "branch": branch,
            "branches": [dict(b) for b in self.store.branches()],
            "head_seq": self.store.head(branch),
            "at_seq": at_seq if at_seq is not None else head_seq,
            "checked_out": at_seq is not None,
            "materials": materials,
            "current_refs": current_refs,
            "integration": integration,
            "fit_sets": sorted(fit_sets, key=lambda f: f["created_seq"]),
            "decisions": decisions,
            "timeline": timeline,
            "diagnostics": diagnostics,
            "units": {"time": "s", "power": "uJ/s", "heat": "uJ",
                      "volume": "uL", "concentration": "mM",
                      "temperature": "K"},
        }

    def _fit_view(self, row, payload, deps, stale):
        return {"result_id": row["result_id"], "created_seq": row["created_seq"],
                "payload": payload, "deps": deps, "stale": stale}

    def _integration_current(self, deps: dict[str, Any],
                             current_refs: dict[str, str]) -> bool:
        for name, spec in deps.items():
            if not spec:
                if name in current_refs:
                    return False
                continue
            if current_refs.get(name) != ref(spec[0], spec[1]):
                return False
        return True

    def _fitset_staleness(self, row, payload, deps, branch, at_seq,
                          visible_seqs) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for name in ("concentrations", "cell_volume", "temperature"):
            spec = deps.get(name)
            cur = self.current_at(name, at_seq, branch)
            if spec and (cur is None or cur[0] != spec[1]):
                reasons.append(f"{name}:{ref(spec[0], spec[1])}->"
                               f"{ref(name, cur[0]) if cur else 'missing'}")
        integ_id = deps.get("integration", [None])[0]
        integ_row = self.store.get_result(integ_id) if integ_id else None
        if integ_row is None or integ_row["created_seq"] not in visible_seqs:
            reasons.append("integration:not_visible")
        else:
            integ_deps = json.loads(integ_row["deps"])
            # 直接用当前视图材料判断父积分是否过期
            for name, spec in integ_deps.items():
                if not spec:
                    continue
                c = self.current_at(name, at_seq, branch)
                if c is None or c[0] != spec[1]:
                    reasons.append(f"integration:{name}@v{spec[1]} stale")
                    break
        return bool(reasons), reasons

    def _material_summary(self, kind: str, p: dict[str, Any]) -> dict[str, Any]:
        if kind in ("power", "blank", "baseline"):
            t = p.get("t") or []
            return {"n_samples": len(t),
                    "t_range_s": [t[0], t[-1]] if t else None,
                    "basis": p.get("basis", {})}
        if kind == "schedule":
            return {"n_injections": len(p.get("injections", [])),
                    "basis": p.get("basis", {})}
        if kind == "concentrations":
            return {"cell_mM": p.get("cell_mM"),
                    "syringe_mM": p.get("syringe_mM"),
                    "basis": p.get("basis", {})}
        if kind == "temperature":
            return {"K": p.get("K"), "basis": p.get("basis", {})}
        if kind == "cell_volume":
            return {"uL": p.get("uL"), "basis": p.get("basis", {})}
        return {}
