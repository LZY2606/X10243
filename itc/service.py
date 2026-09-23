"""应用服务层：业务规则编排在服务端（不只藏在浏览器）。

每个计算结果都记录：inputs（材料内容摘要+版本）、params（规则版本）、
parents（父级产物/事件）。输入不完整时抛带位置诊断的 ITCError，
绝不用空值或默认值伪装成功。
"""
from __future__ import annotations

import copy

from .materials import import_material
from .storage import Store, digest, IMPORT_EVENT
from .models import (
    Diagnostic, ITCError, FormatCorrupted, BudgetExhausted,
    EV_POWER_PATCHED, EV_CONCENTRATION_REVISED, EV_FIRST_DISCARD_SET,
    EV_INTEGRATION_RUN, EV_BLANK_MATCH_SET, EV_FIT_RUN, EV_DECISION,
    EV_INVALIDATED, MODEL_ORDER,
)
from .integration import integrate_injections, INTEGRATION_RULE_VERSION
from .dilution import apply_dilution, DILUTION_RULE_VERSION
from .binding import fit_models, BINDING_RULE_VERSION
from . import state as state_mod

MAIN_CHAIN = "main"


class Service:
    def __init__(self, store: Store):
        self.store = store

    # ---------- 导入（不可变材料；同内容幂等；事件链记录版本）----------
    def import_raw(self, kind: str, raw_text: str, chain_id: str = MAIN_CHAIN,
                   parent_event_id: str | None = None, note: str = "") -> dict:
        mat = import_material(kind, raw_text)  # 可能抛 FormatCorrupted/Budget/ITC
        is_new = self.store.put_material(mat["id"], kind, raw_text, mat["parsed"])
        payload = {"material_id": mat["id"], "kind": kind,
                   "note": note, "reused": not is_new}
        ev = self.store.add_event(
            IMPORT_EVENT[kind], chain_id, payload, parent_event_id,
            inputs=[{"material_id": mat["id"], "kind": kind,
                     "content_digest": mat["id"].split("_", 1)[1]}],
            params={"material_kind_version": "material-1.0.0"},
            parents=[mat["id"]])
        return {"event": ev, "material": mat, "created_new": is_new}

    # ---------- 首注舍弃（规则决定；不删任何积分）----------
    def set_first_discard(self, parent_event_id: str, ordinals,
                          chain_id: str = MAIN_CHAIN,
                          window_s: float | None = None, note: str = "") -> dict:
        cur = self._view(parent_event_id)
        if window_s is None:
            if not cur.get("integration"):
                raise ITCError(
                    "尚无积分结果，无法设置舍弃规则",
                    [Diagnostic("missing_integration", "请先运行注射积分",
                                "integration")])
            window_s = cur["integration"]["params"]["window_s"]
        payload = {"discard_ordinals": sorted(int(x) for x in ordinals),
                   "window_s": float(window_s), "note": note}
        ev = self.store.add_event(
            EV_FIRST_DISCARD_SET, chain_id, payload, parent_event_id,
            params={"rule_version": INTEGRATION_RULE_VERSION},
            parents=[parent_event_id])
        return {"event": ev}

    # ---------- 积分（重叠峰允许、缺口不跨窗、端点裁剪）----------
    def run_integration(self, parent_event_id: str, window_s: float,
                        chain_id: str = MAIN_CHAIN,
                        discard_ordinals=None,
                        baseline_id: str | None = None, note: str = "") -> dict:
        v = self._view(parent_event_id)
        mats = v["materials"]
        for need in ("power", "injections"):
            if need not in mats:
                raise ITCError(
                    f"缺少必需材料 {need}",
                    [Diagnostic("missing_material",
                                f"运行积分需要 {need}", f"material:{need}")])
        if baseline_id is None:
            baseline = mats.get("baseline")
        else:
            baseline = self.store.get_material(baseline_id)
            if baseline is None or baseline["kind"] != "baseline":
                raise ITCError(
                    f"基线材料不存在: {baseline_id}",
                    [Diagnostic("unknown_baseline", "指定的基线版本不存在",
                                f"material:{baseline_id}")])
        doc = integrate_injections(
            mats["power"]["parsed"],
            baseline["parsed"] if baseline else None,
            mats["injections"]["parsed"],
            {"window_s": float(window_s),
             "discard_ordinals": list(discard_ordinals or [])},
        )
        doc["artifact_id"] = None
        aid = "art_int_" + digest([
            mats["power"]["id"],
            baseline["id"] if baseline else None,
            mats["injections"]["id"],
            doc["params"], INTEGRATION_RULE_VERSION,
        ])[:20]
        # entries 已带各自 input_digest；产物整体内容身份
        doc["id"] = aid
        ev = self.store.add_event(
            EV_INTEGRATION_RUN, chain_id,
            {"artifact_id": aid, "window_s": float(window_s),
             "note": note,
             "n_entries": len(doc["entries"]),
             "n_discarded": sum(1 for e in doc["entries"] if e["discarded"]),
             "n_overlap": sum(1 for e in doc["entries"] if e["overlaps_with"]),
             "n_gap_clipped": sum(1 for e in doc["entries"]
                                  if e["excluded_gap_intervals_s"])},
            parent_event_id,
            inputs=[{"power": mats["power"]["id"]},
                    {"baseline": baseline["id"] if baseline else None},
                    {"injections": mats["injections"]["id"]}],
            params={"window_s": float(window_s),
                    "rule_version": INTEGRATION_RULE_VERSION,
                    "discard_ordinals": list(discard_ordinals or [])},
            parents=[mats["power"]["id"], mats["injections"]["id"]]
                    + ([baseline["id"]] if baseline else [])
                    + [parent_event_id],
            artifacts=[(aid, "integration", doc)])
        return {"event": ev, "artifact": self.store.get_artifact(aid)}

    # ---------- 稀释热对应（规则必须显式）----------
    def apply_blank(self, parent_event_id: str, rule: str = "by_index",
                    chain_id: str = MAIN_CHAIN,
                    blank_id: str | None = None,
                    blank_baseline_id: str | None = None,
                    note: str = "") -> dict:
        v = self._view(parent_event_id)
        if not v.get("integration"):
            raise ITCError("尚无积分结果，无法扣除稀释热",
                           [Diagnostic("missing_integration",
                                       "请先运行注射积分", "integration")])
        int_doc = v["integration"]
        blank = None
        if blank_id is not None:
            blank = self.store.get_material(blank_id)
            if blank is None or blank["kind"] != "blank":
                raise ITCError(f"空白材料不存在: {blank_id}",
                               [Diagnostic("unknown_blank",
                                           "指定的空白版本不存在",
                                           f"material:{blank_id}")])
        else:
            blank = v["materials"].get("blank")
        blank_baseline = (self.store.get_material(blank_baseline_id)
                          if blank_baseline_id else None)
        doc = apply_dilution(
            {"artifact_id": int_doc["id"], **int_doc},
            {"id": blank["id"], "parsed": blank["parsed"]} if blank else None,
            {"id": blank_baseline["id"], "parsed": blank_baseline["parsed"]}
            if blank_baseline else None,
            {"rule": rule})
        aid = "art_dil_" + digest([
            int_doc["id"], blank["id"] if blank else None, rule,
            DILUTION_RULE_VERSION])[:20]
        doc["id"] = aid
        n_matched = doc["summary"].get("matched", 0)
        n_missing = sum(1 for e in doc["entries"]
                        if e.get("blank_status") not in (None, "matched"))
        ev = self.store.add_event(
            EV_BLANK_MATCH_SET, chain_id,
            {"artifact_id": aid, "rule": rule,
             "blank_material": blank["id"] if blank else None,
             "n_matched": n_matched, "n_blank_missing": n_missing,
             "note": note},
            parent_event_id,
            inputs=[{"integration": int_doc["id"]},
                    {"blank": blank["id"] if blank else None}],
            params={"rule": rule, "rule_version": DILUTION_RULE_VERSION},
            parents=[int_doc["id"]]
                    + ([blank["id"]] if blank else [])
                    + [parent_event_id],
            artifacts=[(aid, "dilution", doc)])
        return {"event": ev, "artifact": self.store.get_artifact(aid)}

    # ---------- 绑定模型（等优候选全部保留）----------
    def run_fit(self, parent_event_id: str, chain_id: str = MAIN_CHAIN,
                models=MODEL_ORDER, note: str = "") -> dict:
        v = self._view(parent_event_id)
        if not v.get("dilution"):
            raise ITCError("尚无稀释扣除结果，无法拟合绑定模型",
                           [Diagnostic("missing_dilution",
                                       "请先说明稀释热对应规则并扣除",
                                       "dilution")])
        for need in ("conditions", "concentration"):
            if need not in v["materials"]:
                raise ITCError(f"缺少必需材料 {need}",
                               [Diagnostic("missing_material",
                                           f"拟合需要 {need}",
                                           f"material:{need}")])
        dilution = copy.deepcopy(v["dilution"])
        doc = fit_models(
            dilution,
            v["materials"]["conditions"]["parsed"],
            v["materials"]["concentration"]["parsed"],
            models=tuple(models))
        conc_id = v["materials"]["concentration"]["id"]
        aid = "art_fit_" + digest([
            v["dilution"]["id"], conc_id,
            v["materials"]["conditions"]["id"],
            sorted(models), BINDING_RULE_VERSION])[:20]
        doc["id"] = aid
        doc["inputs"] = {
            "dilution_artifact": v["dilution"]["id"],
            "concentration_material": conc_id,
            "conditions_material": v["materials"]["conditions"]["id"],
        }
        winners = [c["model"] for c in doc["candidates"] if c.get("is_winner")]
        ev = self.store.add_event(
            EV_FIT_RUN, chain_id,
            {"artifact_id": aid, "models": list(models),
             "winners": winners, "tie_size": len(winners),
             "note": note},
            parent_event_id,
            inputs=[{"dilution": v["dilution"]["id"]},
                    {"concentration": conc_id},
                    {"conditions": v["materials"]["conditions"]["id"]}],
            params={"models": list(models),
                    "rule_version": BINDING_RULE_VERSION},
            parents=[v["dilution"]["id"], conc_id,
                     v["materials"]["conditions"]["id"], parent_event_id],
            artifacts=[(aid, "binding", doc)])
        return {"event": ev, "artifact": self.store.get_artifact(aid)}

    # ---------- 人工决定（接受/否定/参数调整；被采纳与未采纳都保留）----------
    def decide(self, parent_event_id: str, target_artifact: str,
               verdict: str, chain_id: str = MAIN_CHAIN,
               parameter_overrides: dict | None = None,
               note: str = "") -> dict:
        if verdict not in ("accept", "reject", "adjust"):
            raise ITCError(
                f"非法决定 {verdict!r}",
                [Diagnostic("bad_verdict",
                            "verdict 必须是 accept/reject/adjust",
                            "verdict")])
        art = self.store.get_artifact(target_artifact)
        if art is None:
            raise ITCError(f"决定目标不存在: {target_artifact}",
                           [Diagnostic("unknown_target", "目标产物不存在",
                                       target_artifact)])
        if verdict == "adjust" and not parameter_overrides:
            raise ITCError(
                "adjust 决定必须携带 parameter_overrides",
                [Diagnostic("missing_overrides",
                            "参数调整必须说明调整了哪些参数",
                            "parameter_overrides")])
        payload = {"target_artifact": target_artifact, "verdict": verdict,
                   "parameter_overrides": parameter_overrides or {},
                   "note": note,
                   "target_kind": art["kind"]}
        ev = self.store.add_event(
            EV_DECISION, chain_id, payload, parent_event_id,
            params={"decision_rule_version": "decision-1.0.0"},
            parents=[target_artifact, parent_event_id])
        return {"event": ev}

    # ---------- 浓度修订：模型过期，但不改原积分 ----------
    def revise_concentration(self, parent_event_id: str, raw_text: str,
                             chain_id: str = MAIN_CHAIN,
                             note: str = "") -> dict:
        v = self._view(parent_event_id)
        old_conc_id = (v["materials"].get("concentration") or {}).get("id")
        mat = import_material("concentration", raw_text)
        self.store.put_material(mat["id"], "concentration", raw_text,
                                mat["parsed"])
        payload = {"material_id": mat["id"],
                   "previous_material_id": old_conc_id,
                   "reason": "concentration_revision", "note": note}
        ev = self.store.add_event(
            EV_CONCENTRATION_REVISED, chain_id, payload, parent_event_id,
            inputs=[{"concentration": mat["id"]},
                    {"previous": old_conc_id}],
            params={"material_kind_version": "material-1.0.0"},
            parents=[mat["id"]] + ([old_conc_id] if old_conc_id else []))
        # 仅令下游“绑定拟合”过期；积分与稀释的输入不含浓度，保持原结果
        invalid = []
        if v.get("fit"):
            invalid.append(v["fit"]["id"])
        if invalid:
            self.store.add_event(
                EV_INVALIDATED, chain_id,
                {"artifact_ids": invalid,
                 "reason": "concentration_revision",
                 "new_concentration_material": mat["id"],
                 "note": "浓度修订使模型过期；原积分与稀释扣除不变"},
                ev["id"],
                params={"invalidation_rule_version": "invalid-1.0.0"},
                parents=[mat["id"]] + invalid)
        return {"event": ev, "material": mat,
                "invalidated_fit_ids": invalid}

    # ---------- 局部源数据改动（功率补丁；影响范围由 input_digest 决定）----------
    def patch_power(self, parent_event_id: str, raw_text: str,
                    chain_id: str = MAIN_CHAIN, scope: str = "",
                    note: str = "") -> dict:
        v = self._view(parent_event_id)
        old_power_id = v["materials"]["power"]["id"]
        mat = import_material("power", raw_text)
        self.store.put_material(mat["id"], "power", raw_text, mat["parsed"])
        invalid = []
        for key in ("integration", "dilution", "fit"):
            if v.get(key):
                invalid.append(v[key]["id"])
        ev = self.store.add_event(
            EV_POWER_PATCHED, chain_id,
            {"material_id": mat["id"], "previous_material_id": old_power_id,
             "scope": scope, "reason": "local_source_patch", "note": note},
            parent_event_id,
            inputs=[{"power": mat["id"]}, {"previous": old_power_id},
                    {"scope": scope}],
            params={"patch_rule_version": "patch-1.0.0"},
            parents=[mat["id"], old_power_id])
        self.store.add_event(
            EV_INVALIDATED, chain_id,
            {"artifact_ids": invalid, "reason": "power_patch",
             "new_power_material": mat["id"], "scope": scope},
            ev["id"],
            params={"invalidation_rule_version": "invalid-1.0.0"},
            parents=[mat["id"]] + invalid)
        return {"event": ev, "material": mat,
                "invalidated_artifact_ids": invalid}

    # ---------- 分支（fork：保留所有决定与历史，不覆盖）----------
    def fork(self, parent_event_id: str, new_chain_id: str,
             note: str = "") -> dict:
        if self.store.get_event(parent_event_id) is None:
            raise ITCError(f"fork 基点事件不存在: {parent_event_id}",
                           [Diagnostic("unknown_parent",
                                       "fork 基点事件不存在", parent_event_id)])
        existing = set(self.store.head_events().keys())
        if new_chain_id in existing:
            raise ITCError(f"分支已存在: {new_chain_id}",
                           [Diagnostic("chain_exists", "分支名已被使用",
                                       new_chain_id)])
        # fork 标记事件：属于新链，parent 指向基点（跨链祖先）
        ev = self.store.add_event(
            "fork", new_chain_id, {"forked_from": parent_event_id,
                                   "note": note},
            parent_event_id,
            params={"branch_rule_version": "branch-1.0.0"},
            parents=[parent_event_id])
        return {"event": ev}

    # ---------- 视图 ----------
    def _view(self, event_id: str) -> dict:
        if self.store.get_event(event_id) is None:
            raise ITCError(f"事件不存在: {event_id}",
                           [Diagnostic("unknown_event", "事件不存在", event_id)])
        return state_mod.reconstruct(self.store, event_id)

    def view(self, event_id: str) -> dict:
        return self._view(event_id)

    def impact_compare(self, before_event_id: str, after_event_id: str,
                       kind: str = "integration") -> dict:
        """对比源数据改动前后逐条输入摘要，验证影响是否被限制在正确范围。"""
        a = self._view(before_event_id)
        b = self._view(after_event_id)
        art_a = a.get(kind) or {}
        art_b = b.get(kind) or {}
        ea = {e["ordinal"]: e.get("input_digest")
              for e in art_a.get("entries", [])}
        eb = {e["ordinal"]: e.get("input_digest")
              for e in art_b.get("entries", [])}
        changed, unchanged, added, removed = [], [], [], []
        for k in sorted(set(ea) | set(eb)):
            if k in ea and k in eb:
                (changed if ea[k] != eb[k] else unchanged).append(k)
            elif k in eb:
                added.append(k)
            else:
                removed.append(k)
        return {"changed_ordinals": changed, "unchanged_ordinals": unchanged,
                "added": added, "removed": removed,
                "artifact_a": art_a.get("id"), "artifact_b": art_b.get("id")}
