"""事件溯源：不可变材料 vs 可修改决定的分层与可重放状态。

事件只记录“事实”：材料导入、人工决定、以及计算请求（计算结果由 reducer 重放，
保证等价输入得到等价结果）。所有结果记录输入摘要、参数版本与父级依赖。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    MATERIAL_IMPORTED = "material_imported"
    FIRST_DISCARD_SET = "first_discard_set"
    ENDPOINTS_SET = "endpoints_set"
    INTEGRATION_REQUESTED = "integration_requested"
    BLANK_RULE_SET = "blank_rule_set"
    SUBTRACTION_REQUESTED = "subtraction_requested"
    FIT_REQUESTED = "fit_requested"
    MODEL_DECISION = "model_decision"          # 接受 / 拒绝 / 参数调整
    COUNTEREXAMPLE_RECORDED = "counterexample_recorded"


# 材料种类的稳定版本顺序（基线与空白独立版本化）
MATERIAL_VERSION_ORDER = (
    "power", "injections", "concentration", "temperature",
    "cell_volume", "baseline", "blank",
)


@dataclass(frozen=True)
class Event:
    seq: int                     # 分支内单调序号
    event_id: str
    branch: str
    parent_event_id: str         # 分支链上的父事件
    type: EventType
    payload: dict
    created_note: str = ""


def event_hash(branch: str, parent_event_id: str, etype: EventType,
               payload: dict, seq: int) -> str:
    body = repr((seq, branch, parent_event_id, etype.value,
                 _stable(payload)))
    return "evt_" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _stable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return tuple(sorted((k, _stable(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_stable(v) for v in obj)
    return obj


@dataclass
class BranchState:
    branch: str = "main"
    # kind -> {"id","version","summary"}
    materials: dict[str, dict] = field(default_factory=dict)
    discarded_first_n: int = 1
    endpoints: dict[int, dict] = field(default_factory=dict)
    blank_rule: dict | None = None
    last_integration: dict | None = None
    last_subtraction: dict | None = None
    last_fit: dict | None = None
    accepted_models: dict[str, bool] = field(default_factory=dict)
    counterexamples: list[dict] = field(default_factory=list)
    history: list[Event] = field(default_factory=list)
    heads: dict[str, str] = field(default_factory=lambda: {"main": "ROOT"})

    def material_versions(self) -> dict[str, str]:
        return {k: v["id"] for k, v in self.materials.items()}

    def required_material(self, kind: str) -> dict | None:
        return self.materials.get(kind)


class Reducer:
    """纯函数式重放；返回投影状态，供任意历史时点复查。"""

    def __init__(self) -> None:
        self.states: dict[str, BranchState] = {"main": BranchState()}

    def apply(self, ev: Event) -> BranchState:
        st = self.states.setdefault(ev.branch, BranchState(ev.branch))
        p = ev.payload
        if ev.type is EventType.MATERIAL_IMPORTED:
            st.materials[p["kind"]] = {
                "id": p["material_id"],
                "version": p["version"],
                "summary": p.get("summary", {}),
            }
        elif ev.type is EventType.FIRST_DISCARD_SET:
            st.discarded_first_n = int(p["n"])
        elif ev.type is EventType.ENDPOINTS_SET:
            for item in p["overrides"]:
                st.endpoints[int(item["injection_index"])] = item
        elif ev.type is EventType.INTEGRATION_REQUESTED:
            st.last_integration = {
                "id": p["integration_id"],
                "depends": dict(p["depends"]),
                "summary": p.get("summary", {}),
                "event_id": ev.event_id,
            }
        elif ev.type is EventType.BLANK_RULE_SET:
            st.blank_rule = {"rule": p["rule"], "mapping": p.get("mapping", {}),
                             "note": p.get("note", ""), "rule_id": p["rule_id"]}
        elif ev.type is EventType.SUBTRACTION_REQUESTED:
            st.last_subtraction = {
                "id": p["subtraction_id"],
                "depends": dict(p["depends"]),
                "rule_id": p.get("rule_id"),
                "summary": p.get("summary", {}),
                "event_id": ev.event_id,
            }
        elif ev.type is EventType.FIT_REQUESTED:
            st.last_fit = {
                "id": p["fit_id"],
                "depends": dict(p["depends"]),
                "input_digest": p.get("input_digest", ""),
                "summary": p.get("summary", {}),
                "event_id": ev.event_id,
            }
        elif ev.type is EventType.MODEL_DECISION:
            st.accepted_models[p["candidate_id"]] = p["accepted"]
            md = st.last_fit
            if md is not None:
                md.setdefault("decisions", {})[p["candidate_id"]] = {
                    "accepted": p["accepted"],
                    "note": p.get("note", ""),
                    "params": p.get("params"),
                    "event_id": ev.event_id,
                }
        elif ev.type is EventType.COUNTEREXAMPLE_RECORDED:
            st.counterexamples.append(p)
        st.history.append(ev)
        st.heads[ev.branch] = ev.event_id
        return st


def replay(events: list[Event]) -> dict[str, BranchState]:
    r = Reducer()
    for ev in sorted(events, key=lambda e: (e.branch != "main", e.seq)):
        r.apply(ev)
    return r.states


def state_at(events: list[Event], branch: str, event_id: str) -> BranchState:
    """重放到指定事件（含），恢复当时的依赖版本与可见结论。"""
    r = Reducer()
    for ev in events:
        if ev.branch != branch:
            continue
        r.apply(ev)
        if ev.event_id == event_id:
            return r.states[branch]
    raise KeyError(f"事件 {event_id} 不在分支 {branch}")
