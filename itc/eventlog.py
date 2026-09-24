"""只追加事件与重放。

事件类型：
- project_created
- material_added(kind, role, version_id, ...)      新材料或新版本
- integration_rule_set(blank_rule, window_mode...) 积分规则决定
- integration_computed(result_id)
- binding_computed(result_id)
- decision(accept | reject | adjust_endpoints | select_model, target)
- branch_created(name, parent_event_id)            从旧事件分叉
- note

回到任意旧事件：沿分支祖先链取 seq 前缀重放，恢复当时的依赖版本与可见结论，
不拿最新结果覆盖历史。
"""
from __future__ import annotations

import time
import uuid
from typing import Any

EVENT_TYPES = {
    "project_created", "material_added", "integration_rule_set",
    "integration_computed", "binding_computed", "decision",
    "branch_created", "note",
}


def new_event_id() -> str:
    return str(uuid.uuid4())


class EventStore:
    def __init__(self, conn):
        self.conn = conn

    def append(self, *, project_id, branch_id, event_type, payload,
               actor="reviewer", parent_event_id=None,
               supersedes_event_id=None):
        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型 {event_type}")
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),-1)+1 AS n FROM events WHERE branch_id=?",
            (branch_id,)).fetchone()
        seq = int(row["n"])
        event_id = new_event_id()
        self.conn.execute(
            "INSERT INTO events(event_id,project_id,branch_id,seq,"
            "parent_event_id,at,actor,event_type,payload,supersedes_event_id)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_id, project_id, branch_id, seq, parent_event_id,
             time.time(), actor, event_type,
             __import__("json").dumps(payload, ensure_ascii=False,
                                      sort_keys=True),
             supersedes_event_id))
        self.conn.execute("UPDATE branches SET tip_event_id=? WHERE branch_id=?",
                          (event_id, branch_id))
        return event_id

    def events_on_branch(self, branch_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE branch_id=? ORDER BY seq",
            (branch_id,)).fetchall()
        return [self._decode(r) for r in rows]

    def ancestry_chain(self, branch_id: str) -> list[dict]:
        """返回从根到该分支 tip 的完整事件列表（含父分支历史）。"""
        out: list[dict] = []
        seen: set[str] = set()
        cur = branch_id
        while cur:
            br = self.conn.execute(
                "SELECT * FROM branches WHERE branch_id=?", (cur,)).fetchone()
            if br is None:
                break
            evs = [self._decode(r) for r in self.conn.execute(
                "SELECT * FROM events WHERE branch_id=? ORDER BY seq",
                (cur,)).fetchall()]
            out = evs + [e for e in out if e["event_id"] not in
                         {x["event_id"] for x in evs}]
            cur = br["parent_event_id"] and self._branch_of_event(
                br["parent_event_id"])
            if cur in seen:
                break
            seen.add(cur)
        out.sort(key=lambda e: (e["at"], e["event_id"]))
        return out

    def _branch_of_event(self, event_id: str) -> str | None:
        r = self.conn.execute("SELECT branch_id FROM events WHERE event_id=?",
                              (event_id,)).fetchone()
        return r["branch_id"] if r else None

    @staticmethod
    def _decode(r) -> dict:
        import json
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d
