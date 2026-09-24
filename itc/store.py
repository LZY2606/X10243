"""SQLite 存储层：不可变材料分版本、计算血缘、决定历史、机器可读事件。

分层：
- materials     不可变材料，按 (id, version) 追加，永不改写。
- computations  计算结果，记录输入摘要、参数版本、父级依赖；内容寻址，
                等价输入得到同一身份与稳定顺序。
- decisions     可修改的人工决定（接受/否定/调整），按分支记录。
- events        一切变化的事件日志，判断依据机器可读。
- write_intents 写意图日志，用于半写恢复。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY, seq INTEGER UNIQUE, branch TEXT,
  type TEXT, payload TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS materials(
  id TEXT, version INTEGER, kind TEXT, raw TEXT,
  event_id TEXT, PRIMARY KEY(id, version));
CREATE TABLE IF NOT EXISTS computations(
  id TEXT PRIMARY KEY, kind TEXT, param_version TEXT, parents TEXT,
  status TEXT, result TEXT, event_id TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  id TEXT PRIMARY KEY, subject TEXT, action TEXT, rationale TEXT,
  branch TEXT, event_id TEXT);
CREATE TABLE IF NOT EXISTS write_intents(
  id TEXT PRIMARY KEY, effects TEXT, done INTEGER DEFAULT 0);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_of(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.recoveries = self._recover()

    # ---------- 半写恢复 ----------
    def begin_intent(self, effects: list) -> str:
        """多步写入前登记写意图（独立提交，崩溃后仍可见）。"""
        iid = "intent-" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO write_intents(id, effects, done) VALUES(?,?,0)",
            (iid, json.dumps(effects, ensure_ascii=False)))
        self.conn.commit()
        return iid

    def complete_intent(self, iid: str) -> None:
        self.conn.execute("UPDATE write_intents SET done=1 WHERE id=?", (iid,))
        self.conn.commit()

    def _recover(self) -> list:
        """启动时恢复半写：效果齐全则前滚确认，否则回滚孤儿行。"""
        recovered = []
        rows = self.conn.execute(
            "SELECT id, effects FROM write_intents WHERE done=0").fetchall()
        for row in rows:
            effects = json.loads(row["effects"])
            missing, present = [], []
            for table, key in effects:
                hit = self.conn.execute(
                    f"SELECT 1 FROM {table} WHERE id=?", (key,)).fetchone()
                (present if hit else missing).append((table, key))
            if missing:
                for table, key in present:
                    self.conn.execute(f"DELETE FROM {table} WHERE id=?", (key,))
                action = "rolled_back"
            else:
                action = "rolled_forward"
            self.conn.execute("UPDATE write_intents SET done=1 WHERE id=?",
                              (row["id"],))
            self.conn.commit()
            evt = self.add_event("recovery", {
                "intent": row["id"], "action": action,
                "missing": missing, "kept": present,
                "basis": {"rule": "half_write_recovery",
                          "meaning": "写意图未完成：效果齐全则确认，否则删除孤儿行"},
            })
            recovered.append({"intent": row["id"], "action": action,
                              "event": evt["id"]})
        return recovered

    # ---------- 事件 ----------
    def _next_seq(self) -> int:
        row = self.conn.execute("SELECT MAX(seq) AS m FROM events").fetchone()
        return (row["m"] or 0) + 1

    def add_event(self, type_: str, payload: dict, branch: str = "main",
                  event_id: Optional[str] = None,
                  seq: Optional[int] = None) -> dict:
        eid = event_id or "evt-" + uuid.uuid4().hex[:12]
        seq = seq if seq is not None else self._next_seq()
        self.conn.execute(
            "INSERT INTO events(id, seq, branch, type, payload, created)"
            " VALUES(?,?,?,?,?,?)",
            (eid, seq, branch, type_,
             json.dumps(payload, ensure_ascii=False), _now()))
        self.conn.commit()
        return {"id": eid, "seq": seq}

    # ---------- 材料（不可变、分版本） ----------
    def register_material(self, kind: str, material_id: str, raw: Any,
                          branch: str = "main", version: Optional[int] = None,
                          event_id: Optional[str] = None,
                          event_seq: Optional[int] = None,
                          basis: Optional[dict] = None) -> dict:
        if version is None:
            row = self.conn.execute(
                "SELECT MAX(version) AS m FROM materials WHERE id=?",
                (material_id,)).fetchone()
            version = (row["m"] or 0) + 1
        evt = self.add_event("material_registered", {
            "kind": kind, "id": material_id, "version": version,
            "basis": basis or {"rule": "immutable_material_versioned",
                               "meaning": "材料不可变，修订产生新版本"},
        }, branch, event_id=event_id, seq=event_seq)
        self.conn.execute(
            "INSERT INTO materials(id, version, kind, raw, event_id)"
            " VALUES(?,?,?,?,?)",
            (material_id, version, kind,
             json.dumps(raw, ensure_ascii=False), evt["id"]))
        self.conn.commit()
        # 过期传播：只影响把该材料列为父级依赖的计算
        affected = []
        for comp in self._computations():
            parents = json.loads(comp["parents"])
            if material_id in parents and parents[material_id] < version:
                affected.append(comp["id"])
        if affected:
            self.add_event("computations_staled", {
                "material": material_id, "version": version,
                "affected": sorted(affected),
                "basis": {
                    "rule": "dependency_scoped_staleness",
                    "meaning": "只有声明了该父级依赖的计算过期；"
                               "浓度修订使模型过期但不改原积分",
                }}, branch)
        return {"id": material_id, "version": version, "event": evt}

    # ---------- 计算（内容寻址、血缘） ----------
    def put_computation(self, kind: str, param_version: str,
                        parents: dict, status: str, result: dict,
                        branch: str = "main") -> tuple:
        """等价输入得到同一 id；重复计算幂等。返回 (id, created)。"""
        cid = f"cmp-{kind}-" + digest_of({
            "kind": kind, "param_version": param_version,
            "parents": parents})[:12]
        row = self.conn.execute(
            "SELECT id FROM computations WHERE id=?", (cid,)).fetchone()
        if row:
            return cid, False
        evt = self.add_event("computation_recorded", {
            "id": cid, "kind": kind, "status": status,
            "param_version": param_version, "parents": parents,
            "input_digest": digest_of(result.get("input_summary", {}))[:12],
            "rule_basis": result.get("rule_basis", {}),
        }, branch)
        self.conn.execute(
            "INSERT INTO computations(id, kind, param_version, parents,"
            " status, result, event_id) VALUES(?,?,?,?,?,?,?)",
            (cid, kind, param_version, json.dumps(parents, sort_keys=True),
             status, json.dumps(result, ensure_ascii=False), evt["id"]))
        self.conn.commit()
        return cid, True

    # ---------- 决定 ----------
    def add_decision(self, subject: str, action: str, rationale: str,
                     branch: str = "main") -> dict:
        did = "dec-" + uuid.uuid4().hex[:12]
        evt = self.add_event("decision_made", {
            "id": did, "subject": subject, "action": action,
            "rationale": rationale,
            "basis": {"rule": "human_decision_event",
                      "meaning": "接受/否定/调整都生成新事件，不改写历史"},
        }, branch)
        self.conn.execute(
            "INSERT INTO decisions(id, subject, action, rationale, branch,"
            " event_id) VALUES(?,?,?,?,?,?)",
            (did, subject, action, rationale, branch, evt["id"]))
        self.conn.commit()
        return {"id": did, "event": evt}

    # ---------- 查询 / 状态折叠 ----------
    def is_empty(self) -> bool:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()
        return row["c"] == 0

    def _events(self, to_seq: Optional[int] = None) -> list:
        q = "SELECT * FROM events"
        if to_seq is not None:
            q += f" WHERE seq<={int(to_seq)}"
        return self.conn.execute(q + " ORDER BY seq").fetchall()

    def _computations(self) -> list:
        return self.conn.execute("SELECT * FROM computations").fetchall()

    def state_at(self, to_seq: Optional[int] = None) -> dict:
        """折叠到指定事件序号：恢复当时的依赖版本与可见结论。"""
        events = self._events(to_seq)
        visible_evt = {r["id"] for r in events}
        mats = {}
        for r in self.conn.execute("SELECT * FROM materials").fetchall():
            if r["event_id"] not in visible_evt:
                continue
            cur = mats.get(r["kind"])
            if cur is None or r["version"] > cur["version"]:
                mats[r["kind"]] = {"id": r["id"], "version": r["version"],
                                   "raw": json.loads(r["raw"]),
                                   "event_id": r["event_id"]}
        current_version = {m["id"]: m["version"] for m in mats.values()}
        comps = []
        for r in self._computations():
            if r["event_id"] not in visible_evt:
                continue
            parents = json.loads(r["parents"])
            stale = any(current_version.get(pid, 0) > pv
                        for pid, pv in parents.items())
            comps.append({"id": r["id"], "kind": r["kind"],
                          "param_version": r["param_version"],
                          "parents": parents, "status": r["status"],
                          "stale": stale, "result": json.loads(r["result"])})
        comps.sort(key=lambda c: c["id"])  # 稳定顺序
        decisions = []
        for r in self.conn.execute("SELECT * FROM decisions").fetchall():
            if r["event_id"] in visible_evt:
                decisions.append({"id": r["id"], "subject": r["subject"],
                                  "action": r["action"],
                                  "rationale": r["rationale"],
                                  "branch": r["branch"]})
        # 冲突：同一对象在不同分支上接受与否定并存
        by_subject = {}
        for d in decisions:
            by_subject.setdefault(d["subject"], set()).add(d["action"])
        conflicts = sorted(s for s, acts in by_subject.items()
                           if "accept" in acts and "reject" in acts)
        return {
            "seq": events[-1]["seq"] if events else 0,
            "latest_seq": self._next_seq() - 1,
            "branches": sorted({r["branch"] for r in events}),
            "materials": mats,
            "computations": comps,
            "decisions": decisions,
            "conflicts": conflicts,
            "recoveries": self.recoveries,
            "events": [{"seq": r["seq"], "id": r["id"], "branch": r["branch"],
                        "type": r["type"],
                        "payload": json.loads(r["payload"])} for r in events],
        }

    def close(self) -> None:
        self.conn.close()
