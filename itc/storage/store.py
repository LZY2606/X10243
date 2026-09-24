"""仓储：项目、分支、事务化事件追加与材料版本管理。"""
from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any

from ..core.constants import (EVT_BRANCH_OPENED, EVT_DECISION,
                              EVT_MATERIAL_UPLOADED, EVT_PROJECT_CREATED,
                              EVT_REVIEW, MATERIAL_SLOTS)
from ..core.hashing import digest
from .db import dumps, loads


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, conn):
        self.conn = conn

    # ---------- 事务外壳（半写恢复的依据） ----------
    def _open_txn(self, project_id: str, branch_id: str,
                  simulate_crash: bool = False) -> str:
        txn = new_id("txn")
        self.conn.execute(
            "INSERT INTO txn_journal(txn_id,project_id,branch_id,state,"
            "created_at) VALUES(?,?,?,?,?)",
            (txn, project_id, branch_id,
             "crashed" if simulate_crash else "pending", now()))
        self.conn.commit()
        return txn

    def _append_rows(self, txn: str, project_id: str, branch_id: str,
                     rows: list[tuple[str, dict[str, Any], str | None,
                                      str]]) -> list[str]:
        seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) AS s FROM events WHERE branch_id=?",
            (branch_id,)).fetchone()
        seq = seq_row["s"]
        parent_row = self.conn.execute(
            "SELECT id FROM events WHERE branch_id=? ORDER BY seq DESC LIMIT 1",
            (branch_id,)).fetchone()
        parent = parent_row["id"] if parent_row else None
        ids = []
        for etype, payload, actor, note in rows:
            seq += 1
            eid = new_id("evt")
            self.conn.execute(
                "INSERT INTO events(id,project_id,branch_id,seq,"
                "parent_event_id,type,payload_json,actor,note,created_at,"
                "txn_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (eid, project_id, branch_id, seq, parent, etype,
                 dumps(payload), actor or "reviewer", note, now(), txn))
            parent = eid
            ids.append(eid)
        return ids

    def _commit_txn(self, txn: str) -> None:
        self.conn.execute(
            "UPDATE txn_journal SET state='committed',finished_at=? "
            "WHERE txn_id=?", (now(), txn))
        self.conn.commit()

    # ---------- 项目 / 分支 ----------
    def create_project(self, name: str, project_id: str | None = None,
                       actor: str = "reviewer") -> dict[str, Any]:
        pid = project_id or new_id("prj")
        bid = new_id("br")
        ts = now()
        self.conn.execute("BEGIN")
        txn = self._open_txn(pid, bid)
        self.conn.execute(
            "INSERT INTO projects(id,name,created_at) VALUES(?,?,?)",
            (pid, name, ts))
        self.conn.execute(
            "INSERT INTO branches(id,project_id,name,fork_event_id,"
            "created_at) VALUES(?,?,?,?,?)",
            (bid, pid, "main", "", ts))
        eid = self._append_rows(txn, pid, bid,
                                [(EVT_PROJECT_CREATED, {"name": name},
                                  actor, "项目创建")])[0]
        self.conn.execute(
            "UPDATE projects SET head_event_id=? WHERE id=?", (eid, pid))
        self.conn.execute(
            "UPDATE branches SET head_event_id=? WHERE id=?", (eid, bid))
        self._commit_txn(txn)
        return {"project_id": pid, "branch_id": bid, "event_id": eid}

    def open_branch(self, project_id: str, name: str,
                    from_event_id: str | None = None,
                    actor: str = "reviewer") -> dict[str, Any]:
        src = self.get_main_branch(project_id)
        head = from_event_id or src["head_event_id"]
        bid = new_id("br")
        self.conn.execute("BEGIN")
        txn = self._open_txn(project_id, bid)
        self.conn.execute(
            "INSERT INTO branches(id,project_id,name,parent_branch_id,"
            "fork_event_id,created_at,head_event_id) VALUES(?,?,?,?,?,?,?)",
            (bid, project_id, name, src["id"], src["id"], head, now(),
             head))
        self._commit_txn(txn)
        # 分支事件流从 fork 点之后追加；首事件父为 fork 事件
        self.conn.execute("BEGIN")
        txn2 = self._open_txn(project_id, bid)
        eid = self._append_rows(txn2, project_id, bid,
                                [(EVT_BRANCH_OPENED,
                                  {"name": name, "fork_event_id": head},
                                  actor, f"分支 {name}")])[0]
        self.conn.execute(
            "UPDATE branches SET head_event_id=? WHERE id=?", (eid, bid))
        self._commit_txn(txn2)
        return {"branch_id": bid, "event_id": eid}

    def get_main_branch(self, project_id: str) -> dict[str, Any]:
        r = self.conn.execute(
            "SELECT * FROM branches WHERE project_id=? ORDER BY created_at "
            "LIMIT 1", (project_id,)).fetchone()
        return dict(r)

    def list_branches(self, project_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches WHERE project_id=? ORDER BY created_at",
            (project_id,))]

    # ---------- 材料 ----------
    def upload_material(self, project_id: str, branch_id: str, slot: str,
                        payload: dict[str, Any], label: str | None = None,
                        material_id: str | None = None,
                        actor: str = "instrument",
                        note: str | None = None) -> dict[str, Any]:
        if slot not in MATERIAL_SLOTS:
            raise ValueError(f"未知材料槽 {slot}")
        ver_row = self.conn.execute(
            "SELECT COALESCE(MAX(version),0) v FROM materials WHERE id=?",
            (material_id,)).fetchone() if material_id else None
        mid = material_id or f"mat_{slot}_{new_id('')}"
        version = (ver_row["v"] + 1) if ver_row else 1
        raw_digest = digest(payload)
        self.conn.execute("BEGIN")
        txn = self._open_txn(project_id, branch_id)
        if version > 1:
            prev = self.conn.execute(
                "SELECT introduced_event_id FROM materials WHERE id=? "
                "ORDER BY version DESC LIMIT 1", (mid,)).fetchone()
            eid = self._append_rows(
                txn, project_id, branch_id,
                [(EVT_MATERIAL_UPLOADED,
                  {"slot": slot, "material_id": mid, "version": version,
                   "raw_digest": raw_digest, "label": label,
                   "supersedes_event": prev["introduced_event_id"]
                   if prev else None},
                  actor, note or f"{slot} v{version}")])[0]
        else:
            eid = self._append_rows(
                txn, project_id, branch_id,
                [(EVT_MATERIAL_UPLOADED,
                  {"slot": slot, "material_id": mid, "version": 1,
                   "raw_digest": raw_digest, "label": label},
                  actor, note or f"{slot} v1")])[0]
        self.conn.execute(
            "INSERT INTO materials(id,version,project_id,slot,label,"
            "payload_json,raw_digest,introduced_event_id) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (mid, version, project_id, slot, label, dumps(payload),
             raw_digest, eid))
        self._advance_head(project_id, branch_id, eid)
        self._commit_txn(txn)
        return {"material_id": mid, "version": version, "event_id": eid,
                "raw_digest": raw_digest}

    # ---------- 决定 / 审阅 ----------
    def apply_decision(self, project_id: str, branch_id: str, key: str,
                       value: Any, kind: str,
                       actor: str = "reviewer",
                       note: str | None = None) -> str:
        return self._event(project_id, branch_id, EVT_DECISION,
                           {"key": key, "value": value, "kind": kind},
                           actor, note or f"决定 {key}")

    def apply_review(self, project_id: str, branch_id: str, subject: str,
                     verdict: str, params: dict[str, Any] | None = None,
                     actor: str = "reviewer",
                     note: str | None = None) -> str:
        return self._event(project_id, branch_id, EVT_REVIEW,
                           {"subject": subject, "verdict": verdict,
                            "params": params or {}},
                           actor, note or f"审阅 {subject}: {verdict}")

    def add_note(self, project_id: str, branch_id: str, text: str,
                 actor: str = "reviewer") -> str:
        return self._event(project_id, branch_id, "note",
                           {"text": text}, actor, text)

    def _event(self, project_id: str, branch_id: str, etype: str,
               payload: dict[str, Any], actor: str, note: str) -> str:
        self.conn.execute("BEGIN")
        txn = self._open_txn(project_id, branch_id)
        eid = self._append_rows(txn, project_id, branch_id,
                                [(etype, payload, actor, note)])[0]
        self._advance_head(project_id, branch_id, eid)
        self._commit_txn(txn)
        return eid

    def _advance_head(self, project_id: str, branch_id: str,
                      eid: str) -> None:
        self.conn.execute("UPDATE branches SET head_event_id=? WHERE id=?",
                          (eid, branch_id))

    # ---------- 读取 ----------
    def events(self, project_id: str, branch_id: str,
               upto: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.conn.execute(
            "SELECT * FROM events WHERE project_id=? AND branch_id=? "
            "ORDER BY seq", (project_id, branch_id)))
        if upto:
            cut = next((i for i, r in enumerate(rows)
                        if r["id"] == upto), len(rows) - 1)
            rows = rows[:cut + 1]
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = loads(d.pop("payload_json"))
            out.append(d)
        return out

    def all_materials(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM materials WHERE project_id=? ORDER BY version",
            (project_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = loads(d.pop("payload_json"))
            out.append(d)
        return out
