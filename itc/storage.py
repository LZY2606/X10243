"""Append-only event store in SQLite.

The event stream is the only source of truth. A write batch becomes visible
atomically through the `committed` flag: rows inserted inside an open batch
are recovered (rolled back) on the next open, which implements half-write
recovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    project_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    parent_id TEXT,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    ts REAL NOT NULL,
    committed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, event_id)
);
CREATE TABLE IF NOT EXISTS branches (
    project_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    name TEXT NOT NULL,
    head_id TEXT,
    fork_seq INTEGER,
    PRIMARY KEY (project_id, branch_id)
);
CREATE TABLE IF NOT EXISTS write_batches (
    project_id TEXT NOT NULL,
    batch_id TEXT PRIMARY KEY,
    opened_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kvmeta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def digest_event(project_id: str, branch_id: str, parent_id: str | None,
                 etype: str, payload: dict) -> str:
    # Identity is content-order based (project + parent chain + type +
    # payload). branch_id is intentionally excluded: the same decision taken
    # from the same parent must keep one stable identity, and the events table
    # still records which branch each row belongs to.
    basis = json.dumps(
        {"p": project_id, "parent": parent_id,
         "type": etype, "payload": payload},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return "evt_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def sid(prefix: str, *parts: Any) -> str:
    basis = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return prefix + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._mem: list[sqlite3.Connection] = []
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        with self._lock:
            self.conn.executescript(SCHEMA)
            self._recover_pending()

    # ---- low level -------------------------------------------------------
    def _recover_pending(self) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE committed=0")
        n = cur.fetchone()["n"]
        if n:
            self.conn.execute("DELETE FROM events WHERE committed=0")
            self.conn.execute("DELETE FROM write_batches")
        return n

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _next_seq(self, project_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE project_id=?",
            (project_id,),
        ).fetchone()
        return int(row["m"]) + 1

    # ---- project / branch -----------------------------------------------
    def projects(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        return dict(r) if r else None

    def list_branches(self, project_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM branches WHERE project_id=? ORDER BY fork_seq, branch_id",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def branch_head(self, project_id: str, branch_id: str) -> str | None:
        r = self.conn.execute(
            "SELECT head_id FROM branches WHERE project_id=? AND branch_id=?",
            (project_id, branch_id),
        ).fetchone()
        return r["head_id"] if r else None

    def branch_name(self, project_id: str, branch_id: str) -> str:
        r = self.conn.execute(
            "SELECT name FROM branches WHERE project_id=? AND branch_id=?",
            (project_id, branch_id),
        ).fetchone()
        return r["name"] if r else branch_id

    # ---- write batches ---------------------------------------------------
    def begin_batch(self, project_id: str, batch_id: str | None = None) -> str:
        """Open a recovery unit.

        Rows are persisted immediately but marked committed=0; the batch only
        becomes visible atomically in commit_batch(). A crash at any point
        leaves either the old committed state or recoverable pending rows.
        """
        with self._lock:
            batch_id = batch_id or sid("bat_", project_id, time.time_ns(), os.getpid())
            self.conn.execute(
                "INSERT INTO write_batches(project_id, batch_id, opened_at) VALUES (?,?,?)",
                (project_id, batch_id, time.time()),
            )
            self._open_batch = (project_id, batch_id)
            return batch_id

    def append_event(self, project_id: str, branch_id: str, etype: str, payload: dict,
                     *, parent_id: str | None = None, batch: bool = True,
                     ts: float | None = None) -> dict:
        """Append one event. Requires an open batch when batch=True.

        Event id/seq are content-order based, so replaying identical streams
        yields identical identities across export/import.
        """
        if batch:
            if not getattr(self, "_open_batch", None):
                raise RuntimeError("append_event outside of write batch")
        parent_id = parent_id if parent_id is not None else self.branch_head(project_id, branch_id)
        event_id = digest_event(project_id, branch_id, parent_id, etype, payload)
        exists = self.conn.execute(
            "SELECT 1 FROM events WHERE project_id=? AND event_id=?",
            (project_id, event_id),
        ).fetchone()
        if exists:
            return {"event_id": event_id, "dedup": True}
        seq = self._next_seq(project_id)
        ts = ts if ts is not None else time.time()
        self.conn.execute(
            "INSERT INTO events(project_id, seq, event_id, branch_id, parent_id,"
            " type, payload, ts, committed) VALUES (?,?,?,?,?,?,?,?,0)",
            (project_id, seq, event_id, branch_id, parent_id, etype,
             json.dumps(payload, ensure_ascii=False), ts),
        )
        self.conn.execute(
            "UPDATE branches SET head_id=? WHERE project_id=? AND branch_id=?",
            (event_id, project_id, branch_id),
        )
        return {"event_id": event_id, "dedup": False, "seq": seq}

    def commit_batch(self) -> None:
        with self._lock:
            if not getattr(self, "_open_batch", None):
                raise RuntimeError("commit_batch without open batch")
            project_id, batch_id = self._open_batch
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute("UPDATE events SET committed=1 WHERE committed=0")
                self.conn.execute("DELETE FROM write_batches WHERE batch_id=?", (batch_id,))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            self._open_batch = None

    def rollback_batch(self) -> None:
        with self._lock:
            if getattr(self, "_open_batch", None):
                project_id, batch_id = self._open_batch
                self.conn.execute("DELETE FROM events WHERE committed=0")
                self.conn.execute("DELETE FROM write_batches WHERE batch_id=?", (batch_id,))
                self._open_batch = None

    def crash_during_batch(self) -> None:
        """Test hook: lose the connection while rows are committed=0."""
        with self._lock:
            open_batch = getattr(self, "_open_batch", None)
            try:
                self.conn.close()
            except Exception:
                pass
            self._open_batch = None
            self.conn = sqlite3.connect(self.path, check_same_thread=False,
                                        isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            return self._recover_pending() if open_batch else 0

    # ---- lifecycle -------------------------------------------------------
    def create_project(self, project_id: str, name: str, *, branch_id: str = "main",
                       branch_name: str = "main", seed: dict | None = None,
                       ts: float | None = None) -> str:
        with self._lock:
            if self.get_project(project_id):
                raise ValueError(f"project {project_id} already exists")
            ts = ts if ts is not None else time.time()
            self.conn.execute(
                "INSERT INTO projects(project_id, name, created_at) VALUES (?,?,?)",
                (project_id, name, ts),
            )
            self.conn.execute(
                "INSERT INTO branches(project_id, branch_id, name, head_id, fork_seq)"
                " VALUES (?,?,?,?,0)",
                (project_id, branch_id, branch_name, None),
            )
            self.begin_batch(project_id)
            res = self.append_event(
                project_id, branch_id, "project_created",
                {"name": name, "seed": seed or {}}, parent_id=None, ts=ts,
            )
            self.commit_batch()
            return res["event_id"]

    def create_branch(self, project_id: str, name: str, *, from_event_id: str,
                      branch_id: str | None = None) -> str:
        """Fork: the new branch starts at an existing ancestor event."""
        with self._lock:
            anchor = self.get_event(project_id, from_event_id)
            if anchor is None:
                raise KeyError(f"event {from_event_id} not found")
            branch_id = branch_id or sid("br_", project_id, name, from_event_id)
            if self.conn.execute(
                "SELECT 1 FROM branches WHERE project_id=? AND branch_id=?",
                (project_id, branch_id),
            ).fetchone():
                raise ValueError(f"branch {branch_id} exists")
            self.conn.execute(
                "INSERT INTO branches(project_id, branch_id, name, head_id, fork_seq)"
                " VALUES (?,?,?,?,?)",
                (project_id, branch_id, name, from_event_id, anchor["seq"]),
            )
            return branch_id

    def get_event(self, project_id: str, event_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? AND event_id=? AND committed=1",
            (project_id, event_id),
        ).fetchone()
        return self._row_to_event(r) if r else None

    @staticmethod
    def _row_to_event(r: sqlite3.Row) -> dict:
        return {
            "seq": r["seq"], "event_id": r["event_id"], "branch_id": r["branch_id"],
            "parent_id": r["parent_id"], "type": r["type"],
            "payload": json.loads(r["payload"]), "ts": r["ts"],
        }

    def branch_events(self, project_id: str, branch_id: str, *, upto: str | None = None
                      ) -> list[dict]:
        """Committed events on this branch, including pre-fork history.

        Walking the parent chain yields exactly the events needed to replay
        the branch state, in seq order.
        """
        branch = self.conn.execute(
            "SELECT * FROM branches WHERE project_id=? AND branch_id=?",
            (project_id, branch_id),
        ).fetchone()
        if branch is None:
            raise KeyError(f"unknown branch {branch_id}")
        head = upto if upto is not None else branch["head_id"]
        return self.ancestry(project_id, head) if head else []

    def ancestry(self, project_id: str, event_id: str) -> list[dict]:
        """Full committed ancestor chain including pre-fork events."""
        rows = {
            r["event_id"]: r
            for r in self.conn.execute(
                "SELECT * FROM events WHERE project_id=? AND committed=1", (project_id,)
            ).fetchall()
        }
        out, cur, seen = [], event_id, set()
        while cur and cur in rows and cur not in seen:
            seen.add(cur)
            out.append(self._row_to_event(rows[cur]))
            cur = rows[cur]["parent_id"]
        return list(reversed(out))

    # ---- export / import -------------------------------------------------
    BUNDLE_FORMAT = "itc-event-bundle/1"

    def export_project(self, project_id: str) -> dict:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        branches = self.list_branches(project_id)
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? AND committed=1 ORDER BY seq",
            (project_id,),
        ).fetchall()
        events = [self._row_to_event(r) for r in rows]
        bundle = {
            "format": self.BUNDLE_FORMAT,
            "project": {"project_id": project_id, "name": project["name"],
                        "created_at": project["created_at"]},
            "branches": branches,
            "events": events,
        }
        body = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        bundle["checksum_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return bundle

    def import_project(self, bundle: dict, *, replace: bool = False) -> dict:
        self._validate_bundle(bundle)
        project = bundle["project"]
        pid = project["project_id"]
        with self._lock:
            if self.get_project(pid):
                if not replace:
                    raise ValueError(f"project {pid} already exists")
                self.conn.execute("DELETE FROM events WHERE project_id=?", (pid,))
                self.conn.execute("DELETE FROM branches WHERE project_id=?", (pid,))
                self.conn.execute("DELETE FROM projects WHERE project_id=?", (pid,))
                self.conn.execute("DELETE FROM write_batches WHERE project_id=?", (pid,))
            self.begin_batch(pid, "import_" + sid("", pid)[-12:])
            try:
                self.conn.execute(
                    "INSERT INTO projects(project_id, name, created_at) VALUES (?,?,?)",
                    (pid, project["name"], project["created_at"]),
                )
                for b in bundle["branches"]:
                    self.conn.execute(
                        "INSERT INTO branches(project_id, branch_id, name, head_id, fork_seq)"
                        " VALUES (?,?,?,?,?)",
                        (pid, b["branch_id"], b["name"], b["head_id"],
                         b.get("fork_seq", 0)),
                    )
                for ev in bundle["events"]:
                    self.conn.execute(
                        "INSERT INTO events(project_id, seq, event_id, branch_id, parent_id,"
                        " type, payload, ts, committed) VALUES (?,?,?,?,?,?,?,?,1)",
                        (pid, ev["seq"], ev["event_id"], ev["branch_id"], ev["parent_id"],
                         ev["type"], json.dumps(ev["payload"], ensure_ascii=False), ev["ts"]),
                    )
                self.commit_batch()
            except Exception:
                self.rollback_batch()
                raise
        return {"project_id": pid, "events": len(bundle["events"]),
                "branches": len(bundle["branches"])}

    @staticmethod
    def _validate_bundle(bundle: Any) -> None:
        from .errors import FormatCorrupt

        if not isinstance(bundle, dict):
            raise FormatCorrupt("bundle is not a JSON object")
        checksum = bundle.get("checksum_sha256")
        body_bundle = {k: v for k, v in bundle.items() if k != "checksum_sha256"}
        body = json.dumps(body_bundle, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        if not checksum or hashlib.sha256(body.encode("utf-8")).hexdigest() != checksum:
            raise FormatCorrupt("bundle checksum mismatch; refusing import")
        if bundle.get("format") != Store.BUNDLE_FORMAT:
            raise FormatCorrupt(f"unsupported bundle format {bundle.get('format')!r}")
        project = bundle.get("project") or {}
        if not project.get("project_id") or not project.get("name"):
            raise FormatCorrupt("bundle project missing identity")
        seqs = [e.get("seq") for e in bundle.get("events", [])]
        if seqs and (sorted(seqs) != list(range(1, len(seqs) + 1)) or len(set(seqs)) != len(seqs)):
            raise FormatCorrupt("bundle event sequence is not a gap-free permutation")
        ids = {e["event_id"] for e in bundle.get("events", [])}
        for ev in bundle.get("events", []):
            if ev.get("parent_id") and ev["parent_id"] not in ids and ev["parent_id"] is not None:
                # project_created root carries parent None; every other parent
                # must resolve inside the bundle.
                raise FormatCorrupt(f"event {ev['event_id']} parent missing in bundle")
        heads = {b.get("head_id") for b in bundle.get("branches", [])}
        for h in heads:
            if h is not None and h not in ids:
                raise FormatCorrupt("branch head missing in bundle")
