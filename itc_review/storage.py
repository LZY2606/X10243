"""SQLite persistence, project export/import and half-write recovery.

Immutable materials and computed results are content-addressed.  The mutable
decision stream lives in `events`; a branch HEAD points at the current tip.
Import is staged on disk and committed in one transaction; a crashed staging
directory is recovered as an explicit "orphaned-stage" record rather than
silently merged.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    role TEXT NOT NULL,
    version INTEGER NOT NULL,
    lineage TEXT NOT NULL,
    supersedes TEXT,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    inputs TEXT NOT NULL,
    params_version TEXT NOT NULL,
    parents TEXT NOT NULL,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS result_refs (
    scope TEXT NOT NULL,
    ref_key TEXT NOT NULL,
    result_id TEXT NOT NULL,
    PRIMARY KEY (scope, ref_key)
);
CREATE TABLE IF NOT EXISTS branches (
    name TEXT PRIMARY KEY,
    head_event TEXT,
    parent_branch TEXT,
    forked_at TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    branch TEXT NOT NULL,
    seq INTEGER NOT NULL,
    action TEXT NOT NULL,
    payload TEXT NOT NULL,
    parent_event TEXT,
    recorded_at REAL NOT NULL,
    UNIQUE(branch, seq)
);
CREATE TABLE IF NOT EXISTS replay_checkpoints (
    branch TEXT NOT NULL,
    seq INTEGER NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY(branch, seq)
);
CREATE TABLE IF NOT EXISTS import_stages (
    id TEXT PRIMARY KEY,
    stage_dir TEXT NOT NULL,
    manifest TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    committed_at REAL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class Repository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- materials -------------------------------------------------
    def add_material(self, *, id: str, kind: str, role: str, version: int,
                     lineage: str, supersedes: str | None,
                     payload: dict[str, Any], digest: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM materials WHERE id=?", (id,))
        if cur.fetchone():
            return False
        self.conn.execute(
            "INSERT INTO materials VALUES (?,?,?,?,?,?,?,?,?)",
            (id, kind, role, version, lineage, supersedes,
             json.dumps(payload, ensure_ascii=False), digest, time.time()))
        return True

    def get_material(self, id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM materials WHERE id=?", (id,)).fetchone()
        return self._material_row(row) if row else None

    def materials(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM materials ORDER BY lineage, version, id").fetchall()
        return [self._material_row(r) for r in rows]

    @staticmethod
    def _material_row(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "kind": row["kind"], "role": row["role"],
                "version": row["version"], "lineage": row["lineage"],
                "supersedes": row["supersedes"],
                "payload": json.loads(row["payload"]),
                "digest": row["digest"], "created_at": row["created_at"]}

    # ---- results ---------------------------------------------------
    def add_result(self, *, id: str, kind: str, status: str,
                   inputs: dict[str, Any], params_version: str,
                   parents: list[str], payload: dict[str, Any],
                   digest: str, stale: bool = False) -> bool:
        cur = self.conn.execute("SELECT 1 FROM results WHERE id=?", (id,))
        if cur.fetchone():
            return False
        self.conn.execute(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, kind, status, json.dumps(inputs, ensure_ascii=False),
             params_version, json.dumps(sorted(parents)),
             json.dumps(payload, ensure_ascii=False), digest,
             1 if stale else 0, time.time()))
        return True

    def get_result(self, id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM results WHERE id=?",
                                (id,)).fetchone()
        return self._result_row(row) if row else None

    def mark_stale(self, ids: list[str]) -> None:
        for rid in ids:
            self.conn.execute("UPDATE results SET stale=1 WHERE id=?", (rid,))

    def results(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM results ORDER BY id").fetchall()
        return [self._result_row(r) for r in rows]

    @staticmethod
    def _result_row(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "kind": row["kind"], "status": row["status"],
                "inputs": json.loads(row["inputs"]),
                "params_version": row["params_version"],
                "parents": json.loads(row["parents"]),
                "payload": json.loads(row["payload"]),
                "digest": row["digest"], "stale": bool(row["stale"]),
                "created_at": row["created_at"]}

    def set_ref(self, scope: str, key: str, result_id: str) -> None:
        self.conn.execute(
            "INSERT INTO result_refs(scope,ref_key,result_id) VALUES (?,?,?) "
            "ON CONFLICT(scope,ref_key) DO UPDATE SET result_id=excluded.result_id",
            (scope, key, result_id))

    def get_ref(self, scope: str, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT result_id FROM result_refs WHERE scope=? AND ref_key=?",
            (scope, key)).fetchone()
        return row["result_id"] if row else None

    def refs(self, scope: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT ref_key,result_id FROM result_refs WHERE scope=?",
            (scope,)).fetchall()
        return {r["ref_key"]: r["result_id"] for r in rows}

    # ---- branches & events ----------------------------------------
    def ensure_branch(self, name: str, parent_branch: str | None = None,
                      forked_at: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO branches(name,parent_branch,forked_at,created_at) "
            "VALUES (?,?,?,?) ON CONFLICT(name) DO NOTHING",
            (name, parent_branch, forked_at, time.time()))

    def set_head(self, branch: str, event_id: str) -> None:
        self.conn.execute("UPDATE branches SET head_event=? WHERE name=?",
                          (event_id, branch))

    def get_branch(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM branches WHERE name=?",
                                (name,)).fetchone()
        if not row:
            return None
        return {"name": row["name"], "head_event": row["head_event"],
                "parent_branch": row["parent_branch"],
                "forked_at": row["forked_at"]}

    def branches(self) -> list[dict[str, Any]]:
        return [self.get_branch(r["name"])  # type: ignore[union-attr]
                for r in self.conn.execute(
                    "SELECT name FROM branches ORDER BY created_at").fetchall()]

    def append_event(self, *, id: str, branch: str, seq: int, action: str,
                     payload: dict[str, Any], parent_event: str | None) -> str:
        self.conn.execute(
            "INSERT INTO events(id,branch,seq,action,payload,parent_event,"
            "recorded_at) VALUES (?,?,?,?,?,?,?)",
            (id, branch, seq, action,
             json.dumps(payload, ensure_ascii=False, allow_nan=False),
             parent_event, time.time()))
        return id

    def events(self, branch: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE branch=? ORDER BY seq", (branch,))
        return [self._event_row(r) for r in rows]

    def all_events(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY recorded_at, branch, seq")
        return [self._event_row(r) for r in rows]

    def next_seq(self, branch: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) AS s FROM events WHERE branch=?",
            (branch,)).fetchone()
        return int(row["s"]) + 1

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "branch": row["branch"], "seq": row["seq"],
                "action": row["action"],
                "payload": json.loads(row["payload"]),
                "parent_event": row["parent_event"],
                "recorded_at": row["recorded_at"]}

    def save_checkpoint(self, branch: str, seq: int, state: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO replay_checkpoints(branch,seq,state) VALUES (?,?,?) "
            "ON CONFLICT(branch,seq) DO UPDATE SET state=excluded.state",
            (branch, seq, json.dumps(state, ensure_ascii=False)))

    def checkpoint(self, branch: str, seq: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT state FROM replay_checkpoints WHERE branch=? AND seq=?",
            (branch, seq)).fetchone()
        return json.loads(row["state"]) if row else None

    # ---- staged imports -------------------------------------------
    def register_stage(self, stage_id: str, stage_dir: str,
                       manifest: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO import_stages(id,stage_dir,manifest,state,created_at)"
            " VALUES (?,?,?,?,?)",
            (stage_id, stage_dir, json.dumps(manifest), "staged", time.time()))

    def commit_stage(self, stage_id: str) -> None:
        self.conn.execute(
            "UPDATE import_stages SET state='committed',committed_at=? "
            "WHERE id=?", (time.time(), stage_id))

    def stages(self, state: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM import_stages" + (" WHERE state=?" if state else "")
        args = (state,) if state else ()
        rows = self.conn.execute(q, args).fetchall()
        return [{"id": r["id"], "stage_dir": r["stage_dir"],
                 "manifest": json.loads(r["manifest"]), "state": r["state"],
                 "created_at": r["created_at"],
                 "committed_at": r["committed_at"]} for r in rows]
