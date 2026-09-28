"""SQLite persistence with explicit half-write recovery.

Writers run in one transaction that is either fully committed or rolled
back. Longer operations (project import) stage their payload in a journaled
file first; ``recover_pending`` removes staging left behind by a simulated
crash so a half-written bundle can never masquerade as a saved project.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from .errors import FormatCorrupt

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ops_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op_kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    staging_path TEXT,
    state TEXT NOT NULL CHECK (state IN ('pending', 'done')),
    created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.recover_pending()

    def close(self) -> None:
        self._conn.close()

    def _op(self, kind: str, ref: str, staging_path: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO ops_journal(op_kind, ref, staging_path, state, created_at) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (kind, ref, staging_path, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def _op_done(self, op_id: int) -> None:
        self._conn.execute("UPDATE ops_journal SET state='done' WHERE id=?", (op_id,))

    def recover_pending(self) -> list[dict[str, Any]]:
        """Roll half-written operations back. Called on every open."""
        rows = self._conn.execute(
            "SELECT id, op_kind, ref, staging_path FROM ops_journal WHERE state='pending'"
        ).fetchall()
        recovered = []
        for row in rows:
            if row["staging_path"] and os.path.exists(row["staging_path"]):
                os.remove(row["staging_path"])
            self._conn.execute("DELETE FROM ops_journal WHERE id=?", (row["id"],))
            recovered.append({"op_id": row["id"], "op_kind": row["op_kind"], "ref": row["ref"]})
        self._conn.commit()
        return recovered

    def list_projects(self) -> list[dict[str, str]]:
        rows = self._conn.execute("SELECT id, updated_at FROM projects ORDER BY id").fetchall()
        return [{"id": r["id"], "updated_at": r["updated_at"]} for r in rows]

    def load(self, project_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def save(self, project: dict[str, Any]) -> None:
        payload = json.dumps(project, ensure_ascii=False, sort_keys=True)
        op_id = self._op("save", project["id"])
        try:
            self._conn.execute(
                "INSERT INTO projects(id, payload, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (project["id"], payload, time.time()),
            )
            self._op_done(op_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def delete(self, project_id: str) -> None:
        op_id = self._op("delete", project_id)
        try:
            self._conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            self._op_done(op_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def stage_import(self, project_id: str, raw_bytes: bytes, staging_dir: str) -> str:
        os.makedirs(staging_dir, exist_ok=True)
        op_id = self._op("import", project_id)
        staging_path = os.path.join(staging_dir, f"import-{op_id}.itc.json")
        with open(staging_path, "wb") as fh:
            fh.write(raw_bytes)
        self._staged_op = op_id
        self._staged_path = staging_path
        return staging_path

    def commit_staged_import(self, project: dict[str, Any]) -> None:
        op_id = getattr(self, "_staged_op", None)
        staging_path = getattr(self, "_staged_path", None)
        if op_id is None:
            raise FormatCorrupt("no staged import to commit")
        self.save(project)
        if staging_path and os.path.exists(staging_path):
            os.remove(staging_path)
        self._op_done(op_id)
        self._conn.commit()
        self._staged_op = None
        self._staged_path = None
