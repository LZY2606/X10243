from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = _connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_seq INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS branches (
                    project_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    parent_branch_id TEXT,
                    fork_seq INTEGER,
                    name TEXT NOT NULL,
                    PRIMARY KEY (project_id, branch_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    project_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (project_id, branch_id, seq),
                    UNIQUE (project_id, branch_id, event_id)
                );
                """
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        self.conn.close()

    def create_project(self, project_id: str, name: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO projects(project_id, name, created_seq) VALUES (?, ?, 0)", (project_id, name)
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO branches(project_id, branch_id, parent_branch_id, fork_seq, name) VALUES (?, 'main', NULL, NULL, 'main')",
                (project_id,),
            )

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY project_id").fetchall()
        return [dict(row) for row in rows]

    def list_branches(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM branches WHERE project_id=? ORDER BY branch_id", (project_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def create_branch(self, project_id: str, branch_id: str, name: str, parent_branch_id: str, fork_seq: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO branches(project_id, branch_id, parent_branch_id, fork_seq, name) VALUES (?, ?, ?, ?, ?)",
                (project_id, branch_id, parent_branch_id, fork_seq, name),
            )

    def next_seq(self, project_id: str, branch_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM events WHERE project_id=? AND branch_id=?",
            (project_id, branch_id),
        ).fetchone()
        return int(row["max_seq"]) + 1

    def append_event(
        self,
        project_id: str,
        branch_id: str,
        event_type: str,
        payload: dict[str, Any],
        event_id: str | None = None,
        seq: int | None = None,
    ) -> dict[str, Any]:
        event_id = event_id or f"evt_{uuid.uuid4().hex[:16]}"
        seq = seq if seq is not None else self.next_seq(project_id, branch_id)
        record = {
            "project_id": project_id,
            "branch_id": branch_id,
            "seq": seq,
            "event_id": event_id,
            "event_type": event_type,
            "payload": payload,
        }
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO events(project_id, branch_id, seq, event_id, event_type, payload) VALUES (?, ?, ?, ?, ?, ?)",
                    (project_id, branch_id, seq, event_id, event_type, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"event identity or sequence conflict: {exc}") from exc
        return record

    def events(self, project_id: str, branch_id: str, upto_seq: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM events WHERE project_id=? AND branch_id=?"
        params: list[Any] = [project_id, branch_id]
        if upto_seq is not None:
            query += " AND seq<=?"
            params.append(upto_seq)
        query += " ORDER BY seq"
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "project_id": row["project_id"],
                "branch_id": row["branch_id"],
                "seq": row["seq"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def export_project(self, project_id: str) -> dict[str, Any]:
        project = self.conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if project is None:
            raise KeyError(f"unknown project {project_id}")
        branches = self.list_branches(project_id)
        return {
            "format": "itc-review-export",
            "format_version": 1,
            "project": dict(project),
            "branches": branches,
            "events": {
                branch["branch_id"]: [
                    {
                        "seq": event["seq"],
                        "event_id": event["event_id"],
                        "event_type": event["event_type"],
                        "payload": event["payload"],
                    }
                    for event in self.events(project_id, branch["branch_id"])
                ]
                for branch in branches
            },
        }

    def import_project(self, bundle: dict[str, Any]) -> str:
        if bundle.get("format") != "itc-review-export":
            raise ValueError("FORMAT_CORRUPT: unsupported export format")
        project = bundle.get("project") or {}
        project_id = project.get("project_id")
        if not project_id:
            raise ValueError("FORMAT_CORRUPT: project_id missing")
        with self.conn:
            self.conn.execute("DELETE FROM events WHERE project_id=?", (project_id,))
            self.conn.execute("DELETE FROM branches WHERE project_id=?", (project_id,))
            self.conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
            self.conn.execute(
                "INSERT INTO projects(project_id, name, created_seq) VALUES (?, ?, ?)",
                (project_id, project.get("name", project_id), int(project.get("created_seq", 0))),
            )
            for branch in bundle.get("branches", []):
                self.conn.execute(
                    "INSERT INTO branches(project_id, branch_id, parent_branch_id, fork_seq, name) VALUES (?, ?, ?, ?, ?)",
                    (project_id, branch["branch_id"], branch.get("parent_branch_id"), branch.get("fork_seq"), branch.get("name", branch["branch_id"])),
                )
            for branch_id, events in (bundle.get("events") or {}).items():
                for event in events:
                    self.conn.execute(
                        "INSERT INTO events(project_id, branch_id, seq, event_id, event_type, payload) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            project_id,
                            branch_id,
                            int(event["seq"]),
                            event["event_id"],
                            event["event_type"],
                            json.dumps(event["payload"], ensure_ascii=False, sort_keys=True),
                        ),
                    )
        return project_id

    def delete_project(self, project_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM events WHERE project_id=?", (project_id,))
            self.conn.execute("DELETE FROM branches WHERE project_id=?", (project_id,))
            self.conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
