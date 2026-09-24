"""SQLite persistence for the event-sourced project state.

Everything needed to reconstruct a project is stored in two tables:

* ``projects``  - identity/metadata and full serialised state;
* ``ingest_log`` - append marker rows used to demonstrate half-write recovery.

Writes happen inside an immediate transaction; a crash (or simulated fault)
mid-write leaves the previous committed state intact.  A separate
``expected_writes`` ledger lets ``recover`` distinguish three outcomes:
committed, rolled-back-half-write, and corrupt-on-disk.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .engine import Event, ProjectState, DerivedRecord
from .errors import Diagnostic, FormatCorrupt

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    state_json   TEXT NOT NULL,
    version      INTEGER NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT NOT NULL,
    label        TEXT NOT NULL,
    committed    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS expected_writes (
    token        TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    label        TEXT NOT NULL,
    done         INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = str(path)
    fresh = not os.path.exists(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def serialize_state(state: ProjectState) -> str:
    obj = {
        "project_id": state.project_id,
        "title": state.title,
        "events": [e.to_dict() for e in state.events],
        "decisions": state.decisions,
        "derived": [r.to_dict() for r in state.derived],
        "branches": state.branches,
        "head": state.head,
        "runs": {rid: {"streams": sorted(meta["streams"])}
                 for rid, meta in state.runs.items()},
        "diagnostics": state.diagnostics,
        "materials": _serialize_materials(state),
    }
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _serialize_materials(state: ProjectState) -> dict:
    out = {}
    for mid, streams in state.materials.items():
        out[mid] = {}
        for use_stream, entry in streams.items():
            out[mid][use_stream] = {
                "current": entry["current"],
                "versions": {str(v): rev.to_event_dict()
                             for v, rev in entry["versions"].items()},
            }
    return out


def deserialize_state(text: str) -> ProjectState:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FormatCorrupt(
            "stored state is not valid JSON",
            [Diagnostic("storage:bad_json", str(exc), "state_json", None)])
    state = ProjectState(obj["project_id"], obj["title"])
    state.events = [Event.from_dict(e) for e in obj.get("events", [])]
    state.decisions = obj.get("decisions", [])
    state.derived = [DerivedRecord.from_dict(r) for r in obj.get("derived", [])]
    state.branches = {k: list(v) for k, v in obj.get("branches", {}).items()}
    state.head = dict(obj.get("head", {}))
    state.diagnostics = obj.get("diagnostics", [])
    for rid, meta in obj.get("runs", {}).items():
        state.runs[rid] = {"streams": set(meta.get("streams", []))}
    for mid, streams in obj.get("materials", {}).items():
        state.materials[mid] = {}
        for use_stream, entry in streams.items():
            versions = {int(v): _material_from(vv)
                        for v, vv in entry["versions"].items()}
            state.materials[mid][use_stream] = {"current": entry["current"],
                                                "versions": versions}
    return state


def _material_from(obj: dict):
    from .materials import rehydrate_revision
    return rehydrate_revision(obj)


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #

class Store:
    def __init__(self, path: str | Path = "itc_review.db"):
        self.path = str(path)
        self.conn = connect(self.path)

    def save_project(self, state: ProjectState, *, version: int | None = None) -> int:
        text = serialize_state(state)
        row = self.conn.execute(
            "SELECT version FROM projects WHERE project_id=?", (state.project_id,)).fetchone()
        cur_version = row["version"] if row else 0
        next_version = version or cur_version + 1
        self.conn.execute(
            "INSERT INTO projects(project_id,title,state_json,version,updated_at) "
            "VALUES(?,?,?,?,datetime('now')) "
            "ON CONFLICT(project_id) DO UPDATE SET title=excluded.title,"
            "state_json=excluded.state_json,version=excluded.version,"
            "updated_at=datetime('now')",
            (state.project_id, state.title, text, next_version))
        self.conn.commit()
        return next_version

    def load_project(self, project_id: str) -> ProjectState:
        row = self.conn.execute(
            "SELECT state_json FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return deserialize_state(row["state_json"])

    def list_projects(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT project_id,title,version,updated_at FROM projects "
            "ORDER BY project_id").fetchall()
        return [dict(r) for r in rows]

    def delete_project(self, project_id: str) -> None:
        self.conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
        self.conn.execute("DELETE FROM ingest_log WHERE project_id=?", (project_id,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- half-write recovery ------------------------------------------------ #
    def begin_ingest(self, token: str, project_id: str, label: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO expected_writes(token,project_id,label,done)"
            " VALUES(?,?,?,0)", (token, project_id, label))
        self.conn.execute(
            "INSERT INTO ingest_log(project_id,label,committed) VALUES(?,?,0)",
            (project_id, label))
        self.conn.commit()

    def commit_ingest(self, token: str) -> None:
        self.conn.execute(
            "UPDATE ingest_log SET committed=1 WHERE id="
            "(SELECT MAX(id) FROM ingest_log WHERE project_id="
            "(SELECT project_id FROM expected_writes WHERE token=?))", (token,))
        self.conn.execute("UPDATE expected_writes SET done=1 WHERE token=?", (token,))
        self.conn.commit()

    def pending_ingests(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT w.token,w.project_id,w.label,w.done,"
            "COALESCE((SELECT MAX(committed) FROM ingest_log l "
            "WHERE l.project_id=w.project_id AND l.label=w.label),0) AS committed "
            "FROM expected_writes w WHERE w.done=0 ORDER BY w.token").fetchall()
        return [dict(r) for r in rows]

    def recover(self) -> dict:
        """Rollback bookkeeping for interrupted writes and report outcome.

        Returns a structured verdict:
        ``{"status": "clean" | "half_write_rolled_back" | "corrupt", ...}``.
        A half write (expected marker exists, no commit) is rolled back;
        an unreadable project row is ``corrupt`` (a distinct outcome).
        """
        pending = self.pending_ingests()
        rolled = []
        for item in pending:
            rolled.append(item)
            self.conn.execute("DELETE FROM expected_writes WHERE token=?",
                              (item["token"],))
        self.conn.commit()
        corrupt = []
        for row in self.conn.execute("SELECT project_id,state_json FROM projects"):
            try:
                json.loads(row["state_json"])
            except Exception:
                corrupt.append(row["project_id"])
        if corrupt:
            return {"status": "corrupt", "projects": corrupt, "rolled_back": rolled}
        if rolled:
            return {"status": "half_write_rolled_back", "projects": [],
                    "rolled_back": rolled}
        return {"status": "clean", "projects": [], "rolled_back": []}

