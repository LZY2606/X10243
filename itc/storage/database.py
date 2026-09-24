"""SQLite persistence with atomic writes, crash/half-write recovery and an
identity/order-preserving export/import bundle."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time

from domain.events import Event
from domain.state import DEFAULT_BRANCH, Project

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS branches(
  branch TEXT PRIMARY KEY,
  info TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  branch TEXT NOT NULL,
  seq INTEGER NOT NULL,
  event_id TEXT NOT NULL,
  parent_event_id TEXT,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  inputs_digest TEXT,
  parent_dependencies TEXT NOT NULL,
  author TEXT NOT NULL,
  note TEXT NOT NULL,
  epoch_us INTEGER NOT NULL,
  PRIMARY KEY(branch, seq)
);
CREATE TABLE IF NOT EXISTS write_log(
  tx_id TEXT PRIMARY KEY,
  started_us INTEGER NOT NULL,
  description TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self.recovery_report = self.recover()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript(SCHEMA)

    # ---- recovery -----------------------------------------------------------
    def recover(self) -> dict:
        """Roll back any transaction left half-written by a crash."""
        rows = self.conn.execute(
            "SELECT tx_id, started_us, description FROM write_log").fetchall()
        report = {"recovered_tx": [], "rolled_back_events": 0}
        if rows:
            for r in rows:
                report["recovered_tx"].append(dict(r))
            with self.conn:
                self.conn.execute("DELETE FROM write_log")
        return report

    def _begin(self, tx_id, description):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO write_log(tx_id,started_us,description)"
                " VALUES(?,?,?)",
                (tx_id, time.time_ns() // 1000, description))

    def _commit(self, tx_id):
        with self.conn:
            self.conn.execute("DELETE FROM write_log WHERE tx_id=?", (tx_id,))

    # ---- project ------------------------------------------------------------
    def save_project(self, project: Project, description="save"):
        tx = f"tx-{time.time_ns()}"
        self._begin(tx, description)
        try:
            with self.conn:
                for branch, info in project.branches.items():
                    self.conn.execute(
                        "INSERT OR REPLACE INTO branches(branch,info)"
                        " VALUES(?,?)", (branch, json.dumps(info)))
                self.conn.execute("DELETE FROM events")
                for e in sorted(project.events,
                                key=lambda x: (x.branch, x.seq)):
                    self._insert_event(e)
            self._commit(tx)
        except Exception:
            self.recover()
            raise

    def _insert_event(self, e: Event):
        self.conn.execute(
            "INSERT INTO events(branch,seq,event_id,parent_event_id,type,"
            "payload,content_hash,inputs_digest,parent_dependencies,author,"
            "note,epoch_us) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (e.branch, e.seq, e.event_id, e.parent_event_id, e.type,
             json.dumps(e.payload, ensure_ascii=False), e.content_hash,
             e.inputs_digest, json.dumps(e.parent_dependencies), e.author,
             e.note, e.epoch_us))

    def load_project(self) -> Project:
        project = Project()
        for r in self.conn.execute("SELECT branch,info FROM branches"):
            project.branches[r["branch"]] = json.loads(r["info"])
        if DEFAULT_BRANCH not in project.branches:
            project.init_branch(DEFAULT_BRANCH)
        for r in self.conn.execute("SELECT * FROM events ORDER BY branch,seq"):
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            d["parent_dependencies"] = json.loads(d["parent_dependencies"])
            d.pop("branch", None); d.pop("seq", None)
            e = Event(branch=r["branch"], seq=r["seq"], **{
                k: d[k] for k in d if k in Event.__dataclass_fields__})
            project.events.append(e)
        return project

    # ---- export / import ----------------------------------------------------
    def export_bundle(self) -> dict:
        events = []
        for r in self.conn.execute(
                "SELECT * FROM events ORDER BY branch,seq"):
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            d["parent_dependencies"] = json.loads(d["parent_dependencies"])
            events.append(d)
        branches = [{"branch": r["branch"], "info": json.loads(r["info"])}
                    for r in self.conn.execute("SELECT * FROM branches")]
        return {
            "format": "itc-review-bundle",
            "format_version": 1,
            "exported_us": time.time_ns() // 1000,
            "branches": branches,
            "events": events,
        }

    def import_bundle(self, bundle: dict) -> dict:
        """Atomically replace database contents with *bundle*.

        A corrupt bundle aborts before existing committed data is touched.
        """
        events = _validate_bundle(bundle)
        tx = f"import-{time.time_ns()}"
        self._begin(tx, "import bundle")
        try:
            with self.conn:
                self.conn.execute("DELETE FROM events")
                self.conn.execute("DELETE FROM branches")
                for b in bundle["branches"]:
                    self.conn.execute(
                        "INSERT INTO branches(branch,info) VALUES(?,?)",
                        (b["branch"], json.dumps(b["info"])))
                for d in events:
                    e = Event(**{k: d[k] for k in Event.__dataclass_fields__})
                    self._insert_event(e)
            self._commit(tx)
            return {"imported_events": len(events),
                    "imported_branches": len(bundle["branches"])}
        except Exception:
            self.recover()
            raise


def _validate_bundle(bundle: dict) -> list[dict]:
    from domain.diagnostics import FORMAT_CORRUPT
    if not isinstance(bundle, dict):
        raise BundleError(FORMAT_CORRUPT, "bundle is not an object")
    if bundle.get("format") != "itc-review-bundle":
        raise BundleError(FORMAT_CORRUPT, "unknown bundle format")
    events = bundle.get("events")
    if not isinstance(events, list):
        raise BundleError(FORMAT_CORRUPT, "events missing")
    seen = set()
    required = {"branch", "seq", "event_id", "type", "payload",
                "content_hash", "parent_dependencies"}
    for d in events:
        if not required.issubset(d):
            raise BundleError(FORMAT_CORRUPT, "event missing required fields")
        key = (d["branch"], d["seq"])
        if key in seen:
            raise BundleError(FORMAT_CORRUPT,
                              f"duplicate event position {key}")
        seen.add(key)
        if digest_payload(d["payload"]) != d["content_hash"]:
            raise BundleError(FORMAT_CORRUPT,
                              f"event {d['event_id']} content hash mismatch")
        d.setdefault("parent_event_id", None)
        d.setdefault("inputs_digest", None)
        d.setdefault("author", "reviewer")
        d.setdefault("note", "")
        d.setdefault("epoch_us", 0)
    return events


def digest_payload(payload) -> str:
    from domain.events import digest
    return digest(payload)


class BundleError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def atomic_write_json(path: str, bundle: dict):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".itc-export-", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
