"""SQLite persistence: immutable materials, append-only events, derived cache.

Layering rules:
  - materials are immutable; a "revision" inserts a new (material_id, version) row
  - events are append-only and carry parent references (machine-readable provenance)
  - derived rows record inputs digest, parameter version and parent dependencies

Half-write recovery:
  - every multi-row write happens inside one transaction
  - meta['committed_seq'] is updated in the same transaction as the events it covers
  - on open, a mismatch between max(events.seq) and committed_seq means a torn
    write: the store rolls forward by deleting rows beyond the committed prefix
  - PRAGMA quick_check failure raises ItcFailure(FORMAT_CORRUPT)
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager

from .diagnostics import ItcFailure, FORMAT_CORRUPT, Diagnostic

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS materials(
  material_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  event_seq INTEGER NOT NULL,
  PRIMARY KEY(material_id, version)
);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL,
  type TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  payload TEXT NOT NULL,
  parents TEXT NOT NULL DEFAULT '[]',
  created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS derived(
  derived_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  inputs_digest TEXT NOT NULL,
  params_version TEXT NOT NULL,
  parents TEXT NOT NULL,
  payload TEXT NOT NULL,
  event_seq INTEGER NOT NULL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        existed = os.path.exists(path) and os.path.getsize(path) > 0
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if existed:
            self._recover()
        self.conn.executescript(SCHEMA)
        if self._get_meta("committed_seq") is None:
            with self.transaction():
                self._set_meta("committed_seq", "0")
                self._set_meta("schema", "itc/1")

    # -- integrity ---------------------------------------------------------
    def _recover(self) -> None:
        try:
            row = self.conn.execute("PRAGMA quick_check").fetchone()
        except sqlite3.DatabaseError as exc:
            raise ItcFailure(FORMAT_CORRUPT, f"database cannot be read: {exc}") from exc
        if not row or row[0] != "ok":
            raise ItcFailure(
                FORMAT_CORRUPT,
                f"database integrity check failed: {row[0] if row else 'unreadable'}",
                [Diagnostic(code="db_corrupt", message=str(row[0] if row else "unreadable"), path=self.path)],
            )
        try:
            committed = self._get_meta("committed_seq")
            max_seq = self.conn.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
        except sqlite3.DatabaseError:
            return  # fresh/empty file; schema will be created
        if committed is not None and max_seq > int(committed):
            # torn write: drop everything beyond the committed prefix
            with self.transaction():
                self.conn.execute("DELETE FROM events WHERE seq > ?", (int(committed),))
                self.conn.execute("DELETE FROM materials WHERE event_seq > ?", (int(committed),))
                self.conn.execute("DELETE FROM derived WHERE event_seq > ?", (int(committed),))

    @contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN")
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- meta --------------------------------------------------------------
    def _get_meta(self, key: str):
        try:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        except sqlite3.DatabaseError:
            return None
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))

    # -- events ------------------------------------------------------------
    def next_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()
        return int(row[0]) + 1

    def append_event(self, event: dict, materials: list = None, derived: list = None) -> int:
        """Append one event plus any material/derived rows atomically."""
        with self.transaction():
            seq = self.next_seq()
            self.conn.execute(
                "INSERT INTO events(seq,event_id,type,branch,payload,parents,created) VALUES(?,?,?,?,?,?,?)",
                (seq, event["event_id"], event["type"], event.get("branch", "main"),
                 json.dumps(event.get("payload", {}), sort_keys=True),
                 json.dumps(event.get("parents", []), sort_keys=True), event["created"]),
            )
            for m in materials or []:
                self.conn.execute(
                    "INSERT INTO materials(material_id,version,kind,payload,event_seq) VALUES(?,?,?,?,?)",
                    (m["material_id"], m["version"], m["kind"],
                     json.dumps(m["payload"], sort_keys=True), seq))
            for d in derived or []:
                self.conn.execute(
                    "INSERT OR REPLACE INTO derived(derived_id,kind,inputs_digest,params_version,parents,payload,event_seq)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (d["derived_id"], d["kind"], d["inputs_digest"], d["params_version"],
                     json.dumps(d["parents"], sort_keys=True),
                     json.dumps(d["payload"], sort_keys=True), seq))
            self._set_meta("committed_seq", str(seq))
        return seq

    def events(self, at_seq: int = None) -> list:
        q = "SELECT * FROM events"
        args = ()
        if at_seq is not None:
            q += " WHERE seq <= ?"
            args = (at_seq,)
        q += " ORDER BY seq"
        return [self._row_to_event(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _row_to_event(r) -> dict:
        return {"seq": r["seq"], "event_id": r["event_id"], "type": r["type"],
                "branch": r["branch"], "payload": json.loads(r["payload"]),
                "parents": json.loads(r["parents"]), "created": r["created"]}

    # -- materials ---------------------------------------------------------
    def material_versions(self, material_id: str, at_seq: int = None) -> list:
        q = "SELECT * FROM materials WHERE material_id=?"
        args: list = [material_id]
        if at_seq is not None:
            q += " AND event_seq <= ?"
            args.append(at_seq)
        q += " ORDER BY version"
        return [{"material_id": r["material_id"], "version": r["version"], "kind": r["kind"],
                 "payload": json.loads(r["payload"]), "event_seq": r["event_seq"]}
                for r in self.conn.execute(q, args)]

    def current_materials(self, at_seq: int = None) -> dict:
        """material_id -> latest version row visible at `at_seq` (stable order by id)."""
        q = "SELECT * FROM materials"
        args = ()
        if at_seq is not None:
            q += " WHERE event_seq <= ?"
            args = (at_seq,)
        q += " ORDER BY material_id, version"
        out: dict = {}
        for r in self.conn.execute(q, args):
            out[r["material_id"]] = {"material_id": r["material_id"], "version": r["version"],
                                     "kind": r["kind"], "payload": json.loads(r["payload"]),
                                     "event_seq": r["event_seq"]}
        return dict(sorted(out.items()))

    # -- derived -----------------------------------------------------------
    def derived_all(self, at_seq: int = None) -> list:
        q = "SELECT * FROM derived"
        args = ()
        if at_seq is not None:
            q += " WHERE event_seq <= ?"
            args = (at_seq,)
        q += " ORDER BY derived_id"
        return [{"derived_id": r["derived_id"], "kind": r["kind"],
                 "inputs_digest": r["inputs_digest"], "params_version": r["params_version"],
                 "parents": json.loads(r["parents"]), "payload": json.loads(r["payload"]),
                 "event_seq": r["event_seq"]}
                for r in self.conn.execute(q, args)]

    def derived_get(self, derived_id: str):
        r = self.conn.execute("SELECT * FROM derived WHERE derived_id=?", (derived_id,)).fetchone()
        if not r:
            return None
        return {"derived_id": r["derived_id"], "kind": r["kind"],
                "inputs_digest": r["inputs_digest"], "params_version": r["params_version"],
                "parents": json.loads(r["parents"]), "payload": json.loads(r["payload"]),
                "event_seq": r["event_seq"]}

    # -- export / import ---------------------------------------------------
    def export_bundle(self) -> dict:
        return {
            "format": "itc-project/1",
            "materials": [m for mid in sorted(self.current_materials())
                          for m in self.material_versions(mid)],
            "events": self.events(),
            "derived": self.derived_all(),
        }

    def import_bundle(self, bundle: dict) -> None:
        """All-or-nothing import preserving identity, order and provenance."""
        with self.transaction():
            for e in bundle["events"]:
                self.conn.execute(
                    "INSERT INTO events(seq,event_id,type,branch,payload,parents,created) VALUES(?,?,?,?,?,?,?)",
                    (e["seq"], e["event_id"], e["type"], e.get("branch", "main"),
                     json.dumps(e["payload"], sort_keys=True),
                     json.dumps(e.get("parents", []), sort_keys=True), e["created"]))
            for m in bundle["materials"]:
                self.conn.execute(
                    "INSERT INTO materials(material_id,version,kind,payload,event_seq) VALUES(?,?,?,?,?)",
                    (m["material_id"], m["version"], m["kind"],
                     json.dumps(m["payload"], sort_keys=True), m["event_seq"]))
            for d in bundle["derived"]:
                self.conn.execute(
                    "INSERT INTO derived(derived_id,kind,inputs_digest,params_version,parents,payload,event_seq)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (d["derived_id"], d["kind"], d["inputs_digest"], d["params_version"],
                     json.dumps(d["parents"], sort_keys=True),
                     json.dumps(d["payload"], sort_keys=True), d["event_seq"]))
            max_seq = bundle["events"][-1]["seq"] if bundle["events"] else 0
            self._set_meta("committed_seq", str(max_seq))
            self._set_meta("schema", "itc/1")

    def close(self) -> None:
        self.conn.close()


def atomic_write_json(path: str, obj) -> None:
    """Crash-safe file write: tmp file + fsync + rename."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
