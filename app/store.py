"""Event-sourced SQLite storage.

Immutable materials vs mutable decisions are separated as different event
payloads.  Each event is chained (prev_hash) and stored through a staging
promotion transaction; a process death between staging and promotion is rolled
back on the next open (half-write recovery).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Optional

from .util import digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  project_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  seq INTEGER NOT NULL,
  id TEXT NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  blobs_json TEXT NOT NULL,
  prev_hash TEXT,
  hash TEXT NOT NULL,
  created_order INTEGER NOT NULL,
  PRIMARY KEY(project_id, branch, seq)
);
CREATE TABLE IF NOT EXISTS event_staging(
  token TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  seq INTEGER NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  blobs_json TEXT NOT NULL,
  prev_hash TEXT,
  created_order INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  media_type TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str = "data/itc.db"):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._recover_half_writes()

    # ------------------------------------------------------------- recovery
    def _recover_half_writes(self):
        with self._lock:
            stale = self.conn.execute("SELECT COUNT(*) c FROM event_staging").fetchone()["c"]
            self.conn.execute("DELETE FROM event_staging")
            self.conn.commit()
            self.last_recovered_staging = int(stale or 0)

    def close(self):
        with self._lock:
            self.conn.close()

    # --------------------------------------------------------------- blobs
    def put_blob(self, content: str, *, kind: str, media_type: str = "text/csv") -> str:
        bid = digest(content, length=24, prefix="blob_")
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO blobs(id,kind,content,media_type) VALUES(?,?,?,?)",
                (bid, kind, content, media_type))
            self.conn.commit()
        return bid

    def get_blob(self, bid: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM blobs WHERE id=?", (bid,)).fetchone()
        return None if row is None else dict(row)

    def all_blobs(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT id,kind,media_type FROM blobs ORDER BY id")]

    # -------------------------------------------------------------- events
    def _next_order(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(created_order),0)+1 n FROM events").fetchone()
        return int(row["n"])

    def _head(self, project_id: str, branch: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT hash FROM events WHERE project_id=? AND branch=? ORDER BY seq DESC LIMIT 1",
            (project_id, branch)).fetchone()
        return None if row is None else row["hash"]

    def append(self, project_id: str, branch: str, etype: str, payload: dict,
               *, blobs: Optional[dict[str, str]] = None, stage_only: bool = False) -> dict:
        """Append (or stage) one event. Returns the event row as dict."""
        blobs = blobs or {}
        with self._lock:
            seq_row = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 n FROM events WHERE project_id=? AND branch=?",
                (project_id, branch)).fetchone()
            seq = int(seq_row["n"])
            prev = self._head(project_id, branch)
            eid = digest(["event", project_id, branch, seq, etype, payload], length=14)
            hbody = {"id": eid, "project": project_id, "branch": branch, "seq": seq,
                     "type": etype, "payload": payload, "blobs": blobs, "prev": prev}
            h = digest(hbody, length=40)
            order = self._next_order()
            if stage_only:
                self.conn.execute(
                    """INSERT INTO event_staging(token,project_id,branch,seq,type,
                       payload_json,blobs_json,prev_hash,created_order)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (h, project_id, branch, seq, etype,
                     json.dumps(payload, ensure_ascii=False),
                     json.dumps(blobs, ensure_ascii=False), prev, order))
                self.conn.commit()
                return {"id": eid, "hash": h, "seq": seq, "branch": branch,
                        "type": etype, "payload": payload, "blobs": blobs,
                        "prev_hash": prev, "staged": True}
            self._promote_row(project_id, branch, seq, etype, payload, blobs, prev, order, h, eid)
            return {"id": eid, "hash": h, "seq": seq, "branch": branch,
                    "type": etype, "payload": payload, "blobs": blobs,
                    "prev_hash": prev, "staged": False}

    def _promote_row(self, project_id, branch, seq, etype, payload, blobs, prev, order, h, eid):
        with self.conn:  # atomic promotion
            self.conn.execute(
                """INSERT INTO events(project_id,branch,seq,id,type,payload_json,
                   blobs_json,prev_hash,hash,created_order)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (project_id, branch, seq, eid, etype,
                 json.dumps(payload, ensure_ascii=False),
                 json.dumps(blobs, ensure_ascii=False), prev, h, order))

    def promote_staged(self, token: str):
        with self._lock:
            row = self.conn.execute("SELECT * FROM event_staging WHERE token=?",
                                    (token,)).fetchone()
            if row is None:
                return None
            self._promote_row(
                row["project_id"], row["branch"], row["seq"], row["type"],
                json.loads(row["payload_json"]), json.loads(row["blobs_json"]),
                row["prev_hash"], row["created_order"], row["token"], row["token"])
            self.conn.execute("DELETE FROM event_staging WHERE token=?", (token,))

    def list_events(self, project_id: str, branch: Optional[str] = None):
        q = "SELECT * FROM events WHERE project_id=?"
        args: list[Any] = [project_id]
        if branch:
            q += " AND branch=?"
            args.append(branch)
        q += " ORDER BY seq"
        rows = []
        for r in self.conn.execute(q, args):
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            d["blobs"] = json.loads(d.pop("blobs_json"))
            rows.append(d)
        return rows

    def branches(self, project_id: str):
        return [r["branch"] for r in self.conn.execute(
            "SELECT DISTINCT branch FROM events WHERE project_id=? ORDER BY branch",
            (project_id,))]

    def projects(self):
        return [r["project_id"] for r in self.conn.execute(
            "SELECT DISTINCT project_id FROM events ORDER BY project_id")]

    def set_meta(self, key, value):
        self.conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key, default=None):
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if r is None else json.loads(r["value"])

    # ----------------------------------------------------------- raw export
    def snapshot(self):
        events = []
        for r in self.conn.execute("SELECT * FROM events ORDER BY created_order"):
            d = {k: r[k] for k in r.keys()}
            d["payload"] = json.loads(d.pop("payload_json"))
            d["blobs"] = json.loads(d.pop("blobs_json"))
            events.append(d)
        blobs = [dict(r) for r in self.conn.execute("SELECT * FROM blobs ORDER BY id")]
        return {"events": events, "blobs": blobs}

    def reset(self):
        with self._lock:
            self.conn.executescript(
                "DELETE FROM events; DELETE FROM event_staging; DELETE FROM blobs;")
            self.conn.commit()
