"""SQLite event store.

All domain state is derived by folding events. Writes happen inside batches;
a batch row starts 'open' and becomes 'committed' only after every event is
persisted. A crash in the middle leaves an 'open' batch behind; the next
open() recovers by dropping its events and recording a `store.recovered`
event — half writes are never half visible.

Export/import preserves identity: event ids, seq order and digests travel
with the file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .results import Diagnostic, FormatCorrupt, Ok, SemanticUnsolvable

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches(
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    note TEXT,
    ts REAL
);
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    batch_id TEXT,
    ts REAL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv(
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

EXPORT_FORMAT = "itc-review/1"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def event_digest(event_id: str, kind: str, payload) -> str:
    return sha(f"{event_id}|{kind}|{canonical(payload)}")


class _Writer:
    def __init__(self, store, batch_id):
        self.store = store
        self.batch_id = batch_id

    def append(self, kind, payload):
        eid = "ev-" + uuid.uuid4().hex[:12]
        ts = time.time()
        digest = event_digest(eid, kind, payload)
        cur = self.store.conn.execute(
            "INSERT INTO events(batch_id,id,ts,kind,payload,digest) VALUES (?,?,?,?,?,?)",
            (self.batch_id, eid, ts, kind, canonical(payload), digest),
        )
        return {"seq": cur.lastrowid, "id": eid, "kind": kind, "ts": ts,
                "payload": payload, "digest": digest}


class Store:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._recover()

    def close(self):
        self.conn.close()

    def _recover(self):
        rows = self.conn.execute(
            "SELECT id FROM batches WHERE status='open'").fetchall()
        if not rows:
            return
        dropped = []
        for row in rows:
            evs = self.conn.execute(
                "SELECT id FROM events WHERE batch_id=?", (row["id"],)).fetchall()
            dropped.extend(e["id"] for e in evs)
            self.conn.execute("DELETE FROM events WHERE batch_id=?", (row["id"],))
            self.conn.execute("UPDATE batches SET status='recovered' WHERE id=?",
                              (row["id"],))
        with self.batch("crash recovery") as w:
            w.append("store.recovered", {
                "recovered_batches": [r["id"] for r in rows],
                "dropped_event_ids": dropped,
            })

    @contextmanager
    def batch(self, note=""):
        bid = "batch-" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO batches(id,status,note,ts) VALUES (?,?,?,?)",
            (bid, "open", note, time.time()))
        writer = _Writer(self, bid)
        try:
            yield writer
        except BaseException:
            raise  # batch stays 'open'; the next open() recovers it
        else:
            self.conn.execute("UPDATE batches SET status='committed' WHERE id=?",
                              (bid,))

    def events(self, upto=None):
        sql = "SELECT seq,id,ts,kind,payload,digest FROM events"
        args = ()
        if upto is not None:
            sql += " WHERE seq<=?"
            args = (upto,)
        sql += " ORDER BY seq"
        out = []
        for row in self.conn.execute(sql, args):
            out.append({
                "seq": row["seq"], "id": row["id"], "ts": row["ts"],
                "kind": row["kind"], "payload": json.loads(row["payload"]),
                "digest": row["digest"],
            })
        return out

    def head_seq(self):
        row = self.conn.execute("SELECT MAX(seq) AS s FROM events").fetchone()
        return row["s"] or 0

    def is_empty(self):
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE kind != 'store.recovered'").fetchone()
        return row["c"] == 0

    def get_kv(self, key, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_kv(self, key, value):
        self.conn.execute(
            "INSERT INTO kv(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, canonical(value)))

    def export_dict(self):
        return {
            "format": EXPORT_FORMAT,
            "exported_head_seq": self.head_seq(),
            "kv": {k: self.get_kv(k) for k in ("project",)},
            "events": self.events(),
        }

    def export_file(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.export_dict(), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)  # atomic: no half-written export files
        return Ok(path)

    def import_dict(self, data):
        if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
            return FormatCorrupt(detail="not an itc-review/1 export", position="$")
        if not self.is_empty():
            return SemanticUnsolvable([Diagnostic(
                "identity_conflict",
                "import target is not empty; identities would collide",
                "store")])
        events = data.get("events")
        if not isinstance(events, list):
            return FormatCorrupt(detail="'events' must be a list", position="$.events")
        for i, ev in enumerate(events):
            try:
                if event_digest(ev["id"], ev["kind"], ev["payload"]) != ev["digest"]:
                    return FormatCorrupt(
                        detail="event digest mismatch (tampered or truncated)",
                        position=f"$.events[{i}]")
            except (KeyError, TypeError) as exc:
                return FormatCorrupt(detail=str(exc), position=f"$.events[{i}]")
        bid = "batch-import-" + uuid.uuid4().hex[:8]
        self.conn.execute(
            "INSERT INTO batches(id,status,note,ts) VALUES (?,?,?,?)",
            (bid, "committed", "import", time.time()))
        for ev in events:
            self.conn.execute(
                "INSERT INTO events(seq,id,batch_id,ts,kind,payload,digest)"
                " VALUES (?,?,?,?,?,?,?)",
                (ev["seq"], ev["id"], bid, ev["ts"], ev["kind"],
                 canonical(ev["payload"]), ev["digest"]))
        for key, value in (data.get("kv") or {}).items():
            if value is not None:
                self.set_kv(key, value)
        return Ok({"imported_events": len(events)})

    def import_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return FormatCorrupt(detail="file not found", position=path)
        except json.JSONDecodeError as exc:
            return FormatCorrupt(detail=str(exc),
                                 position=f"{path}:{exc.lineno}:{exc.colno}")
        return self.import_dict(data)
