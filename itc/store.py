"""SQLite store: separately versioned immutable materials, append-only events,
derived results with provenance, staged writes for half-write recovery,
and identity-preserving export/import."""
from __future__ import annotations

import json
import sqlite3

from .domain import RESULT_FORMAT, canonical, digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials(
  kind TEXT NOT NULL, version INTEGER NOT NULL,
  payload TEXT NOT NULL, event_seq INTEGER NOT NULL,
  PRIMARY KEY(kind, version));
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT, kind TEXT NOT NULL, payload TEXT NOT NULL, parents TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
  input_digest TEXT NOT NULL, param_version TEXT NOT NULL,
  parents TEXT NOT NULL, payload TEXT NOT NULL, event_seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS staging(skey TEXT PRIMARY KEY, committed INTEGER NOT NULL);
"""

ARCHIVE_FORMAT = "itc-archive/1"


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        # Half-write recovery: any staged write that never reached its commit
        # marker is discarded on open and reported, never silently applied.
        self.recovered = [
            r["skey"]
            for r in self.conn.execute("SELECT skey FROM staging WHERE committed=0")
        ]
        self.conn.execute("DELETE FROM staging WHERE committed=0")
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---------------- events ----------------
    def append_event(self, kind, payload, parents=()):
        cur = self.conn.execute(
            "INSERT INTO events(kind,payload,parents) VALUES(?,?,?)",
            (kind, canonical(payload), canonical(list(parents))),
        )
        seq = cur.lastrowid
        eid = f"evt:{seq:04d}"
        self.conn.execute("UPDATE events SET id=? WHERE seq=?", (eid, seq))
        self.conn.commit()
        return {"seq": seq, "id": eid, "kind": kind, "payload": payload, "parents": list(parents)}

    def events(self, upto_seq=None):
        q, args = "SELECT * FROM events", []
        if upto_seq is not None:
            q += " WHERE seq<=?"
            args.append(upto_seq)
        q += " ORDER BY seq"
        return [
            dict(seq=r["seq"], id=r["id"], kind=r["kind"],
                 payload=json.loads(r["payload"]), parents=json.loads(r["parents"]))
            for r in self.conn.execute(q, args)
        ]

    # ---------------- materials (immutable, separately versioned) ----------------
    def next_version(self, kind):
        row = self.conn.execute(
            "SELECT MAX(version) v FROM materials WHERE kind=?", (kind,)
        ).fetchone()
        return (row["v"] or 0) + 1

    def add_material(self, kind, payload, event_seq):
        version = self.next_version(kind)
        self.conn.execute(
            "INSERT INTO materials VALUES(?,?,?,?)",
            (kind, version, canonical(payload), event_seq),
        )
        self.conn.commit()
        return f"mat:{kind}:v{version}"

    def material(self, kind, version=None, upto_seq=None):
        q, args = "SELECT * FROM materials WHERE kind=?", [kind]
        if version is not None:
            q += " AND version=?"
            args.append(version)
        if upto_seq is not None:
            q += " AND event_seq<=?"
            args.append(upto_seq)
        q += " ORDER BY version DESC LIMIT 1"
        r = self.conn.execute(q, args).fetchone()
        return self._mat_row(r) if r else None

    def materials(self, upto_seq=None):
        """Latest version per kind as of the given event sequence."""
        out = {}
        q, args = "SELECT * FROM materials", []
        if upto_seq is not None:
            q += " WHERE event_seq<=?"
            args.append(upto_seq)
        q += " ORDER BY kind, version"
        for r in self.conn.execute(q, args):
            out[r["kind"]] = self._mat_row(r)
        return out

    def materials_all(self):
        return [
            self._mat_row(r)
            for r in self.conn.execute("SELECT * FROM materials ORDER BY kind, version")
        ]

    @staticmethod
    def _mat_row(r):
        return {
            "id": f"mat:{r['kind']}:v{r['version']}",
            "kind": r["kind"],
            "version": r["version"],
            "payload": json.loads(r["payload"]),
            "event_seq": r["event_seq"],
        }

    # ---------------- results (derived, content-addressed) ----------------
    def put_result(self, kind, status, input_digest, param_version, parents, payload,
                   event_seq, rid=None):
        rid = rid or f"res:{kind}:{input_digest[:12]}"
        self.conn.execute(
            "INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?)",
            (rid, kind, status, input_digest, param_version,
             canonical(list(parents)), canonical(payload), event_seq),
        )
        self.conn.commit()
        return rid

    def result(self, rid):
        r = self.conn.execute("SELECT * FROM results WHERE id=?", (rid,)).fetchone()
        return self._res_row(r) if r else None

    def results(self, kind=None, upto_seq=None):
        q, args, cond = "SELECT * FROM results", [], []
        if kind:
            cond.append("kind=?")
            args.append(kind)
        if upto_seq is not None:
            cond.append("event_seq<=?")
            args.append(upto_seq)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY event_seq, id"
        return [self._res_row(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _res_row(r):
        return dict(id=r["id"], kind=r["kind"], status=r["status"],
                    input_digest=r["input_digest"], param_version=r["param_version"],
                    parents=json.loads(r["parents"]), payload=json.loads(r["payload"]),
                    event_seq=r["event_seq"])

    # ---------------- staged writes (half-write recovery) ----------------
    def staged_write(self, key, fn):
        self.conn.execute("INSERT OR REPLACE INTO staging VALUES(?,0)", (key,))
        self.conn.commit()
        fn()
        self.conn.execute("UPDATE staging SET committed=1 WHERE skey=?", (key,))
        self.conn.execute("DELETE FROM staging WHERE skey=?", (key,))
        self.conn.commit()

    # ---------------- export / import ----------------
    def export(self):
        data = {
            "format": ARCHIVE_FORMAT,
            "events": self.events(),
            "materials": self.materials_all(),
            "results": self.results(),
        }
        data["digest"] = digest({k: data[k] for k in ("events", "materials", "results")})
        return data

    @classmethod
    def import_archive(cls, path, archive):
        """Returns (store, status). Format damage is its own distinct outcome."""
        if not isinstance(archive, dict) or archive.get("format") != ARCHIVE_FORMAT:
            return None, {"kind": RESULT_FORMAT, "message": "unknown archive format"}
        body = {k: archive.get(k) for k in ("events", "materials", "results")}
        if not all(isinstance(v, list) for v in body.values()) or digest(body) != archive.get("digest"):
            return None, {"kind": RESULT_FORMAT, "message": "archive digest mismatch"}
        store = cls(path)
        for e in body["events"]:
            store.conn.execute(
                "INSERT INTO events(seq,id,kind,payload,parents) VALUES(?,?,?,?,?)",
                (e["seq"], e["id"], e["kind"], canonical(e["payload"]), canonical(e["parents"])),
            )
        for m in body["materials"]:
            store.conn.execute(
                "INSERT INTO materials VALUES(?,?,?,?)",
                (m["kind"], m["version"], canonical(m["payload"]), m["event_seq"]),
            )
        for r in body["results"]:
            store.conn.execute(
                "INSERT INTO results VALUES(?,?,?,?,?,?,?,?)",
                (r["id"], r["kind"], r["status"], r["input_digest"], r["param_version"],
                 canonical(r["parents"]), canonical(r["payload"]), r["event_seq"]),
            )
        store.conn.commit()
        return store, {"kind": "ok"}
