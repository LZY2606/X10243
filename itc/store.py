"""SQLite storage: immutable versioned materials, machine-readable
events, and derived results with provenance.

Layering rule: materials are immutable (a revision appends a new
version); decisions are events; derived rows record input digest,
parameter version and parent dependencies, and are superseded — never
rewritten — so any historical event can be re-entered with the
dependency versions and visible conclusions of its time.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS materials(
  material_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  kind TEXT NOT NULL,
  content_json TEXT NOT NULL,
  hash TEXT NOT NULL,
  units_json TEXT NOT NULL,
  created_seq INTEGER NOT NULL,
  PRIMARY KEY(material_id, version));
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  branch TEXT NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS derived(
  derived_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  kind TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  parents_json TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_seq INTEGER NOT NULL,
  superseded_seq INTEGER,
  PRIMARY KEY(derived_id, version));
"""

SCHEMA_VERSION = "1"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_hash(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,))
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('event_counter', '0')")
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- events -----------------------------------------------------
    def next_event_id(self) -> str:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='event_counter'").fetchone()
        n = int(row[0]) + 1
        return f"ev-{n:06d}"

    def record_event(self, event_type: str, payload: dict,
                     branch: str = "main", conn=None) -> tuple:
        c = conn or self.conn
        row = c.execute(
            "SELECT value FROM meta WHERE key='event_counter'").fetchone()
        n = int(row[0]) + 1
        c.execute("UPDATE meta SET value=? WHERE key='event_counter'",
                  (str(n),))
        event_id = f"ev-{n:06d}"
        cur = c.execute(
            "INSERT INTO events(event_id, branch, type, payload_json)"
            " VALUES(?,?,?,?)",
            (event_id, branch, event_type, canonical_json(payload)))
        if conn is None:
            self.conn.commit()
        return cur.lastrowid, event_id

    def events(self, branch: Optional[str] = None,
               upto_seq: Optional[int] = None) -> list:
        sql = "SELECT seq, event_id, branch, type, payload_json FROM events"
        cond, args = [], []
        if branch is not None:
            cond.append("branch = ?")
            args.append(branch)
        if upto_seq is not None:
            cond.append("seq <= ?")
            args.append(upto_seq)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY seq"
        return [{"seq": r[0], "event_id": r[1], "branch": r[2], "type": r[3],
                 "payload": json.loads(r[4])}
                for r in self.conn.execute(sql, args)]

    def branches(self) -> list:
        rows = self.conn.execute(
            "SELECT DISTINCT branch FROM events ORDER BY branch").fetchall()
        return [r[0] for r in rows] or ["main"]

    # ---- materials --------------------------------------------------
    def add_material(self, material_id: str, kind: str, content: dict,
                     units: dict, created_seq: int, conn=None) -> int:
        c = conn or self.conn
        row = c.execute(
            "SELECT MAX(version) FROM materials WHERE material_id=?",
            (material_id,)).fetchone()
        version = (row[0] or 0) + 1
        c.execute(
            "INSERT INTO materials(material_id, version, kind, content_json,"
            " hash, units_json, created_seq) VALUES(?,?,?,?,?,?,?)",
            (material_id, version, kind, canonical_json(content),
             content_hash(content), canonical_json(units), created_seq))
        if conn is None:
            self.conn.commit()
        return version

    def material(self, material_id: str,
                 version: Optional[int] = None) -> Optional[dict]:
        if version is None:
            row = self.conn.execute(
                "SELECT material_id, version, kind, content_json, hash,"
                " units_json, created_seq FROM materials WHERE material_id=?"
                " ORDER BY version DESC LIMIT 1", (material_id,)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT material_id, version, kind, content_json, hash,"
                " units_json, created_seq FROM materials"
                " WHERE material_id=? AND version=?",
                (material_id, version)).fetchone()
        if row is None:
            return None
        return {"material_id": row[0], "version": row[1], "kind": row[2],
                "content": json.loads(row[3]), "hash": row[4],
                "units": json.loads(row[5]), "created_seq": row[6]}

    def material_versions(self, material_id: str) -> list:
        return [r[0] for r in self.conn.execute(
            "SELECT version FROM materials WHERE material_id=?"
            " ORDER BY version", (material_id,))]

    def all_materials(self) -> list:
        return [{"material_id": r[0], "version": r[1], "kind": r[2],
                 "content": json.loads(r[3]), "hash": r[4],
                 "units": json.loads(r[5]), "created_seq": r[6]}
                for r in self.conn.execute(
                    "SELECT material_id, version, kind, content_json, hash,"
                    " units_json, created_seq FROM materials"
                    " ORDER BY material_id, version")]

    # ---- derived ----------------------------------------------------
    def put_derived(self, derived_id: str, kind: str, payload: dict,
                    parents: Sequence[dict], status: str,
                    event_seq: int, conn=None) -> int:
        """Insert a new version of a derived result; the previously
        visible version is marked superseded at this event seq."""
        c = conn or self.conn
        row = c.execute(
            "SELECT MAX(version) FROM derived WHERE derived_id=?",
            (derived_id,)).fetchone()
        version = (row[0] or 0) + 1
        if version > 1:
            c.execute(
                "UPDATE derived SET superseded_seq=? WHERE derived_id=?"
                " AND version=?",
                (event_seq, derived_id, version - 1))
        digest = content_hash({"parents": list(parents), "kind": kind})
        c.execute(
            "INSERT INTO derived(derived_id, version, kind, input_digest,"
            " parents_json, payload_json, status, created_seq)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (derived_id, version, kind, digest, canonical_json(list(parents)),
             canonical_json(payload), status, event_seq))
        if conn is None:
            self.conn.commit()
        return version

    def visible_derived(self, kind: Optional[str] = None,
                        as_of: Optional[int] = None) -> list:
        """Latest visible version per derived id at event seq `as_of`
        (None = now). Equivalent inputs always yield a stable order:
        sorted by derived_id."""
        cond, args = [], []
        if kind is not None:
            cond.append("kind = ?")
            args.append(kind)
        if as_of is not None:
            cond.append("created_seq <= ?")
            args.append(as_of)
        where = (" WHERE " + " AND ".join(cond)) if cond else ""
        sql = (
            "SELECT d.derived_id, d.version, d.kind, d.input_digest,"
            " d.parents_json, d.payload_json, d.status, d.created_seq,"
            " d.superseded_seq FROM derived d"
            " JOIN (SELECT derived_id, MAX(version) AS mv FROM derived"
            + where.replace("kind", "kind").replace("created_seq", "created_seq")
            + " GROUP BY derived_id) latest"
            " ON d.derived_id = latest.derived_id AND d.version = latest.mv")
        if cond:
            sql = (
                "SELECT d.derived_id, d.version, d.kind, d.input_digest,"
                " d.parents_json, d.payload_json, d.status, d.created_seq,"
                " d.superseded_seq FROM derived d"
                " JOIN (SELECT derived_id, MAX(version) AS mv FROM derived"
                f" WHERE {' AND '.join(cond)} GROUP BY derived_id) latest"
                " ON d.derived_id = latest.derived_id AND d.version = latest.mv")
        rows = self.conn.execute(sql + " ORDER BY d.derived_id", args).fetchall()
        out = []
        for r in rows:
            superseded = r[8]
            if as_of is not None and superseded is not None and superseded <= as_of:
                continue
            out.append({"derived_id": r[0], "version": r[1], "kind": r[2],
                        "input_digest": r[3], "parents": json.loads(r[4]),
                        "payload": json.loads(r[5]), "status": r[6],
                        "created_seq": r[7], "superseded_seq": superseded})
        return out

    def all_derived(self) -> list:
        return [{"derived_id": r[0], "version": r[1], "kind": r[2],
                 "input_digest": r[3], "parents": json.loads(r[4]),
                 "payload": json.loads(r[5]), "status": r[6],
                 "created_seq": r[7], "superseded_seq": r[8]}
                for r in self.conn.execute(
                    "SELECT derived_id, version, kind, input_digest,"
                    " parents_json, payload_json, status, created_seq,"
                    " superseded_seq FROM derived"
                    " ORDER BY derived_id, version")]

    def stale_derived(self, as_of: Optional[int] = None) -> list:
        """Visible derived rows whose parent versions are no longer the
        current ones (one level; sufficient for material->derived and
        derived->derived edges used here)."""
        visible = self.visible_derived(as_of=as_of)
        latest_derived = {d["derived_id"]: d["version"]
                          for d in self.visible_derived()}
        stale = []
        for d in visible:
            for p in d["parents"]:
                if p["type"] == "material":
                    cur = self.material(p["id"])
                    if cur is None or cur["version"] != p["version"]:
                        stale.append(d)
                        break
                elif p["type"] == "derived":
                    if latest_derived.get(p["id"]) != p["version"]:
                        stale.append(d)
                        break
        return stale

    # ---- export / import ---------------------------------------------
    def dump(self) -> dict:
        return {
            "meta": {r[0]: r[1] for r in self.conn.execute(
                "SELECT key, value FROM meta ORDER BY key")},
            "materials": self.all_materials(),
            "events": self.events(),
            "derived": self.all_derived(),
        }

    def is_empty(self) -> bool:
        n = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return n == 0

    def restore(self, dump: dict) -> None:
        """Load a dump in a single transaction. Any failure rolls back
        completely, so a half-written import can never be observed."""
        c = self.conn.cursor()
        try:
            c.execute("BEGIN")
            meta = dump["meta"]
            for k, v in meta.items():
                c.execute("INSERT INTO meta(key, value) VALUES(?,?)"
                          " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (k, str(v)))
            for m in dump["materials"]:
                c.execute(
                    "INSERT INTO materials(material_id, version, kind,"
                    " content_json, hash, units_json, created_seq)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (m["material_id"], m["version"], m["kind"],
                     canonical_json(m["content"]), m["hash"],
                     canonical_json(m["units"]), m["created_seq"]))
            for e in dump["events"]:
                c.execute(
                    "INSERT INTO events(seq, event_id, branch, type,"
                    " payload_json) VALUES(?,?,?,?,?)",
                    (e["seq"], e["event_id"], e["branch"], e["type"],
                     canonical_json(e["payload"])))
            for d in dump["derived"]:
                c.execute(
                    "INSERT INTO derived(derived_id, version, kind,"
                    " input_digest, parents_json, payload_json, status,"
                    " created_seq, superseded_seq)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (d["derived_id"], d["version"], d["kind"],
                     d["input_digest"], canonical_json(d["parents"]),
                     canonical_json(d["payload"]), d["status"],
                     d["created_seq"], d["superseded_seq"]))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
