"""Layered SQLite store.

Layers:
  materials     - immutable source payloads, versioned *separately* per category
                  (power_curve / injection_schedule / concentration / temperature /
                   cell_volume / baseline / blank_curve)
  events        - append-only log; the mutable decision layer is event-sourced
  computations  - derived results with input digest, param version, parent deps
  diagnostics   - positioned diagnostics, never silently dropped
  write_journal - write intents so a half-written state is recovered on open
"""
from __future__ import annotations

import json
import sqlite3
import time

from .domain import FORMAT_CORRUPT, SEMANTIC_UNSOLVABLE, Failure, digest, make_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS materials(
  material_id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  version INTEGER NOT NULL,
  payload TEXT NOT NULL,
  supersedes TEXT,
  created_seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  rationale TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS computations(
  comp_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  param_version TEXT NOT NULL,
  deps TEXT NOT NULL,
  parent_ids TEXT NOT NULL,
  status TEXT NOT NULL,
  result TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  created_seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS diagnostics(
  diag_id TEXT PRIMARY KEY,
  severity TEXT NOT NULL,
  code TEXT NOT NULL,
  message TEXT NOT NULL,
  location TEXT NOT NULL,
  entity_ids TEXT NOT NULL,
  created_seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS write_journal(
  intent_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  detail TEXT NOT NULL,
  created_at TEXT NOT NULL);
"""

MATERIAL_CATEGORIES = (
    "power_curve", "injection_schedule", "concentration", "temperature",
    "cell_volume", "baseline", "blank_curve",
)

EXPORT_FORMAT = "itc-review-export"
EXPORT_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.recovered_intents = self._recover()

    def close(self):
        self.conn.close()

    # ---------------- meta ----------------
    def meta_get(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def meta_set(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
        self.conn.commit()

    # ---------------- sequence ----------------
    def current_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0) AS s FROM events").fetchone()
        return row["s"]

    def _next_seq(self) -> int:
        return self.current_seq() + 1

    # ---------------- materials (immutable, per-category versions) ----------------
    def add_material(self, category: str, payload: dict, supersedes=None, with_event=True):
        assert category in MATERIAL_CATEGORIES, f"unknown category {category}"
        mid = make_id(f"mat-{category}", payload)
        existing = self.get_material(mid)
        if existing:
            return existing
        seq = self._next_seq()
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version),0) AS v FROM materials WHERE category=?",
            (category,)).fetchone()
        version = row["v"] + 1
        self.conn.execute(
            "INSERT INTO materials VALUES(?,?,?,?,?,?)",
            (mid, category, version, json.dumps(payload), supersedes, seq))
        self.conn.commit()
        if with_event:
            self.record_event(
                "import_material" if supersedes is None else "revise_material",
                {"material_id": mid, "category": category, "version": version,
                 "supersedes": supersedes})
        return self.get_material(mid)

    def get_material(self, material_id):
        row = self.conn.execute(
            "SELECT * FROM materials WHERE material_id=?", (material_id,)).fetchone()
        return self._mat(row) if row else None

    def _mat(self, row):
        return {"material_id": row["material_id"], "category": row["category"],
                "version": row["version"], "payload": json.loads(row["payload"]),
                "supersedes": row["supersedes"], "created_seq": row["created_seq"]}

    def materials(self, category=None):
        q = "SELECT * FROM materials"
        args = ()
        if category:
            q += " WHERE category=?"
            args = (category,)
        q += " ORDER BY category, version"
        return [self._mat(r) for r in self.conn.execute(q, args).fetchall()]

    def current_materials(self, at_seq=None):
        """Latest version per category visible at seq (None = latest overall)."""
        out = {}
        for m in self.materials():
            if at_seq is not None and m["created_seq"] > at_seq:
                continue
            cur = out.get(m["category"])
            if cur is None or m["version"] > cur["version"]:
                out[m["category"]] = m
        return out

    # ---------------- events (append-only decision layer) ----------------
    def record_event(self, kind, payload, rationale=""):
        seq = self._next_seq()
        eid = f"evt-{seq:06d}"
        self.conn.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?)",
            (eid, seq, kind, json.dumps(payload), rationale, _now()))
        self.conn.commit()
        return self.get_event(eid)

    def get_event(self, event_id):
        row = self.conn.execute(
            "SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return self._evt(row) if row else None

    def _evt(self, row):
        return {"event_id": row["event_id"], "seq": row["seq"], "kind": row["kind"],
                "payload": json.loads(row["payload"]), "rationale": row["rationale"],
                "created_at": row["created_at"]}

    def events(self, at_seq=None):
        q = "SELECT * FROM events"
        args = ()
        if at_seq is not None:
            q += " WHERE seq<=?"
            args = (at_seq,)
        q += " ORDER BY seq"
        return [self._evt(r) for r in self.conn.execute(q, args).fetchall()]

    # ---------------- computations (derived layer) ----------------
    def add_computation(self, kind, deps, param_version, parent_ids, status,
                        result, diagnostics):
        input_digest = digest({"deps": deps, "param_version": param_version})
        cid = make_id(f"cmp-{kind}", {"kind": kind, "input_digest": input_digest})
        existing = self.get_computation(cid)
        if existing:
            return existing
        seq = self._next_seq()
        self.conn.execute(
            "INSERT INTO computations VALUES(?,?,?,?,?,?,?,?,?,?)",
            (cid, kind, input_digest, param_version, json.dumps(deps),
             json.dumps(parent_ids), status, json.dumps(result),
             json.dumps(diagnostics), seq))
        self.conn.commit()
        for d in diagnostics:
            self.add_diagnostic(d.get("severity", "info"), d.get("code", "INFO"),
                                d.get("message", ""), d.get("location", {}),
                                d.get("entity_ids", []))
        return self.get_computation(cid)

    def get_computation(self, comp_id):
        row = self.conn.execute(
            "SELECT * FROM computations WHERE comp_id=?", (comp_id,)).fetchone()
        return self._cmp(row) if row else None

    def _cmp(self, row):
        return {"comp_id": row["comp_id"], "kind": row["kind"],
                "input_digest": row["input_digest"],
                "param_version": row["param_version"],
                "deps": json.loads(row["deps"]),
                "parent_ids": json.loads(row["parent_ids"]),
                "status": row["status"], "result": json.loads(row["result"]),
                "diagnostics": json.loads(row["diagnostics"]),
                "created_seq": row["created_seq"]}

    def computations(self, kind=None, at_seq=None):
        q, args = "SELECT * FROM computations WHERE 1=1", []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if at_seq is not None:
            q += " AND created_seq<=?"
            args.append(at_seq)
        q += " ORDER BY created_seq, comp_id"
        return [self._cmp(r) for r in self.conn.execute(q, args).fetchall()]

    def latest_computation(self, kind, at_seq=None):
        comps = self.computations(kind=kind, at_seq=at_seq)
        return comps[-1] if comps else None

    # ---------------- diagnostics ----------------
    def add_diagnostic(self, severity, code, message, location=None, entity_ids=None):
        did = make_id("diag", {"c": code, "m": message, "l": location or {},
                               "e": entity_ids or []})
        row = self.conn.execute(
            "SELECT diag_id FROM diagnostics WHERE diag_id=?", (did,)).fetchone()
        if row:
            return did
        self.conn.execute(
            "INSERT INTO diagnostics VALUES(?,?,?,?,?,?,?)",
            (did, severity, code, message, json.dumps(location or {}),
             json.dumps(entity_ids or []), self.current_seq()))
        self.conn.commit()
        return did

    def diagnostics(self, at_seq=None):
        q, args = "SELECT * FROM diagnostics", []
        if at_seq is not None:
            q += " WHERE created_seq<=?"
            args.append(at_seq)
        q += " ORDER BY diag_id"
        out = []
        for r in self.conn.execute(q, args).fetchall():
            out.append({"diag_id": r["diag_id"], "severity": r["severity"],
                        "code": r["code"], "message": r["message"],
                        "location": json.loads(r["location"]),
                        "entity_ids": json.loads(r["entity_ids"]),
                        "created_seq": r["created_seq"]})
        return out

    # ---------------- write journal / half-write recovery ----------------
    def begin_write(self, intent_id, detail=""):
        self.conn.execute(
            "INSERT OR IGNORE INTO write_journal VALUES(?,?,?,?)",
            (intent_id, "pending", detail, _now()))
        self.conn.commit()

    def complete_write(self, intent_id):
        self.conn.execute(
            "UPDATE write_journal SET status='committed' WHERE intent_id=?",
            (intent_id,))
        self.conn.commit()

    def atomic(self, intent_id, fn, detail=""):
        """Run fn(conn) as one journaled, transactional write."""
        self.begin_write(intent_id, detail)
        try:
            with self.conn:
                fn(self.conn)
            self.complete_write(intent_id)
        except Exception:
            self.conn.execute(
                "UPDATE write_journal SET status='aborted' WHERE intent_id=?",
                (intent_id,))
            self.conn.commit()
            raise

    def _recover(self):
        """On open: any pending intent means a crash mid-write; mark aborted and
        surface a diagnostic instead of pretending success."""
        rows = self.conn.execute(
            "SELECT * FROM write_journal WHERE status='pending'").fetchall()
        recovered = []
        for r in rows:
            self.conn.execute(
                "UPDATE write_journal SET status='aborted' WHERE intent_id=?",
                (r["intent_id"],))
            recovered.append(r["intent_id"])
        self.conn.commit()
        for intent_id in recovered:
            self.add_diagnostic(
                "warning", "WRITE_RECOVERED",
                f"write intent {intent_id} was pending at open; marked aborted, "
                "no partial state applied", {"intent_id": intent_id}, [])
        return recovered

    # ---------------- export / import ----------------
    def export(self):
        return {
            "format": EXPORT_FORMAT,
            "format_version": EXPORT_VERSION,
            "meta": {r["key"]: r["value"] for r in
                     self.conn.execute("SELECT * FROM meta").fetchall()},
            "materials": [
                {"material_id": r["material_id"], "category": r["category"],
                 "version": r["version"], "payload": json.loads(r["payload"]),
                 "supersedes": r["supersedes"], "created_seq": r["created_seq"]}
                for r in self.conn.execute(
                    "SELECT * FROM materials ORDER BY category, version").fetchall()],
            "events": self.events(),
            "computations": self.computations(),
            "diagnostics": self.diagnostics(),
            "write_journal": [
                {"intent_id": r["intent_id"], "status": r["status"],
                 "detail": r["detail"], "created_at": r["created_at"]}
                for r in self.conn.execute("SELECT * FROM write_journal").fetchall()],
        }

    def import_export(self, data):
        """Import an export document. Corrupt format and semantic conflicts are
        two distinct failure kinds."""
        if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
            return Failure(FORMAT_CORRUPT, "not an itc-review-export document")
        if data.get("format_version") != EXPORT_VERSION:
            return Failure(FORMAT_CORRUPT,
                           f"unsupported format_version {data.get('format_version')}")
        for key in ("materials", "events", "computations", "diagnostics"):
            if not isinstance(data.get(key), list):
                return Failure(FORMAT_CORRUPT, f"missing or invalid section {key!r}")
        # semantic check: event seq conflicts with different identity
        incoming = {e["seq"]: e["event_id"] for e in data["events"]
                    if isinstance(e, dict) and "seq" in e and "event_id" in e}
        if len(incoming) != len(data["events"]):
            return Failure(FORMAT_CORRUPT, "events lack seq/event_id")
        for e in self.events():
            if e["seq"] in incoming and incoming[e["seq"]] != e["event_id"]:
                return Failure(
                    SEMANTIC_UNSOLVABLE,
                    f"event seq {e['seq']} already bound to {e['event_id']}, "
                    f"cannot also bind {incoming[e['seq']]}")
        def _write(conn):
            for k, v in data.get("meta", {}).items():
                conn.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, v))
            for m in data["materials"]:
                conn.execute(
                    "INSERT OR IGNORE INTO materials VALUES(?,?,?,?,?,?)",
                    (m["material_id"], m["category"], m["version"],
                     json.dumps(m["payload"]), m.get("supersedes"), m["created_seq"]))
            for e in data["events"]:
                conn.execute(
                    "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?)",
                    (e["event_id"], e["seq"], e["kind"], json.dumps(e["payload"]),
                     e.get("rationale", ""), e.get("created_at", _now())))
            for c in data["computations"]:
                conn.execute(
                    "INSERT OR IGNORE INTO computations VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (c["comp_id"], c["kind"], c["input_digest"], c["param_version"],
                     json.dumps(c["deps"]), json.dumps(c["parent_ids"]),
                     c["status"], json.dumps(c["result"]),
                     json.dumps(c["diagnostics"]), c["created_seq"]))
            for d in data["diagnostics"]:
                conn.execute(
                    "INSERT OR IGNORE INTO diagnostics VALUES(?,?,?,?,?,?,?)",
                    (d["diag_id"], d["severity"], d["code"], d["message"],
                     json.dumps(d["location"]), json.dumps(d["entity_ids"]),
                     d["created_seq"]))
            for j in data.get("write_journal", []):
                conn.execute(
                    "INSERT OR IGNORE INTO write_journal VALUES(?,?,?,?)",
                    (j["intent_id"], j["status"], j["detail"], j["created_at"]))
        self.atomic(make_id("intent", {"import": digest(data)[:16]}), _write,
                    detail="import")
        return {"ok": True, "events": len(data["events"]),
                "materials": len(data["materials"]),
                "computations": len(data["computations"])}
