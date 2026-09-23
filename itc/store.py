"""SQLite store with two layers: immutable versioned materials and mutable
decisions, both driven by an append-only event log. Every derived result
records its input digest, parameter version and parent dependencies.

Writes go through an application-level journal: rows are tagged with a txn id
and only become visible once the journal entry is marked committed. On open,
any txn left open (half-write / crash) is marked aborted and its rows stay
invisible, so recovery is deterministic and testable."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal(
  txn_id TEXT PRIMARY KEY, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS event(
  id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, txn_id TEXT NOT NULL,
  type TEXT NOT NULL, payload TEXT NOT NULL, parents TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS material(
  id TEXT NOT NULL, version INTEGER NOT NULL, kind TEXT NOT NULL,
  payload TEXT NOT NULL, event_id TEXT NOT NULL, txn_id TEXT NOT NULL,
  PRIMARY KEY(id, version));
CREATE TABLE IF NOT EXISTS result(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, event_id TEXT NOT NULL,
  input_digest TEXT NOT NULL, params TEXT NOT NULL, deps TEXT NOT NULL,
  status TEXT NOT NULL, diagnostics TEXT NOT NULL, payload TEXT NOT NULL,
  txn_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decision(
  id TEXT PRIMARY KEY, event_id TEXT NOT NULL, branch TEXT NOT NULL,
  target TEXT NOT NULL, action TEXT NOT NULL, rationale TEXT NOT NULL,
  txn_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

COMMITTED = "txn_id IN (SELECT txn_id FROM journal WHERE status='committed')"


class CorruptFormat(Exception):
    status = "format_corrupt"


def _now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.recover()

    def close(self):
        self.conn.close()

    def recover(self):
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("UPDATE journal SET status='aborted' WHERE status='open'")
        self.conn.execute("COMMIT")
        return self.conn.execute(
            "SELECT COUNT(*) c FROM journal WHERE status='aborted'").fetchone()["c"]

    @contextmanager
    def txn(self):
        txn_id = f"txn-{uuid.uuid4().hex[:12]}"
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("INSERT INTO journal(txn_id, status) VALUES(?, 'open')", (txn_id,))
        try:
            yield cur, txn_id
            cur.execute("UPDATE journal SET status='committed' WHERE txn_id=?", (txn_id,))
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise

    def _next_seq(self, cur):
        row = cur.execute("SELECT COALESCE(MAX(seq),0)+1 s FROM event").fetchone()
        return row["s"]

    def add_event(self, cur, txn_id, type_, payload, parents):
        seq = self._next_seq(cur)
        event_id = f"evt-{seq:04d}"
        cur.execute(
            "INSERT INTO event VALUES(?,?,?,?,?,?,?)",
            (event_id, seq, txn_id, type_, json.dumps(payload, sort_keys=True),
             json.dumps(parents), _now()))
        return {"id": event_id, "seq": seq, "type": type_,
                "payload": payload, "parents": parents}

    def put_material(self, cur, txn_id, kind, payload, event_id, material_id=None):
        material_id = material_id or f"mat-{kind}"
        row = cur.execute("SELECT COALESCE(MAX(version),0)+1 v FROM material WHERE id=?",
                          (material_id,)).fetchone()
        version = row["v"]
        cur.execute("INSERT INTO material VALUES(?,?,?,?,?,?)",
                    (material_id, version, kind,
                     json.dumps(payload, sort_keys=True), event_id, txn_id))
        return {"id": material_id, "version": version, "kind": kind}

    def put_result(self, cur, txn_id, kind, event_id, input_digest, params, deps,
                   status, diagnostics, payload):
        result_id = f"res-{kind}-{input_digest[:12]}"
        cur.execute(
            "INSERT OR REPLACE INTO result VALUES(?,?,?,?,?,?,?,?,?,?)",
            (result_id, kind, event_id, input_digest,
             json.dumps(params, sort_keys=True), json.dumps(deps, sort_keys=True),
             status, json.dumps(diagnostics), json.dumps(payload), txn_id))
        return result_id

    def put_decision(self, cur, txn_id, event_id, branch, target, action, rationale):
        row = cur.execute("SELECT COUNT(*) c FROM decision").fetchone()
        decision_id = f"dec-{row['c']+1:04d}"
        cur.execute("INSERT INTO decision VALUES(?,?,?,?,?,?,?)",
                    (decision_id, event_id, branch, target, action, rationale, txn_id))
        return decision_id

    def is_empty(self):
        return self.conn.execute(
            f"SELECT COUNT(*) c FROM event WHERE {COMMITTED}").fetchone()["c"] == 0

    def events(self, as_of=None):
        sql = f"SELECT * FROM event WHERE {COMMITTED}"
        args = []
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        rows = self.conn.execute(sql + " ORDER BY seq", args).fetchall()
        return [dict(id=r["id"], seq=r["seq"], type=r["type"],
                     payload=json.loads(r["payload"]),
                     parents=json.loads(r["parents"])) for r in rows]

    def materials(self, as_of=None):
        seq = as_of if as_of is not None else self.latest_seq()
        rows = self.conn.execute(
            f"""SELECT m.* FROM material m JOIN event e ON m.event_id=e.id
                WHERE m.{COMMITTED} AND e.seq<=? ORDER BY m.id, m.version""",
            (seq,)).fetchall()
        return [dict(id=r["id"], version=r["version"], kind=r["kind"],
                     payload=json.loads(r["payload"]), event_id=r["event_id"])
                for r in rows]

    def current_materials(self, as_of=None):
        current = {}
        for m in self.materials(as_of):
            if m["id"] not in current or m["version"] > current[m["id"]]["version"]:
                current[m["id"]] = m
        return current

    def results(self, as_of=None):
        sql = f"""SELECT r.* FROM result r JOIN event e ON r.event_id=e.id
                  WHERE r.{COMMITTED}"""
        args = []
        if as_of is not None:
            sql += " AND e.seq<=?"
            args.append(as_of)
        rows = self.conn.execute(sql + " ORDER BY r.id", args).fetchall()
        return [dict(id=r["id"], kind=r["kind"], event_id=r["event_id"],
                     input_digest=r["input_digest"], params=json.loads(r["params"]),
                     deps=json.loads(r["deps"]), status=r["status"],
                     diagnostics=json.loads(r["diagnostics"]),
                     payload=json.loads(r["payload"])) for r in rows]

    def decisions(self, as_of=None, branch=None):
        sql = f"""SELECT d.* FROM decision d JOIN event e ON d.event_id=e.id
                  WHERE d.{COMMITTED}"""
        args = []
        if as_of is not None:
            sql += " AND e.seq<=?"
            args.append(as_of)
        if branch is not None:
            sql += " AND d.branch=?"
            args.append(branch)
        rows = self.conn.execute(sql + " ORDER BY d.id", args).fetchall()
        return [dict(id=r["id"], event_id=r["event_id"], branch=r["branch"],
                     target=r["target"], action=r["action"],
                     rationale=r["rationale"]) for r in rows]

    def latest_seq(self):
        return self.conn.execute(
            f"SELECT COALESCE(MAX(seq),0) s FROM event WHERE {COMMITTED}"
        ).fetchone()["s"]

    def export(self):
        return {
            "format": "itc-review-export",
            "format_version": 1,
            "events": self.events(),
            "materials": self.materials(),
            "results": self.results(),
            "decisions": self.decisions(),
        }

    def import_(self, data):
        if not isinstance(data, dict) or data.get("format") != "itc-review-export":
            raise CorruptFormat("missing or unknown export format marker")
        for key in ("events", "materials", "results", "decisions"):
            if key not in data or not isinstance(data[key], list):
                raise CorruptFormat(f"export is missing section {key!r}")
        with self.txn() as (cur, txn_id):
            for e in data["events"]:
                cur.execute(
                    "INSERT OR IGNORE INTO event VALUES(?,?,?,?,?,?,?)",
                    (e["id"], e["seq"], txn_id, e["type"],
                     json.dumps(e["payload"], sort_keys=True),
                     json.dumps(e.get("parents", [])), _now()))
            for m in data["materials"]:
                cur.execute(
                    "INSERT OR IGNORE INTO material VALUES(?,?,?,?,?,?)",
                    (m["id"], m["version"], m["kind"],
                     json.dumps(m["payload"], sort_keys=True), m["event_id"], txn_id))
            for r in data["results"]:
                cur.execute(
                    "INSERT OR IGNORE INTO result VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (r["id"], r["kind"], r["event_id"], r["input_digest"],
                     json.dumps(r["params"], sort_keys=True),
                     json.dumps(r["deps"], sort_keys=True), r["status"],
                     json.dumps(r["diagnostics"]), json.dumps(r["payload"]), txn_id))
            for d in data["decisions"]:
                cur.execute(
                    "INSERT OR IGNORE INTO decision VALUES(?,?,?,?,?,?,?)",
                    (d["id"], d["event_id"], d["branch"], d["target"],
                     d["action"], d["rationale"], txn_id))
        return {"events": len(data["events"]), "materials": len(data["materials"]),
                "results": len(data["results"]), "decisions": len(data["decisions"])}

    def dump(self):
        return json.dumps(self.export(), sort_keys=True)
