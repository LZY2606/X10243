"""SQLite event store: the only durable state.

All business rules live behind this layer and are directly testable
(the CLI/API is a thin wrapper).  Writes happen inside a single SQLite
transaction: an interrupted append leaves the previous complete state
intact ("half-write recovery").
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import time

from . import replay
from .hashing import canonical, digest
from .materials import Material
from .replay import (
    ID_FIELDS,
    LAYERS_BY_EVENT,
    RESULT_KINDS,
    compute_layer,
    make_event,
    project,
)
from .rules import rule_id

SCHEMA_VERSION = "1"


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path="itc_data.sqlite3"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------
    def _init_schema(self):
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS branches (
                    name TEXT PRIMARY KEY,
                    parent_branch TEXT,
                    fork_seq INTEGER NOT NULL,
                    created_seq INTEGER NOT NULL,
                    label TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    branch TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    parent_hash TEXT,
                    timestamp TEXT NOT NULL,
                    caused_by TEXT);
                CREATE TABLE IF NOT EXISTS results_cache (
                    result_id TEXT PRIMARY KEY,
                    result_kind TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_seq INTEGER NOT NULL);
                """
            )
            row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                    (SCHEMA_VERSION,))
            branch_row = self.conn.execute(
                "SELECT name FROM branches WHERE name='main'").fetchone()
            if branch_row is None:
                self.conn.execute(
                    "INSERT INTO branches(name,parent_branch,fork_seq,created_seq,label)"
                    " VALUES('main',NULL,0,0,'primary review line')")

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def list_branches(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY created_seq, name")]

    def events(self, branch=None, up_to_seq=None):
        q = "SELECT * FROM events"
        clauses, params = [], []
        if branch:
            clauses.append("branch=?")
            params.append(branch)
        if up_to_seq is not None:
            clauses.append("seq<=?")
            params.append(up_to_seq)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY seq"
        out = []
        for r in self.conn.execute(q, params):
            ev = dict(r)
            ev["payload"] = json.loads(ev["payload"])
            out.append(ev)
        return out

    def head_seq(self, branch):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) FROM events WHERE branch=?",
            (branch,)).fetchone()
        return row[0]

    def view(self, branch="main", up_to_seq=None):
        branch_row = self._branch(branch)
        if branch_row is None:
            raise ValueError(f"unknown branch {branch!r}")
        parent = branch_row.get("parent_branch")
        if parent is None:
            events = self.events(branch, up_to_seq)
            return project(events, branch=branch, up_to_seq=up_to_seq,
                           fork_seq=0)
        # include the parent line up to the fork point, then this branch
        parent_events = [e for e in self.events(parent, branch_row["fork_seq"])]
        own = [e for e in self.events(branch, up_to_seq)]
        proj = project(parent_events, branch=parent,
                       up_to_seq=branch_row["fork_seq"], fork_seq=0)
        proj.branch = branch
        proj.fork_seq = branch_row["fork_seq"]
        for ev in own:
            if up_to_seq is not None and ev["seq"] > up_to_seq:
                break
            proj.apply(ev)
            proj.events.append(ev)
        return proj

    def _branch(self, name):
        r = self.conn.execute("SELECT * FROM branches WHERE name=?", (name,)).fetchone()
        return dict(r) if r else None

    def result_body(self, result_id):
        r = self.conn.execute(
            "SELECT body FROM results_cache WHERE result_id=?", (result_id,)).fetchone()
        return json.loads(r["body"]) if r else None

    # ------------------------------------------------------------------
    # write (single transaction; replayable)
    # ------------------------------------------------------------------
    def _next_seq(self):
        r = self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 AS n FROM events").fetchone()
        return int(r["n"])

    def _append_one(self, cur, branch, kind, payload, caused_by,
                    parent_hash=None, timestamp=None):
        seq = self._next_seq()
        if parent_hash is None:
            last = cur.execute(
                "SELECT event_id FROM events WHERE branch=? ORDER BY seq DESC LIMIT 1",
                (branch,)).fetchone()
            parent_hash = last["event_id"] if last else "GENESIS"
        ev = make_event(seq, kind, payload, parent_hash, branch,
                        timestamp or utc_now(), caused_by)
        cur.execute(
            "INSERT INTO events(seq,event_id,branch,kind,payload,parent_hash,"
            "timestamp,caused_by) VALUES(?,?,?,?,?,?,?,?)",
            (seq, ev["event_id"], branch, ev["kind"],
             json.dumps(ev["payload"], ensure_ascii=False, allow_nan=False),
             ev["parent"], ev["timestamp"], ev["caused_by"]))
        return ev

    def _apply_to_projection(self, proj, ev, cur):
        proj.apply(ev)
        proj.events.append(ev)

    def _cascade(self, proj, layers, branch, cur, caused_by, diagnostics):
        for layer in layers:
            result, diags = compute_layer(layer, proj)
            diagnostics.extend(diags or [])
            if result is None:
                continue
            id_field = ID_FIELDS[layer]
            result["diagnostics"] = _dedup_diags(result.get("diagnostics", []) + (diags or []))
            rid = result[id_field]
            cur.execute(
                "INSERT OR IGNORE INTO results_cache(result_id,result_kind,body,created_seq)"
                " VALUES(?,?,?,?)",
                (rid, layer, json.dumps(result, ensure_ascii=False, allow_nan=False),
                 proj.events[-1]["seq"]))
            existing = proj.results.get(layer)
            if existing is None or existing.get(id_field) != rid:
                rev_ev = self._append_one(
                    cur, branch, "RESULT_ISSUED",
                    {"result_kind": layer, "id_field": id_field, "result": result},
                    caused_by=caused_by)
                self._apply_to_projection(proj, rev_ev, cur)

    def add_material(self, material, branch="main", source=None,
                     caused_by="import"):
        with self.conn as conn:
            cur = conn.cursor()
            proj = self.view(branch)
            existing = proj.mat(material.kind)
            kind = "MATERIAL_ADDED" if existing is None else "MATERIAL_SUPERSEDED"
            payload = {"material": material.to_dict(), "source": source
                       or material.payload.get("origin_label")}
            if kind == "MATERIAL_SUPERSEDED":
                payload["supersedes"] = existing.version_id
            ev = self._append_one(cur, branch, kind, payload, caused_by)
            self._apply_to_projection(proj, ev, cur)
            diagnostics = []
            self._cascade(proj, replay.recompute_plan(kind, material.kind),
                          branch, cur, caused_by, diagnostics)
            return {"event": ev, "material_version": material.version_id,
                    "kind": kind, "diagnostics": _dedup_diags(diagnostics)}

    def set_rules(self, rule_kind, body, branch="main", reason=None,
                  caused_by="reviewer"):
        body = dict(body)
        rid = rule_id(rule_kind, body)
        with self.conn as conn:
            cur = conn.cursor()
            proj = self.view(branch)
            ev = self._append_one(cur, branch, "RULE_SET",
                                  {"rule_kind": rule_kind, "body": body,
                                   "rule_id": rid, "reason": reason},
                                  caused_by)
            self._apply_to_projection(proj, ev, cur)
            diagnostics = []
            self._cascade(proj, replay.recompute_plan("RULE_SET", rule_kind),
                          branch, cur, caused_by, diagnostics)
            return {"event": ev, "rule_id": rid,
                    "diagnostics": _dedup_diags(diagnostics)}

    def decision(self, target, decision, params=None, branch="main",
                 reason=None, caused_by="reviewer", candidate_ref=None):
        payload = {"target": target, "decision": decision,
                   "params": params or {}, "reason": reason,
                   "candidate_ref": candidate_ref}
        with self.conn as conn:
            cur = conn.cursor()
            ev = self._append_one(cur, branch, "REVIEW_DECISION", payload,
                                  caused_by)
            return {"event": ev}

    def create_branch(self, name, from_branch="main", at_seq=None,
                      label=None):
        with self.conn as conn:
            cur = conn.cursor()
            if self._branch(name) is not None:
                raise ValueError(f"branch {name!r} already exists")
            at_seq = self.head_seq(from_branch) if at_seq is None else at_seq
            created_seq = self._next_seq()
            cur.execute(
                "INSERT INTO branches(name,parent_branch,fork_seq,created_seq,label)"
                " VALUES(?,?,?,?,?)",
                (name, from_branch, at_seq, created_seq, label or name))
            return {"name": name, "parent_branch": from_branch,
                    "fork_seq": at_seq}

    # ------------------------------------------------------------------
    # export / import
    # ------------------------------------------------------------------
    def export_bundle(self):
        events = self.events()
        return {
            "bundle_type": "itc_titration_review",
            "schema_version": SCHEMA_VERSION,
            "exported_at": utc_now(),
            "branches": self.list_branches(),
            "events": events,
        }

    @classmethod
    def import_bundle(cls, bundle, path, replace=True):
        """Validate then atomically load a bundle into a fresh database.

        A half-written or invalid bundle never mutates a usable database:
        validation runs first, then the target file is written via SQLite's
        own transaction.
        """
        problems = validate_bundle(bundle)
        if problems:
            return None, problems
        if replace and os.path.exists(path):
            os.remove(path)
        store = cls(path)
        try:
            with store.conn:
                cur = store.conn.cursor()
                cur.execute("DELETE FROM events")
                cur.execute("DELETE FROM branches")
                cur.execute("DELETE FROM results_cache")
                for b in bundle["branches"]:
                    cur.execute(
                        "INSERT INTO branches(name,parent_branch,fork_seq,"
                        "created_seq,label) VALUES(?,?,?,?,?)",
                        (b["name"], b.get("parent_branch"), b["fork_seq"],
                         b.get("created_seq", 0), b.get("label")))
                for ev in sorted(bundle["events"], key=lambda e: e["seq"]):
                    cur.execute(
                        "INSERT INTO events(seq,event_id,branch,kind,payload,"
                        "parent_hash,timestamp,caused_by) VALUES(?,?,?,?,?,?,?,?)",
                        (ev["seq"], ev["event_id"], ev["branch"], ev["kind"],
                         json.dumps(ev["payload"], ensure_ascii=False),
                         ev["parent_hash"], ev["timestamp"],
                         ev.get("caused_by")))
                    if ev["kind"] == "RESULT_ISSUED":
                        result = ev["payload"]["result"]
                        cur.execute(
                            "INSERT OR IGNORE INTO results_cache"
                            "(result_id,result_kind,body,created_seq)"
                            " VALUES(?,?,?,?)",
                            (result[ev["payload"]["id_field"]],
                             ev["payload"]["result_kind"],
                             json.dumps(result, ensure_ascii=False), ev["seq"]))
        except Exception:
            store.close()
            if os.path.exists(path):
                os.remove(path)
            raise
        return store, []


def _dedup_diags(diags):
    seen, out = set(), []
    for d in diags:
        key = canonical(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def validate_bundle(bundle):
    problems = []
    if not isinstance(bundle, dict):
        return ["bundle is not an object"]
    if bundle.get("bundle_type") != "itc_titration_review":
        problems.append("wrong bundle_type")
    events = bundle.get("events")
    branches = {b["name"] for b in bundle.get("branches", [])}
    if not isinstance(events, list):
        return problems + ["events not a list"]
    last_hash_by_branch = {}
    for ev in sorted(events, key=lambda e: e["seq"]):
        if ev["branch"] not in branches:
            problems.append(f"event {ev['seq']}: unknown branch {ev['branch']}")
        expected = make_event(ev["seq"], ev["kind"], ev["payload"],
                              ev["parent_hash"], ev["branch"], ev["timestamp"],
                              ev.get("caused_by"))
        if expected["event_id"] != ev["event_id"]:
            problems.append(f"event {ev['seq']}: event_id does not match content")
        prev = last_hash_by_branch.get(ev["branch"], "GENESIS")
        if ev["parent_hash"] != prev:
            problems.append(f"event {ev['seq']}: broken parent chain in branch "
                            f"{ev['branch']}")
        last_hash_by_branch[ev["branch"]] = ev["event_id"]
    seqs = [ev["seq"] for ev in events]
    if seqs != list(range(1, len(events) + 1)):
        problems.append("event seq values are not a contiguous 1..N sequence")
    return problems
