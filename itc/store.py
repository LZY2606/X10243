"""SQLite storage layer.

Layers:
  - materials: immutable, separately versioned per kind (power sampling,
    injection schedule, concentrations, temperature, cell volume, baseline,
    blank curve). Raw imported representation is preserved verbatim.
  - events: append-only, monotonically sequenced, branch-labelled. Every
    mutation (import, revision, computation, decision) is an event whose
    payload is machine-readable.
  - results: derived computations (integration sets, dilution corrections,
    model-fit sets). Each records input digest, parameter version and parent
    dependencies. Results are immutable; a newer dependency version marks
    dependents STALE instead of rewriting them.
  - decisions: human accept/reject/adjust verdicts.

State at any historical event is a pure query (time travel): nothing is ever
overwritten, so revisiting event N restores exactly the dependency versions
and visible conclusions of that moment.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Optional

from .diagnostics import Diagnostic, Outcome, Status

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_by_event TEXT NOT NULL,
    PRIMARY KEY (material_id, version)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    branch TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    parents TEXT NOT NULL,
    UNIQUE (seq, branch)
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    param_version TEXT NOT NULL,
    parents TEXT NOT NULL,
    payload TEXT NOT NULL,
    diagnostics TEXT NOT NULL,
    created_by_event TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    target TEXT NOT NULL,
    verdict TEXT NOT NULL,
    rationale TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

MATERIAL_KINDS = (
    "power_trace", "injection_schedule", "concentrations", "temperature",
    "cell_volume", "baseline", "blank_trace",
)


def digest_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        with self.db:
            self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # ------------------------------------------------------------- events
    def add_event(self, kind: str, payload: dict, branch: str = "main",
                  parents: Optional[list] = None, event_id: Optional[str] = None,
                  seq: Optional[int] = None, conn=None) -> dict:
        with self._lock:
            db = conn or self.db
            if seq is None:
                row = db.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS s FROM events").fetchone()
                seq = row["s"]
            if event_id is None:
                event_id = "evt_" + hashlib.sha256(
                    json.dumps({"seq": seq, "branch": branch, "kind": kind,
                                "payload": payload}, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]
            parents = parents or []
            db.execute(
                "INSERT INTO events(event_id, seq, branch, kind, payload, parents) VALUES (?,?,?,?,?,?)",
                (event_id, seq, branch, kind, json.dumps(payload, sort_keys=True, default=str),
                 json.dumps(parents)),
            )
            return {"event_id": event_id, "seq": seq, "branch": branch, "kind": kind,
                    "payload": payload, "parents": parents}

    # ---------------------------------------------------------- materials
    def put_material(self, material_id: str, kind: str, payload: dict,
                     event_id: str, version: Optional[int] = None, conn=None) -> dict:
        if kind not in MATERIAL_KINDS:
            raise ValueError(f"unknown material kind {kind!r}")
        with self._lock:
            db = conn or self.db
            if version is None:
                row = db.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM materials WHERE material_id=?",
                    (material_id,)).fetchone()
                version = row["v"]
            dg = digest_json({"material_id": material_id, "kind": kind,
                              "version": version, "payload": payload})
            db.execute(
                "INSERT INTO materials(material_id, version, kind, payload, digest, created_by_event)"
                " VALUES (?,?,?,?,?,?)",
                (material_id, version, kind, json.dumps(payload, sort_keys=True, default=str), dg, event_id),
            )
            return {"material_id": material_id, "version": version, "kind": kind, "digest": dg}

    def get_material(self, material_id: str, version: Optional[int] = None,
                     at_seq: Optional[int] = None, branch: Optional[str] = None) -> Optional[dict]:
        q = ("SELECT m.* FROM materials m JOIN events e ON m.created_by_event = e.event_id"
             " WHERE m.material_id=?")
        args: list = [material_id]
        if version is not None:
            q += " AND m.version=?"
            args.append(version)
        if at_seq is not None:
            q += " AND e.seq<=?"
            args.append(at_seq)
        if branch is not None:
            q += " AND e.branch=?"
            args.append(branch)
        q += " ORDER BY m.version DESC LIMIT 1"
        row = self.db.execute(q, args).fetchone()
        if row is None:
            return None
        return {"material_id": row["material_id"], "version": row["version"], "kind": row["kind"],
                "payload": json.loads(row["payload"]), "digest": row["digest"],
                "created_by_event": row["created_by_event"]}

    def materials_at(self, at_seq: Optional[int] = None, branch: Optional[str] = None) -> list:
        """Latest version of every material visible at a given event seq."""
        q = ("SELECT m.* FROM materials m JOIN events e ON m.created_by_event=e.event_id"
             " JOIN (SELECT material_id, MAX(version) v FROM materials GROUP BY material_id) latest"
             "   ON m.material_id=latest.material_id AND m.version=latest.v WHERE 1=1")
        args: list = []
        if at_seq is not None:
            q += " AND e.seq<=?"
            args.append(at_seq)
        if branch is not None:
            q += " AND e.branch=?"
            args.append(branch)
        q += " ORDER BY m.kind, m.material_id"
        return [{"material_id": r["material_id"], "version": r["version"], "kind": r["kind"],
                 "payload": json.loads(r["payload"]), "digest": r["digest"],
                 "created_by_event": r["created_by_event"]}
                for r in self.db.execute(q, args).fetchall()]

    # ------------------------------------------------------------ results
    def put_result(self, kind: str, status: str, payload, parents: list,
                   param_version: str, event_id: str, diagnostics: Optional[list] = None,
                   result_id: Optional[str] = None, conn=None) -> dict:
        input_digest = digest_json({"parents": parents, "param_version": param_version})
        if result_id is None:
            result_id = "res_" + hashlib.sha256(
                json.dumps({"kind": kind, "input_digest": input_digest,
                            "payload": payload}, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        diags = [d.to_dict() if isinstance(d, Diagnostic) else d for d in (diagnostics or [])]
        with self._lock:
            db = conn or self.db
            db.execute(
                "INSERT OR REPLACE INTO results(result_id, kind, status, input_digest, param_version,"
                " parents, payload, diagnostics, created_by_event) VALUES (?,?,?,?,?,?,?,?,?)",
                (result_id, kind, status, input_digest, param_version,
                 json.dumps(parents, sort_keys=True), json.dumps(payload, sort_keys=True, default=str),
                 json.dumps(diags, sort_keys=True), event_id),
            )
        return {"result_id": result_id, "kind": kind, "status": status,
                "input_digest": input_digest, "param_version": param_version}

    def get_result(self, result_id: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM results WHERE result_id=?", (result_id,)).fetchone()
        return self._row_to_result(row) if row else None

    def results_at(self, kind: Optional[str] = None, at_seq: Optional[int] = None,
                   branch: Optional[str] = None) -> list:
        q = ("SELECT r.* FROM results r JOIN events e ON r.created_by_event=e.event_id WHERE 1=1")
        args: list = []
        if kind:
            q += " AND r.kind=?"
            args.append(kind)
        if at_seq is not None:
            q += " AND e.seq<=?"
            args.append(at_seq)
        if branch is not None:
            q += " AND e.branch=?"
            args.append(branch)
        q += " ORDER BY r.created_by_event, r.result_id"
        return [self._row_to_result(r) for r in self.db.execute(q, args).fetchall()]

    def set_result_status(self, result_id: str, status: str, conn=None):
        with self._lock:
            db = conn or self.db
            db.execute("UPDATE results SET status=? WHERE result_id=?", (status, result_id))

    @staticmethod
    def _row_to_result(row) -> dict:
        return {"result_id": row["result_id"], "kind": row["kind"], "status": row["status"],
                "input_digest": row["input_digest"], "param_version": row["param_version"],
                "parents": json.loads(row["parents"]), "payload": json.loads(row["payload"]),
                "diagnostics": json.loads(row["diagnostics"]),
                "created_by_event": row["created_by_event"]}

    # ----------------------------------------------------------- decisions
    def put_decision(self, event_id: str, branch: str, target: str, verdict: str,
                     rationale: str, decision_id: Optional[str] = None, conn=None) -> dict:
        if verdict not in ("accepted", "rejected", "adjusted"):
            raise ValueError(f"invalid verdict {verdict!r}")
        if decision_id is None:
            decision_id = "dec_" + hashlib.sha256(
                f"{event_id}|{target}|{verdict}|{rationale}".encode()).hexdigest()[:16]
        with self._lock:
            db = conn or self.db
            db.execute(
                "INSERT OR REPLACE INTO decisions(decision_id, event_id, branch, target, verdict, rationale)"
                " VALUES (?,?,?,?,?,?)",
                (decision_id, event_id, branch, target, verdict, rationale))
        return {"decision_id": decision_id, "event_id": event_id, "target": target, "verdict": verdict}

    def decisions_at(self, at_seq: Optional[int] = None, branch: Optional[str] = None) -> list:
        q = ("SELECT d.* FROM decisions d JOIN events e ON d.event_id=e.event_id WHERE 1=1")
        args: list = []
        if at_seq is not None:
            q += " AND e.seq<=?"
            args.append(at_seq)
        if branch is not None:
            q += " AND d.branch=?"
            args.append(branch)
        q += " ORDER BY e.seq, d.decision_id"
        return [dict(r) for r in self.db.execute(q, args).fetchall()]

    # -------------------------------------------------------------- events
    def events(self, branch: Optional[str] = None, up_to_seq: Optional[int] = None) -> list:
        q = "SELECT * FROM events WHERE 1=1"
        args: list = []
        if branch:
            q += " AND branch=?"
            args.append(branch)
        if up_to_seq is not None:
            q += " AND seq<=?"
            args.append(up_to_seq)
        q += " ORDER BY seq, branch, event_id"
        return [{"event_id": r["event_id"], "seq": r["seq"], "branch": r["branch"],
                 "kind": r["kind"], "payload": json.loads(r["payload"]),
                 "parents": json.loads(r["parents"])}
                for r in self.db.execute(q, args).fetchall()]

    def latest_seq(self) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(seq),0) AS s FROM events").fetchone()
        return row["s"]

    # -------------------------------------------------------- export/import
    def export_dict(self) -> dict:
        return {
            "format": "itc-project/1",
            "materials": [dict(r) for r in self.db.execute(
                "SELECT * FROM materials ORDER BY material_id, version").fetchall()],
            "events": [dict(r) for r in self.db.execute(
                "SELECT * FROM events ORDER BY seq, branch, event_id").fetchall()],
            "results": [dict(r) for r in self.db.execute(
                "SELECT * FROM results ORDER BY created_by_event, result_id").fetchall()],
            "decisions": [dict(r) for r in self.db.execute(
                "SELECT * FROM decisions ORDER BY decision_id").fetchall()],
        }

    def import_dict(self, data: dict) -> Outcome:
        """Import a previously exported project.

        Atomic: any failure rolls the whole import back, so a half-written
        import can never leave a partially populated database. Identities,
        ordering (event seq) and provenance (parents) are preserved verbatim.
        """
        if not isinstance(data, dict) or data.get("format") != "itc-project/1":
            return Outcome(Status.FORMAT_CORRUPT, diagnostics=[Diagnostic(
                code="format_corrupt",
                message="not an itc-project/1 export document",
                path="$",
            )])
        try:
            with self._lock, self.db:
                for e in data["events"]:
                    self.db.execute(
                        "INSERT OR REPLACE INTO events(event_id, seq, branch, kind, payload, parents)"
                        " VALUES (?,?,?,?,?,?)",
                        (e["event_id"], e["seq"], e["branch"], e["kind"], e["payload"], e["parents"]))
                for m in data["materials"]:
                    self.db.execute(
                        "INSERT OR REPLACE INTO materials(material_id, version, kind, payload, digest,"
                        " created_by_event) VALUES (?,?,?,?,?,?)",
                        (m["material_id"], m["version"], m["kind"], m["payload"], m["digest"],
                         m["created_by_event"]))
                for r in data["results"]:
                    self.db.execute(
                        "INSERT OR REPLACE INTO results(result_id, kind, status, input_digest,"
                        " param_version, parents, payload, diagnostics, created_by_event)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (r["result_id"], r["kind"], r["status"], r["input_digest"], r["param_version"],
                         r["parents"], r["payload"], r["diagnostics"], r["created_by_event"]))
                for d in data["decisions"]:
                    self.db.execute(
                        "INSERT OR REPLACE INTO decisions(decision_id, event_id, branch, target, verdict,"
                        " rationale) VALUES (?,?,?,?,?,?)",
                        (d["decision_id"], d["event_id"], d["branch"], d["target"], d["verdict"],
                         d["rationale"]))
        except (KeyError, TypeError) as exc:
            return Outcome(Status.FORMAT_CORRUPT, diagnostics=[Diagnostic(
                code="format_corrupt", message=f"missing/invalid field: {exc}", path="$")])
        except sqlite3.Error as exc:
            return Outcome(Status.FORMAT_CORRUPT, diagnostics=[Diagnostic(
                code="format_corrupt", message=f"import rejected by storage layer: {exc}", path="$")])
        return Outcome(Status.OK, payload={
            "events": len(data["events"]), "materials": len(data["materials"]),
            "results": len(data["results"]), "decisions": len(data["decisions"])})
