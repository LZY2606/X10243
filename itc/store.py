import json
import sqlite3
import uuid
from contextlib import contextmanager

from . import materials as mat_mod
from .util import canon, digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials(
  material_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  raw_json TEXT NOT NULL,
  norm_json TEXT,
  digest TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  ws_id TEXT,
  PRIMARY KEY(material_id, version)
);
CREATE TABLE IF NOT EXISTS results(
  result_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  seq INTEGER NOT NULL,
  status TEXT NOT NULL,
  param_version TEXT,
  parents_json TEXT NOT NULL,
  payload_json TEXT,
  diagnostics_json TEXT NOT NULL,
  ws_id TEXT
);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  ws_id TEXT
);
CREATE TABLE IF NOT EXISTS write_sets(
  ws_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  label TEXT,
  meta_json TEXT
);
"""


def derive_decisions(events):
    by_target = {}
    for e in events:
        if e["kind"] != "decision":
            continue
        p = e["payload"]
        by_target[p["target"]] = {
            "action": p["action"],
            "event_id": e["event_id"],
            "rationale": p.get("rationale", ""),
        }
    accepts = {t: d for t, d in by_target.items() if d["action"] == "accept"}
    scopes = {}
    for t in accepts:
        scopes.setdefault(t.split("#")[0], []).append(t)
    conflicts = [{"scope": s, "targets": sorted(ts)} for s, ts in scopes.items() if len(ts) > 1]
    return {"by_target": by_target, "accepts": accepts, "conflicts": conflicts}


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._seq = None
        self.recover()

    def close(self):
        self.conn.close()

    def _alloc_seq(self):
        if self._seq is None:
            row = self.conn.execute("SELECT MAX(seq) m FROM events").fetchone()
            self._seq = row["m"] or 0
        self._seq += 1
        return self._seq

    @contextmanager
    def write_set(self, label, meta=None):
        ws_id = "ws-" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO write_sets(ws_id,status,label,meta_json) VALUES(?,?,?,?)",
            (ws_id, "pending", label, json.dumps(meta or {}, ensure_ascii=False)),
        )
        self.conn.commit()
        try:
            yield ws_id
        except Exception:
            self.conn.rollback()
            self.conn.execute("DELETE FROM write_sets WHERE ws_id=?", (ws_id,))
            self.conn.commit()
            raise
        else:
            self.conn.execute("UPDATE write_sets SET status='committed' WHERE ws_id=?", (ws_id,))
            self.conn.commit()

    def recover(self):
        pending = [r["ws_id"] for r in self.conn.execute(
            "SELECT ws_id FROM write_sets WHERE status='pending'")]
        if not pending:
            return []
        for ws in pending:
            for tbl in ("materials", "results", "events"):
                self.conn.execute("DELETE FROM %s WHERE ws_id=?" % tbl, (ws,))
            self.conn.execute("DELETE FROM write_sets WHERE ws_id=?", (ws,))
        self.conn.commit()
        with self.write_set("recovery") as ws:
            self.add_event(ws, "recovery", {
                "discarded_write_sets": pending,
                "note": "half-written write sets rolled back",
            })
        return pending

    def add_material(self, ws, material_id, kind, raw, cause=None, phase=None):
        norm, diags, status = mat_mod.normalize(material_id, kind, raw)
        row = self.conn.execute(
            "SELECT MAX(version) v FROM materials WHERE material_id=?",
            (material_id,)).fetchone()
        version = (row["v"] or 0) + 1
        seq = self._alloc_seq()
        dg = digest({"material_id": material_id, "version": version, "raw": raw})
        self.conn.execute(
            "INSERT INTO materials VALUES(?,?,?,?,?,?,?,?,?,?)",
            (material_id, version, seq, kind, status, canon(raw),
             canon(norm) if norm is not None else None, dg, canon(diags), ws))
        payload = {
            "material_id": material_id,
            "version": version,
            "kind": kind,
            "digest": dg,
            "status": status,
            "diagnostics": diags,
        }
        if cause:
            payload["cause"] = cause
        if phase:
            payload["phase"] = phase
        self.add_event(ws, "material_version", payload)
        return version, dg

    def add_event(self, ws, kind, payload):
        seq = self._alloc_seq()
        event_id = "ev-%04d" % seq
        ts = "L%04d" % seq
        self.conn.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?)",
            (seq, event_id, ts, kind, canon(payload), ws))
        return event_id

    def result_id_for(self, kind, parents, param_version):
        input_digest = digest({"kind": kind, "parents": parents, "param_version": param_version})
        return "%s:%s" % (kind, input_digest[:12]), input_digest

    def add_result(self, ws, kind, status, parents, param_version, payload, diagnostics):
        rid, input_digest = self.result_id_for(kind, parents, param_version)
        row = self.conn.execute(
            "SELECT result_id FROM results WHERE result_id=?", (rid,)).fetchone()
        if row:
            return rid, True
        seq = self._alloc_seq()
        self.conn.execute(
            "INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?)",
            (rid, kind, seq, status, param_version, canon(parents),
             canon(payload) if payload is not None else None, canon(diagnostics), ws))
        return rid, False

    def _mat_row(self, r):
        return {
            "material_id": r["material_id"],
            "version": r["version"],
            "seq": r["seq"],
            "kind": r["kind"],
            "status": r["status"],
            "raw": json.loads(r["raw_json"]),
            "norm": json.loads(r["norm_json"]) if r["norm_json"] else None,
            "digest": r["digest"],
            "diagnostics": json.loads(r["diagnostics_json"]),
        }

    def _result_row(self, r):
        return {
            "result_id": r["result_id"],
            "kind": r["kind"],
            "seq": r["seq"],
            "status": r["status"],
            "param_version": r["param_version"],
            "parents": json.loads(r["parents_json"]),
            "payload": json.loads(r["payload_json"]) if r["payload_json"] else None,
            "diagnostics": json.loads(r["diagnostics_json"]),
        }

    def _event_row(self, r):
        return {
            "seq": r["seq"],
            "event_id": r["event_id"],
            "ts": r["ts"],
            "kind": r["kind"],
            "payload": json.loads(r["payload_json"]),
        }

    def current_materials(self, at_seq=None):
        sql = "SELECT * FROM materials"
        args = []
        if at_seq is not None:
            sql += " WHERE seq<=?"
            args.append(at_seq)
        sql += " ORDER BY seq"
        out = {}
        for r in self.conn.execute(sql, args):
            out[r["material_id"]] = self._mat_row(r)
        return out

    def material(self, material_id, version=None):
        if version is None:
            r = self.conn.execute(
                "SELECT * FROM materials WHERE material_id=? ORDER BY version DESC LIMIT 1",
                (material_id,)).fetchone()
        else:
            r = self.conn.execute(
                "SELECT * FROM materials WHERE material_id=? AND version=?",
                (material_id, version)).fetchone()
        return self._mat_row(r) if r else None

    def events(self, at_seq=None):
        sql = "SELECT * FROM events"
        args = []
        if at_seq is not None:
            sql += " WHERE seq<=?"
            args.append(at_seq)
        sql += " ORDER BY seq"
        return [self._event_row(r) for r in self.conn.execute(sql, args)]

    def results(self, kind=None, at_seq=None):
        sql = "SELECT * FROM results WHERE 1=1"
        args = []
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        if at_seq is not None:
            sql += " AND seq<=?"
            args.append(at_seq)
        sql += " ORDER BY seq"
        return [self._result_row(r) for r in self.conn.execute(sql, args)]

    def latest_result(self, kind, at_seq=None):
        rows = self.results(kind=kind, at_seq=at_seq)
        return rows[-1] if rows else None

    def get_result(self, result_id):
        r = self.conn.execute(
            "SELECT * FROM results WHERE result_id=?", (result_id,)).fetchone()
        return self._result_row(r) if r else None

    def record_decision(self, target, action, rationale="", phase=None, extra=None):
        with self.write_set("decision") as ws:
            dec = derive_decisions(self.events())
            conflicts = []
            if action == "accept":
                scope = target.split("#")[0]
                for t, d in dec["accepts"].items():
                    if t.split("#")[0] == scope and t != target:
                        conflicts.append(d["event_id"])
            payload = {
                "target": target,
                "action": action,
                "rationale": rationale,
                "conflicts_with": conflicts,
            }
            if phase:
                payload["phase"] = phase
            if extra:
                payload.update(extra)
            return self.add_event(ws, "decision", payload)

    def state(self, at_seq=None):
        mats = self.current_materials(at_seq)
        results = self.results(at_seq=at_seq)
        events = self.events(at_seq)
        cur_ver = {m: row["version"] for m, row in mats.items()}
        latest_by_kind = {}
        for r in results:
            latest_by_kind[r["kind"]] = r
        for r in results:
            stale = []
            for dep, ver in r["parents"].items():
                if dep.startswith("mat:"):
                    now = cur_ver.get(dep)
                    if now != ver:
                        stale.append({"dep": dep, "was": ver, "now": now})
                elif dep.startswith("result:"):
                    kind = dep.split(":", 1)[1]
                    now = latest_by_kind.get(kind)
                    now_id = now["result_id"] if now else None
                    if now_id != ver:
                        stale.append({"dep": dep, "was": ver, "now": now_id})
            r["stale"] = stale
        decisions = derive_decisions(events)
        seq = at_seq
        if seq is None:
            seq = events[-1]["seq"] if events else 0
        return {
            "seq": seq,
            "materials": mats,
            "results": results,
            "events": events,
            "decisions": decisions,
            "latest": {k: v["result_id"] for k, v in latest_by_kind.items()},
        }

    def is_empty(self):
        row = self.conn.execute("SELECT COUNT(*) c FROM events").fetchone()
        return row["c"] == 0

    def export_doc(self):
        mats = [self._mat_row(r) for r in self.conn.execute(
            "SELECT * FROM materials ORDER BY material_id, version")]
        results = [self._result_row(r) for r in self.conn.execute(
            "SELECT * FROM results ORDER BY seq")]
        events = [self._event_row(r) for r in self.conn.execute(
            "SELECT * FROM events ORDER BY seq")]
        doc = {
            "format": "itc-archive/1",
            "meta": {
                "archive": "itc-counterexample-archive",
                "case": "titration-review-casepack",
                "narrative": "biophysics lab ITC injection-integration and binding-model review",
            },
            "materials": mats,
            "results": results,
            "events": events,
        }
        doc["digest"] = digest({
            "materials": mats,
            "results": results,
            "events": events,
        })
        return doc

    def import_doc(self, doc):
        if not isinstance(doc, dict) or doc.get("format") != "itc-archive/1":
            raise ValueError("unsupported archive format")
        expect = digest({
            "materials": doc["materials"],
            "results": doc["results"],
            "events": doc["events"],
        })
        if doc.get("digest") != expect:
            raise ValueError("archive digest mismatch: format_corrupt")
        counts = {"materials": 0, "results": 0, "events": 0, "skipped": 0}
        with self.write_set("import") as ws:
            for m in doc["materials"]:
                row = self.conn.execute(
                    "SELECT 1 FROM materials WHERE material_id=? AND version=?",
                    (m["material_id"], m["version"])).fetchone()
                if row:
                    counts["skipped"] += 1
                    continue
                self.conn.execute(
                    "INSERT INTO materials VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (m["material_id"], m["version"], m["seq"], m["kind"], m["status"],
                     canon(m["raw"]), canon(m["norm"]) if m.get("norm") is not None else None,
                     m["digest"], canon(m["diagnostics"]), ws))
                counts["materials"] += 1
            for r in doc["results"]:
                row = self.conn.execute(
                    "SELECT 1 FROM results WHERE result_id=?", (r["result_id"],)).fetchone()
                if row:
                    counts["skipped"] += 1
                    continue
                self.conn.execute(
                    "INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?)",
                    (r["result_id"], r["kind"], r["seq"], r["status"], r["param_version"],
                     canon(r["parents"]),
                     canon(r["payload"]) if r.get("payload") is not None else None,
                     canon(r["diagnostics"]), ws))
                counts["results"] += 1
            for e in doc["events"]:
                row = self.conn.execute(
                    "SELECT 1 FROM events WHERE event_id=? OR seq=?",
                    (e["event_id"], e["seq"])).fetchone()
                if row:
                    counts["skipped"] += 1
                    continue
                self.conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?)",
                    (e["seq"], e["event_id"], e["ts"], e["kind"], canon(e["payload"]), ws))
                counts["events"] += 1
            self._seq = None
            self.add_event(ws, "import", {
                "source_digest": doc.get("digest"),
                "counts": counts,
            })
        return counts
