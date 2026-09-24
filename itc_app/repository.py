"""Read/write operations over the SQLite store."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from . import db as dbmod
from .identity import child_id, stable_id

MATERIAL_KINDS = {
    "power_curve",
    "injection_schedule",
    "concentration_set",
    "meta",
    "blank_curve",
    "baseline",
}


class Repo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------------- counters ----------------
    def next_order(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(created_at_order),0)+1 AS n FROM events").fetchone()
        return int(row["n"])

    def next_branch_seq(self, branch: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS n FROM events WHERE branch=?", (branch,)
        ).fetchone()
        return int(row["n"])

    # ---------------- materials ----------------
    def material_exists(self, material_id: str, version: int) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM material_versions WHERE material_id=? AND version=?",
            (material_id, version),
        ).fetchone() is not None

    def latest_version(self, material_id: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM material_versions WHERE material_id=?", (material_id,)
        ).fetchone()
        return None if row["v"] is None else int(row["v"])

    def find_material_by_hash(self, kind: str, content_hash: str) -> Optional[tuple[str, int]]:
        row = self.conn.execute(
            "SELECT material_id, version FROM material_versions WHERE kind=? AND content_hash=? "
            "ORDER BY version LIMIT 1",
            (kind, content_hash),
        ).fetchone()
        return None if row is None else (row["material_id"], int(row["version"]))

    def material_id_for(self, kind: str, logical_key: str) -> str:
        return stable_id("mat", {"kind": kind, "key": logical_key})

    def add_material(
        self,
        *,
        material_id: str,
        kind: str,
        content_hash: str,
        source_label: str,
        payload: dict,
        origin_uri: Optional[str] = None,
        supersedes: Optional[str] = None,
        created_event: Optional[str] = None,
    ) -> int:
        version = (self.latest_version(material_id) or 0) + 1
        self.conn.execute(
            "INSERT INTO material_versions(material_id,kind,version,content_hash,source_label,"
            "origin_uri,payload_json,supersedes,created_event) VALUES(?,?,?,?,?,?,?,?,?)",
            (material_id, kind, version, content_hash, source_label, origin_uri,
             dbmod.jdumps(payload), supersedes, created_event),
        )
        return version

    def add_point(self, *, material_id: str, version: int, seq: int, point: dict) -> None:
        self.conn.execute(
            "INSERT INTO material_points(material_id,version,seq,coord_raw,value_raw,"
            "coord_value,value_value,coord_unit,value_unit,coord_diag,value_diag,gap_after) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (material_id, version, seq, point["coord_raw"], point["value_raw"],
             point.get("coord_value"), point.get("value_value"),
             point.get("coord_unit"), point.get("value_unit"),
             dbmod.jdumps(point["coord_diag"]) if point.get("coord_diag") else None,
             dbmod.jdumps(point["value_diag"]) if point.get("value_diag") else None,
             1 if point.get("gap_after") else 0),
        )

    def get_material(self, material_id: str, version: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM material_versions WHERE material_id=? AND version=?",
            (material_id, version),
        ).fetchone()
        if row is None:
            raise KeyError(f"missing material {material_id} v{version}")
        data = dict(row)
        data["payload"] = json.loads(data["payload_json"])
        return data

    def list_material_versions(self, material_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT material_id,kind,version,content_hash,source_label,supersedes,created_event "
            "FROM material_versions WHERE material_id=? ORDER BY version",
            (material_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_points(self, material_id: str, version: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM material_points WHERE material_id=? AND version=? ORDER BY seq",
            (material_id, version),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["coord_diag"] = json.loads(d["coord_diag"]) if d["coord_diag"] else None
            d["value_diag"] = json.loads(d["value_diag"]) if d["value_diag"] else None
            out.append(d)
        return out

    # ---------------- ingest attempts ----------------
    def add_ingest_attempt(self, *, attempt_id: Optional[str], kind: str, status: str,
                           source_label: str, fragment: str, diagnostics: list,
                           material_id: Optional[str] = None, version: Optional[int] = None,
                           event_id: Optional[str] = None) -> str:
        attempt_id = attempt_id or stable_id("ingest",
                                             {"kind": kind, "label": source_label,
                                              "fragment": fragment, "status": status})
        self.conn.execute(
            "INSERT OR REPLACE INTO ingest_attempts(attempt_id,kind,status,source_label,"
            "fragment,diagnostics,material_id,version,event_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (attempt_id, kind, status, source_label, fragment, dbmod.jdumps(diagnostics),
             material_id, version, event_id),
        )
        return attempt_id

    def list_ingest_attempts(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM ingest_attempts ORDER BY attempt_id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["diagnostics"] = json.loads(d["diagnostics"])
            out.append(d)
        return out

    # ---------------- rules ----------------
    def ensure_rule(self, *, family: str, version: int, params: dict) -> str:
        rule_id = stable_id("rule", {"family": family, "version": version, "params": params})
        existing = self.conn.execute("SELECT rule_id FROM rule_versions WHERE rule_id=?",
                                    (rule_id,)).fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO rule_versions(rule_id,family,version,params_json) VALUES(?,?,?,?)",
                (rule_id, family, version, dbmod.jdumps(params)),
            )
        return rule_id

    def get_rule(self, rule_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM rule_versions WHERE rule_id=?", (rule_id,)).fetchone()
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json"))
        return d

    # ---------------- events ----------------
    def add_event(self, *, kind: str, action: str, payload: dict, branch: str = "main",
                  parent_id: Optional[str] = None, accepted: bool = True,
                  event_id: Optional[str] = None) -> dict:
        event_id = event_id or child_id("evt", parent_id or "ROOT",
                                        {"kind": kind, "action": action, "payload": payload})
        seq = self.next_branch_seq(branch)
        order = self.next_order()
        self.conn.execute(
            "INSERT INTO events(event_id,seq,branch,parent_id,kind,action,payload_json,accepted,"
            "created_at_order) VALUES(?,?,?,?,?,?,?,?,?)",
            (event_id, seq, branch, parent_id, kind, action, dbmod.jdumps(payload),
             1 if accepted else 0, order),
        )
        return {"event_id": event_id, "seq": seq, "branch": branch,
                "parent_id": parent_id, "kind": kind, "action": action,
                "payload": payload, "accepted": accepted, "order": order}

    def get_event(self, event_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json"))
        return d

    def branch_events(self, branch: str) -> list[dict]:
        head = self.get_branch(branch)["head_event_id"]
        return self.events_until(head)

    def events_until(self, event_id: str) -> list[dict]:
        chain: list[dict] = []
        current = event_id
        seen = set()
        while current and current not in seen:
            seen.add(current)
            ev = self.get_event(current)
            chain.append(ev)
            current = ev["parent_id"]
        chain.reverse()
        return chain

    def all_events(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY created_at_order").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    # ---------------- branches ----------------
    def ensure_branch(self, name: str, label: str, head_event_id: Optional[str],
                      fork_event_id: Optional[str] = None) -> None:
        self.conn.execute(
            "INSERT INTO branches(name,head_event_id,label,fork_event_id) VALUES(?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET head_event_id=excluded.head_event_id",
            (name, label, head_event_id, fork_event_id),
        )

    def get_branch(self, name: str) -> dict:
        row = self.conn.execute("SELECT * FROM branches WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"unknown branch {name}")
        return dict(row)

    def list_branches(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY name").fetchall()]

    def advance_branch(self, name: str, head_event_id: str) -> None:
        self.conn.execute("UPDATE branches SET head_event_id=? WHERE name=?",
                         (head_event_id, name))

    # ---------------- results ----------------
    def add_result(self, *, result_id: str, event_id: str, family: str, status: str,
                   rule_id: str, input_summary: dict, parents: list[str], payload: dict,
                   rank_index: int = 0) -> None:
        self.conn.execute(
            "INSERT INTO results(result_id,event_id,family,status,rule_id,input_summary_json,"
            "parents_json,payload_json,rank_index) VALUES(?,?,?,?,?,?,?,?,?)",
            (result_id, event_id, family, status, rule_id,
             dbmod.jdumps(input_summary), dbmod.jdumps(parents), dbmod.jdumps(payload), rank_index),
        )

    def add_result_item(self, *, result_id: str, item_key: str, seq: int, payload: dict) -> str:
        item_id = stable_id("item", {"result": result_id, "key": item_key})
        self.conn.execute(
            "INSERT INTO result_items(item_id,result_id,item_key,seq,payload_json) "
            "VALUES(?,?,?,?,?)",
            (item_id, result_id, item_key, seq, dbmod.jdumps(payload)),
        )
        return item_id

    def add_diagnostic(self, *, result_id: Optional[str], event_id: Optional[str],
                       path: str, message: str, payload: Optional[dict] = None) -> str:
        diag_id = stable_id("diag", {"result": result_id, "event": event_id,
                                    "path": path, "message": message})
        self.conn.execute(
            "INSERT OR REPLACE INTO result_diagnostics(diag_id,result_id,event_id,path,"
            "message,payload_json) VALUES(?,?,?,?,?,?)",
            (diag_id, result_id, event_id, path, message, dbmod.jdumps(payload or {})),
        )
        return diag_id

    def results_for_chain(self, event_ids: list[str]) -> list[dict]:
        if not event_ids:
            return []
        marks = ",".join("?" for _ in event_ids)
        rows = self.conn.execute(
            f"SELECT * FROM results WHERE event_id IN ({marks}) ORDER BY rank_index, result_id",
            event_ids,
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("input_summary_json", "parents_json", "payload_json"):
                d[k.removesuffix("_json")] = json.loads(d.pop(k))
            out.append(d)
        return out

    def items_for(self, result_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM result_items WHERE result_id=? ORDER BY seq", (result_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    def diagnostics_for(self, result_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM result_diagnostics WHERE result_id=? ORDER BY diag_id",
            (result_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    def result_by_id(self, result_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM results WHERE result_id=?", (result_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("input_summary_json", "parents_json", "payload_json"):
            d[k.removesuffix("_json")] = json.loads(d.pop(k))
        return d
