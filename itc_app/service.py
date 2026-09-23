"""Event-sourced project service.

Layering
--------
Immutable materials (seven kinds, separately versioned):
power, injections, concentration, temperature, cell_volume, baseline, blank.

Mutable decisions live only as events: first-injection discard, manual
integration endpoints, blank-correspondence rule, accepted/rejected model,
parameter adjustments, conflicts and branch points.

All derived results (integrations, fits) are immutable records carrying an
input summary, the rule version and parent material versions.  Replaying
events reconstructs exactly what was visible at the time; a later revision
marks downstream results stale but never overwrites them.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from . import storage
from .binding import BindingInput, fit_models
from .errors import (FormatCorrupt, ITCError, ResourceBudgetExhausted,
                     SemanticUnresolvable, diag)
from .identity import event_id, material_id, result_id
from .integration import INTEGRATION_RULE_VERSION, integrate_batch
from .models import MATERIAL_KINDS
from .parsers import (parse_baseline, parse_blank, parse_cell_volume,
                      parse_concentration, parse_injections, parse_power,
                      parse_temperature)

EXT_KIND = {
    ".itcpower": "power", ".itcinj": "injections", ".itcconc": "concentration",
    ".itctemp": "temperature", ".itccell": "cell_volume",
    ".itcbase": "baseline", ".itcblank": "blank",
}


@dataclass
class ProjectState:
    project_id: str
    name: str
    branches: dict[str, list[dict]] = field(default_factory=dict)
    materials: dict[str, dict] = field(default_factory=dict)

    def fresh(self, branch: str = "main") -> "View":
        return View(self.project_id, self.name, branch)


@dataclass
class View:
    project_id: str
    name: str
    branch: str
    events: list[dict] = field(default_factory=list)
    materials: dict[str, dict] = field(default_factory=dict)
    runs: dict[str, dict] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    rules: dict[str, dict] = field(default_factory=dict)
    branch_points: list[dict] = field(default_factory=list)
    head_seq: int = -1

    def result(self, rid):
        return self.results.get(rid)


# ---------------------------------------------------------------------------
# Parsing dispatch
# ---------------------------------------------------------------------------

def parse_material(kind: str, content: str):
    """Return (parsed_dict, diagnostics) for a material kind.

    The verbatim content is always stored separately; parsing failures raise
    one of the three distinct domain errors.
    """
    if kind == "power":
        trace, meta = parse_power(content)
        return {"trace": trace.to_dict(), "meta": meta}, [
            d for d in trace.diagnostics]
    if kind == "baseline":
        trace, meta = parse_baseline(content)
        return {"trace": trace.to_dict(), "meta": meta}, list(trace.diagnostics)
    if kind == "blank":
        trace, meta = parse_blank(content)
        return {"trace": trace.to_dict(), "meta": meta}, list(trace.diagnostics)
    if kind == "injections":
        injections, meta = parse_injections(content)
        diags = []
        for inj in injections:
            if not inj.convertible and inj.diagnostic:
                diags.append(inj.diagnostic)
        return {"injections": [i.to_dict() for i in injections],
                "meta": meta}, diags
    if kind == "concentration":
        parsed, diags = parse_concentration(content)
        return parsed, diags
    if kind == "temperature":
        parsed, diags = parse_temperature(content)
        return parsed, diags
    if kind == "cell_volume":
        parsed, diags = parse_cell_volume(content)
        return parsed, diags
    raise SemanticUnresolvable(f"unknown material kind {kind!r}",
                               [diag("unknown_kind", "not a versioned kind")])


def _trace_from_parsed(parsed):
    from .models import Sample, Trace
    td = parsed["trace"]
    samples = [Sample(**{k: s.get(k) for k in
                         ("index", "raw_time", "raw_value", "time_si",
                          "value_si", "time_unit", "value_unit",
                          "convertible", "diagnostic", "is_gap")})
               for s in td["samples"]]
    return Trace(samples=samples, origin=td["origin"],
                 gaps=[tuple(g) for g in td.get("gaps", [])],
                 diagnostics=td.get("diagnostics", []),
                 valid_range=tuple(td.get("valid_range", (None, None))))


def _injections_from_parsed(parsed):
    from .models import Injection
    out = []
    for i in parsed["injections"]:
        out.append(Injection(**{k: i.get(k) for k in
                                ("index", "time_si", "volume_si", "raw_time",
                                 "raw_volume", "time_unit", "volume_unit",
                                 "convertible", "diagnostic")}))
    return out


class ProjectService:
    """Server-side, test-callable business rules over one SQLite database."""

    def __init__(self, conn):
        self.conn = conn

    # -- project / material ingestion ------------------------------------
    def create_project(self, project_id: str, name: str, metadata=None):
        storage.upsert_project(self.conn, project_id, name, metadata)
        self.conn.commit()
        return project_id

    def ingest_material(self, project_id: str, kind: str, file_name: str,
                        content: str) -> dict[str, Any]:
        if kind not in MATERIAL_KINDS:
            raise SemanticUnresolvable(
                f"{kind!r} is not one of the seven versioned material kinds",
                [diag("unknown_kind", "must be one of " + ",".join(MATERIAL_KINDS),
                      file_name)])
        parsed, diags = parse_material(kind, content)
        mid = material_id(kind, file_name, content)
        sha = hashlib.sha256(content.encode("utf-8", "surrogateescape")).hexdigest()
        record = {"material_id": mid, "kind": kind, "file_name": file_name,
                  "content": content, "media_type": "text/plain",
                  "content_sha256": sha, "parsed": parsed,
                  "diagnostics": diags}
        storage.put_material(self.conn, project_id, mid, kind, file_name, content)
        self.conn.commit()
        return record

    def material_record(self, project_id, mid):
        row = storage.get_material(self.conn, project_id, mid)
        if not row:
            return None
        parsed, diags = parse_material(row["kind"], row["content"])
        sha = hashlib.sha256(row["content"].encode("utf-8", "surrogateescape")).hexdigest()
        return {"material_id": mid, "kind": row["kind"],
                "file_name": row["file_name"], "content": row["content"],
                "media_type": row["media_type"], "content_sha256": sha,
                "parsed": parsed, "diagnostics": diags}

    def list_materials(self, project_id):
        out = []
        for row in storage.all_materials(self.conn, project_id):
            out.append({"material_id": row["material_id"], "kind": row["kind"],
                        "file_name": row["file_name"],
                        "content_sha256": hashlib.sha256(
                            row["content"].encode("utf-8", "surrogateescape")).hexdigest()})
        return out
