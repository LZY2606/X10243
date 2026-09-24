"""Pure event-sourced projection ("structured state, replayable evidence").

Event kinds:

* MATERIAL_ADDED     -- first version of an immutable material
* MATERIAL_SUPERSEDED -- new version of a material kind; the old version is
                         never deleted, it remains an addressable ancestor
* RULE_SET           -- a new version of a computation rule set
* REVIEW_DECISION    -- accept / reject / parameter adjustment by a reviewer
* RESULT_ISSUED      -- a derived result (integral / blank / normalized / fit)

Every event carries a hash-parent chain.  Result identities are
content-addressed over the *consumed* inputs and the rule version, so:

* a concentration revision changes normalized heats and fits only;
* raw integrals are reused unchanged;
* equivalent inputs always produce equivalent identities and order.
"""

from __future__ import annotations

import hashlib
import json

from . import blank as blank_mod
from . import integrate as integ_mod
from . import models as models_mod
from . import normalize as norm_mod
from .hashing import canonical, digest
from .materials import Material
from .rules import (
    DEFAULT_BLANK_RULES,
    DEFAULT_INTEGRATION_RULES,
    DEFAULT_MODEL_RULES,
    RULE_KIND_BLANK,
    RULE_KIND_INTEGRATION,
    RULE_KIND_MODEL,
    rule_id,
)

MATERIAL_KINDS = ("power", "schedule", "conditions", "baseline", "blank")
RESULT_KINDS = ("integral", "blank", "normalized", "fit")


def make_event(seq, kind, payload, parent_hash, branch, timestamp,
               caused_by=None):
    body = {
        "seq": seq,
        "kind": kind,
        "payload": payload,
        "parent": parent_hash,
        "branch": branch,
        "timestamp": timestamp,
        "caused_by": caused_by,
    }
    body["event_id"] = _event_id(body)
    return body


def _event_id(body):
    core = {k: body[k] for k in ("seq", "kind", "payload", "parent",
                                 "branch", "timestamp", "caused_by")}
    return digest("event", core)


def event_hash(ev):
    return hashlib.sha256(canonical(ev).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

class Projection:
    """State visible at one point in a branch's event stream."""

    def __init__(self, branch="main", fork_seq=0):
        self.branch = branch
        self.fork_seq = fork_seq
        self.materials = {}          # kind -> Material (current version)
        self.version_chain = {}      # kind -> [version_id,...]
        self.rules = {
            RULE_KIND_INTEGRATION: dict(DEFAULT_INTEGRATION_RULES),
            RULE_KIND_BLANK: dict(DEFAULT_BLANK_RULES),
            RULE_KIND_MODEL: dict(DEFAULT_MODEL_RULES),
        }
        self.rule_history = {}      # kind -> [{rule_id, body, seq}]
        self.results = {}           # result kind -> latest result dict
        self.result_history = {}    # result kind -> [result_id,...]
        self.decisions = {}         # target -> latest decision payload
        self.events = []
        self.seen_result_ids = set()

    # -- material access -------------------------------------------------
    def mat(self, kind):
        return self.materials.get(kind)

    def rule(self, kind):
        return self.rules[kind]

    # -- application -----------------------------------------------------
    def apply(self, ev):
        kind, payload = ev["kind"], ev["payload"]
        if kind == "MATERIAL_ADDED":
            self._put_material(payload["material"], first=True, source=payload.get("source"))
        elif kind == "MATERIAL_SUPERSEDED":
            self._put_material(payload["material"], first=False,
                               source=payload.get("source"),
                               supersedes=payload.get("supersedes"))
        elif kind == "RULE_SET":
            rk = payload["rule_kind"]
            body = dict(payload["body"])
            self.rules[rk] = body
            self.rule_history.setdefault(rk, []).append(
                {"rule_id": payload["rule_id"], "body": body,
                 "seq": ev["seq"], "reason": payload.get("reason")})
        elif kind == "REVIEW_DECISION":
            self.decisions[payload["target"]] = payload
        elif kind == "RESULT_ISSUED":
            rk = payload["result_kind"]
            result = payload["result"]
            self.results[rk] = result
            self.result_history.setdefault(rk, []).append(result[payload["id_field"]])
            self.seen_result_ids.add(result[payload["id_field"]])
        return self

    def _put_material(self, mdict, first, source, supersedes=None):
        m = Material.from_dict(mdict)
        chain = self.version_chain.setdefault(m.kind, [])
        if m.version_id in chain:
            return
        chain.append(m.version_id)
        self.materials[m.kind] = m


def project(events, branch="main", up_to_seq=None, fork_seq=0):
    """Replay events belonging to ``branch`` up to ``up_to_seq``."""
    proj = Projection(branch=branch, fork_seq=fork_seq)
    for ev in events:
        if ev.get("branch") != branch:
            continue
        if up_to_seq is not None and ev["seq"] > up_to_seq:
            break
        proj.apply(ev)
        proj.events.append(ev)
    return proj


# --------------------------------------------------------------------------
# derived computation with structured reuse
# --------------------------------------------------------------------------

ID_FIELDS = {
    "integral": "result_id",
    "blank": "result_id",
    "normalized": "result_id",
    "fit": "fit_id",
}


def compute_layer(layer, proj):
    """Compute one derived layer from a projection's current inputs."""
    if layer == "integral":
        return _compute_integral(proj)
    if layer == "blank":
        return _compute_blank(proj)
    if layer == "normalized":
        return _compute_normalized(proj)
    if layer == "fit":
        return _compute_fit(proj)
    raise ValueError(layer)


def _compute_integral(proj):
    power, schedule, baseline = proj.mat("power"), proj.mat("schedule"), proj.mat("baseline")
    if not all((power, schedule, baseline)):
        return None, [_missing("integral", [k for k, m in (
            ("power", power), ("schedule", schedule), ("baseline", baseline)) if m is None])]
    res, diags = integ_mod.integrate(power, schedule, baseline,
                                     proj.rules[RULE_KIND_INTEGRATION])
    if res is None:
        return None, diags
    res["kind"] = "integral"
    res["input_summary"] = _input_summary({
        "power": power.version_id, "schedule": schedule.version_id,
        "baseline": baseline.version_id,
        "rule": rule_id(RULE_KIND_INTEGRATION, proj.rules[RULE_KIND_INTEGRATION])})
    res["diagnostics"] = res.get("diagnostics", []) + (diags or [])
    return res, res["diagnostics"]


def _compute_blank(proj):
    blank_m, integral = proj.mat("blank"), proj.results.get("integral")
    if not blank_m:
        return None, [_missing("blank", ["blank"])]
    if integral is None:
        return None, [_missing("blank", ["integral"])]
    res = blank_mod.apply_blank(integral, blank_m, proj.rules[RULE_KIND_BLANK])
    res["kind"] = "blank"
    res["input_summary"] = _input_summary({
        "blank": blank_m.version_id, "integral": integral["result_id"],
        "rule": rule_id(RULE_KIND_BLANK, proj.rules[RULE_KIND_BLANK])})
    return res, res["diagnostics"]


def _compute_normalized(proj):
    sched, cond = proj.mat("schedule"), proj.mat("conditions")
    integral, blank_res = proj.results.get("integral"), proj.results.get("blank")
    if not all((sched, cond, integral, blank_res)):
        miss = [n for n, m in (("schedule", sched), ("conditions", cond),
                               ("integral", integral), ("blank", blank_res)) if m is None]
        return None, [_missing("normalized", miss)]
    res = norm_mod.normalized_heats(integral, blank_res, sched, cond)
    res["kind"] = "normalized"
    res["input_summary"] = _input_summary({
        "schedule": sched.version_id, "conditions": cond.version_id,
        "integral": integral["result_id"], "blank": blank_res["result_id"]})
    return res, res["diagnostics"]


def _compute_fit(proj):
    norm = proj.results.get("normalized")
    if norm is None or not norm["rows"]:
        return None, [_missing("fit", ["normalized"])]
    rule_body = proj.rules[RULE_KIND_MODEL]
    rid = rule_id(RULE_KIND_MODEL, rule_body)
    fit = models_mod.fit_models(norm["rows"],
                                evaluation_budget=rule_body["evaluation_budget"],
                                tie_tolerance=rule_body["tie_relative_tolerance"])
    fit["kind"] = "fit"
    fit["fit_id"] = models_mod.fit_identity(norm["rows"], rid, rule_body)
    fit["parents"] = {"normalized": norm["result_id"]}
    fit["rule_id"] = rid
    fit["input_summary"] = _input_summary({
        "normalized": norm["result_id"], "rule": rid})
    return fit, fit["diagnostics"]


def _input_summary(parts):
    s = canonical(parts)
    return {"parts": parts, "sha256": hashlib.sha256(s.encode()).hexdigest()[:16]}


def _missing(layer, names):
    from .diagnostics import SEMANTIC_INVALID, diagnostic
    return diagnostic(SEMANTIC_INVALID,
                      f"cannot compute {layer}: missing/invalid inputs {', '.join(names)}",
                      layer=layer)


LAYERS_BY_EVENT = {
    "MATERIAL_ADDED": {
        "power": ["integral", "blank", "normalized", "fit"],
        "schedule": ["integral", "blank", "normalized", "fit"],
        "baseline": ["integral", "blank", "normalized", "fit"],
        "blank": ["blank", "normalized", "fit"],
        "conditions": ["normalized", "fit"],
    },
    "MATERIAL_SUPERSEDED": {
        "power": ["integral", "blank", "normalized", "fit"],
        "schedule": ["integral", "blank", "normalized", "fit"],
        "baseline": ["integral", "blank", "normalized", "fit"],
        "blank": ["blank", "normalized", "fit"],
        "conditions": ["normalized", "fit"],
    },
    "RULE_SET": {
        RULE_KIND_INTEGRATION: ["integral", "blank", "normalized", "fit"],
        RULE_KIND_BLANK: ["blank", "normalized", "fit"],
        RULE_KIND_MODEL: ["fit"],
    },
    "REVIEW_DECISION": [],
}


def recompute_plan(event_kind, key):
    if event_kind == "REVIEW_DECISION":
        return []
    return list(LAYERS_BY_EVENT[event_kind][key])
