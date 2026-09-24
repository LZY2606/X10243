"""Event-sourced project state with branch replay and time travel."""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import (BRANCH_CREATED, DECISION_TYPES, DERIVED_TYPES,
                     MATERIAL_COMMITTED, PARAMS_COMMITTED, REVIEW_ACCEPT,
                     REVIEW_ADJUST, REVIEW_REJECT, Event, digest)

DEFAULT_BRANCH = "main"
REVIEW_TYPES = {REVIEW_ACCEPT, REVIEW_REJECT, REVIEW_ADJUST}


@dataclass
class Projection:
    branch: str
    seq: int
    as_of_event_id: str | None
    materials: dict = field(default_factory=dict)        # (kind,role) -> record
    material_history: list = field(default_factory=list)
    params_versions: dict = field(default_factory=dict)  # family -> version event
    decisions: dict = field(default_factory=dict)        # type -> latest payload
    decision_chain: dict = field(default_factory=dict)   # type -> [payload...]
    derived: dict = field(default_factory=dict)          # type -> latest payload
    derived_history: list = field(default_factory=list)
    reviews: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    lineage_events: list = field(default_factory=list)
    events: list = field(default_factory=list)
    branch_info: dict = field(default_factory=dict)

    def material(self, kind, role):
        return self.materials.get((kind, role))


class Project:
    def __init__(self):
        self.events: list[Event] = []
        self.branches: dict[str, dict] = {}

    # ---- mutation -----------------------------------------------------------
    def init_branch(self, branch=DEFAULT_BRANCH):
        if branch not in self.branches:
            self.branches[branch] = {"base_branch": None,
                                     "base_event_id": None, "seq": 0}

    def next_seq(self, branch) -> int:
        info = self.branches[branch]
        info["seq"] += 1
        return info["seq"]

    def append(self, branch, type_, payload, *, parent_event_id=None,
               inputs_digest=None, parent_dependencies=None, note="",
               author="reviewer", epoch_us=0) -> Event:
        self.init_branch(branch)
        seq = self.next_seq(branch)
        ph = digest(payload)
        event_id = f"ev-{branch}-{seq:04d}-{ph[:10]}"
        ev = Event(event_id=event_id, seq=seq, branch=branch,
                   parent_event_id=parent_event_id, type=type_,
                   payload=payload, content_hash=ph,
                   inputs_digest=inputs_digest,
                   parent_dependencies=parent_dependencies or [],
                   author=author, note=note, epoch_us=epoch_us)
        self.events.append(ev)
        return ev

    def create_branch(self, new_branch, base_branch, base_event_id,
                      note="") -> Event:
        if new_branch in self.branches:
            raise ValueError(f"branch {new_branch} already exists")
        self.init_branch(base_branch)
        self.branches[new_branch] = {
            "base_branch": base_branch,
            "base_event_id": base_event_id, "seq": 0}
        payload = {"new_branch": new_branch, "base_branch": base_branch,
                   "base_event_id": base_event_id, "note": note}
        self.init_branch(new_branch)
        seq = self.next_seq(new_branch)
        ph = digest(payload)
        ev = Event(event_id=f"ev-{new_branch}-{seq:04d}-{BRANCH_CREATED}-{ph[:10]}",
                   seq=seq, branch=new_branch, parent_event_id=base_event_id,
                   type=BRANCH_CREATED, payload=payload, content_hash=ph,
                   note=note)
        self.events.append(ev)
        return ev

    # ---- queries ------------------------------------------------------------
    def branch_events(self, branch) -> list[Event]:
        return sorted((e for e in self.events if e.branch == branch),
                      key=lambda e: e.seq)

    def event_by_id(self, event_id) -> Event | None:
        return next((e for e in self.events if e.event_id == event_id), None)


def _lineage(project: Project, branch: str) -> list[Event]:
    """Ordered events for a branch, including its base-branch prefix."""
    info = project.branches[branch]
    if info["base_branch"] is None:
        return project.branch_events(branch)
    base_events = _lineage(project, info["base_branch"])
    cutoff_id = info["base_event_id"]
    prefix = []
    for e in base_events:
        prefix.append(e)
        if e.event_id == cutoff_id:
            break
    own = [e for e in project.branch_events(branch)
           if e.type != BRANCH_CREATED]
    return prefix + own


def replay(project: Project, branch: str = DEFAULT_BRANCH,
           as_of_event_id: str | None = None,
           verify: bool = False) -> Projection:
    project.init_branch(branch)
    pj = Projection(branch=branch, seq=0, as_of_event_id=as_of_event_id)
    pj.branch_info = dict(project.branches[branch])
    lineage = _lineage(project, branch)
    for e in lineage:
        if as_of_event_id is not None:
            if e.event_id == as_of_event_id:
                _apply(pj, e)
                pj.as_of_event_id = as_of_event_id
                break
        _apply(pj, e, verify=verify)
        if as_of_event_id is None:
            pj.seq = max(pj.seq, e.seq)
    if as_of_event_id is None and lineage:
        pj.seq = max(e.seq for e in lineage)
    pj.lineage_events = [ev.event_id for ev in pj.events]
    return pj


def _apply(pj: Projection, e: Event, verify: bool = False):
    pj.events.append(e)
    p = e.payload
    if e.type == MATERIAL_COMMITTED:
        key = (p["record"]["kind"], p["record"].get("role", ""))
        pj.materials[key] = p["record"]
        pj.material_history.append({"event_id": e.event_id, "key": list(key),
                                    "material_id": p["record"]["material_id"],
                                    "version": p["record"]["version"]})
    elif e.type == PARAMS_COMMITTED:
        pj.params_versions[p["family"]] = p
    elif e.type in DECISION_TYPES:
        chain = pj.decision_chain.setdefault(e.type, [])
        if chain:
            prev = chain[-1]
            prev["superseded_by"] = e.event_id
            p["supersedes"] = prev["event_id"]
            pj.conflicts.append({
                "decision_type": e.type,
                "earlier_event_id": prev["event_id"],
                "later_event_id": e.event_id,
                "resolution": "LATEST_EFFECTIVE_HISTORY_RETAINED",
            })
        p["event_id"] = e.event_id
        p["branch"] = e.branch
        chain.append(dict(p))
        pj.decisions[e.type] = dict(p)
    elif e.type in DERIVED_TYPES:
        if verify and e.inputs_digest is not None:
            check = digest(_verification_input(e))
            if check != e.inputs_digest:
                raise ValueError(
                    f"derived event {e.event_id} failed input verification: "
                    f"stored {e.inputs_digest[:10]} != replay {check[:10]}")
        p["event_id"] = e.event_id
        p["branch"] = e.branch
        pj.derived[e.type] = p
        pj.derived_history.append({"event_id": e.event_id, "type": e.type,
                                   "set_id": p.get("set_id")})
    elif e.type in REVIEW_TYPES:
        pj.reviews.append({"event_id": e.event_id, "type": e.type,
                           "payload": p, "note": e.note})


def _verification_input(e: Event):
    # stable subset re-hashed at replay time by the service is embedded in
    # payload under _inputs; storage layer compares the stored digest.
    return e.payload.get("_inputs")
