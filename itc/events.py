"""Append-only events, hash chains, branches and state projection.

Layering:

* **Materials** (``materials.py``) are immutable and versioned per stream.
* **Events** record what was imported and every human decision; each event
  has a parent, so a branch is an authenticated chain.
* **Artifacts** (integrations, model fits) are derived: they record their
  input summary, parameter/rule versions and parent material digests, and
  are recomputed deterministically during replay. Going back to an old event
  restores that event's dependency versions and conclusions - newer
  computations never overwrite history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ident import canonical, digest, event_id, make_id

EVENT_KINDS = (
    "case_created", "material_committed", "first_injection_discard",
    "blank_rule_set", "integration_endpoint", "model_verdict",
    "decision_conflict", "branch_created", "review_note",
)

# which material kinds each decision consumes (for staleness / blast radius)
DECISION_NEEDS = {
    "first_injection_discard": (),
    "blank_rule_set": ("blank",),
    "integration_endpoint": ("power",),
    "model_verdict": ("concentrations", "power", "injections", "blank"),
}


@dataclass
class Event:
    case_id: str
    branch_id: str
    seq: int
    kind: str
    payload: dict[str, Any]
    parent_id: str
    branch_seq: int = 0
    id: str = ""
    label: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = event_id(self.case_id, self.seq, self.parent_id,
                               {"branch": self.branch_id,
                                "branch_seq": self.branch_seq,
                                "kind": self.kind, "payload": self.payload})

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "case_id": self.case_id,
                "branch_id": self.branch_id, "seq": self.seq,
                "branch_seq": self.branch_seq, "kind": self.kind,
                "payload": self.payload, "parent_id": self.parent_id,
                "label": self.label}


def head_chain_id(events: list[Event]) -> str:
    return events[-1].id if events else ""


def base_stream_versions() -> dict[str, None]:
    return {k: None for k in
            ("power", "injections", "concentrations", "temperature",
             "cell_volume", "baseline", "blank")}
