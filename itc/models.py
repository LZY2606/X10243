"""Immutable materials, versioned parameter sets, events and derived nodes.

Layering:
  * Material / ParamSet  - immutable, versioned inputs and rules
  * Event                - append-only decisions and import actions
  * Node                 - content-addressed computation result with an input
                           summary, parameter versions and parent dependencies

Identities are stable hashes of canonical content, so equivalent inputs yield
the same id in a stable (sorted) order, independent of import order.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .errors import Diagnostic

ENGINE_VERSION = "1.0.0"
MAIN_BRANCH = "main"

MaterialKind = Literal[
    "power", "blank", "baseline", "schedule", "blank_schedule",
    "concentration", "runmeta",
]


def stable_hash(payload: Any) -> str:
    """Content hash with sorted keys; stable across processes/order."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _json_default(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Diagnostic):
        return obj.to_dict()
    raise TypeError(f"not serializable: {type(obj)!r}")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=_json_default)


@dataclass
class Material:
    """An immutable piece of source material (a specific role + version)."""

    id: str
    kind: str
    version: int
    source: str
    payload: dict
    diagnostics: list[Diagnostic] = field(default_factory=list)
    seq: int = -1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "version": self.version,
            "source": self.source,
            "payload": self.payload,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        return cls(
            id=d["id"], kind=d["kind"], version=d["version"],
            source=d["source"], payload=d["payload"],
            diagnostics=[Diagnostic.from_dict(x)
                         for x in d.get("diagnostics", [])],
            seq=d.get("seq", -1),
        )


@dataclass
class ParamSet:
    """A versioned set of business rules (integration windows, matching...)."""

    id: str
    kind: str          # integration | scoring
    version: int
    source: str
    params: dict
    seq: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSet":
        return cls(**d)


@dataclass
class Event:
    """Append-only fact: import, analysis milestone or human decision."""

    seq: int
    kind: str
    payload: dict
    actor: str = "reviewer"
    branch: str = "*"          # "*" = globally visible
    scope: str = "global"      # global | branch
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(**d)


@dataclass
class Node:
    """A content-addressed derived result (integration, correction or fit)."""

    id: str
    kind: str
    status: str
    payload: dict
    inputs: dict
    parents: list[str]
    param_versions: dict
    engine_version: str = ENGINE_VERSION
    seq: int = -1
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["diagnostics"] = [x.to_dict() for x in self.diagnostics]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=d["id"], kind=d["kind"], status=d["status"],
            payload=d["payload"], inputs=d["inputs"],
            parents=d["parents"], param_versions=d["param_versions"],
            engine_version=d.get("engine_version", ENGINE_VERSION),
            seq=d.get("seq", -1),
            diagnostics=[Diagnostic.from_dict(x)
                         for x in d.get("diagnostics", [])],
        )


def make_node(kind: str, status: str, payload: dict, inputs: dict,
              parents: list[str], param_versions: dict,
              diagnostics: list[Diagnostic] | None = None) -> Node:
    """Build a node whose id is a hash of everything it depends on."""
    basis = {
        "kind": kind,
        "status_basis": payload,
        "inputs": inputs,
        "parents": sorted(parents),
        "params": param_versions,
        "engine": ENGINE_VERSION,
    }
    node_id = f"node.{kind}.{stable_hash(basis)}"
    return Node(
        id=node_id, kind=kind, status=status, payload=payload, inputs=inputs,
        parents=sorted(parents), param_versions=param_versions,
        diagnostics=diagnostics or [],
    )


INJECTION_EVENTS = {
    "discard_first", "set_endpoint",
}
