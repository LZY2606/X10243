from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MaterialRef:
    kind: str
    material_id: str
    version: int


@dataclass
class Event:
    event_id: str
    branch: str
    seq: int
    kind: str
    timestamp: str
    actor: str
    payload: dict[str, Any]
    parents: list[str] = field(default_factory=list)
    input_summary: dict[str, Any] = field(default_factory=dict)
    rules_version: str = ""
    result_ids: list[str] = field(default_factory=list)


@dataclass
class Result:
    result_id: str
    kind: str
    branch: str
    event_id: str
    status: str
    value: dict[str, Any]
    parents: list[str]
    input_summary: dict[str, Any]
    rules_version: str
