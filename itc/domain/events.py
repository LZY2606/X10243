"""Append-only event log + deterministic content-addressed identity.

Immutable materials and mutable decisions are separated into distinct event
types. Every derived computation records an input digest, parameter versions
and parent dependency ids. Equivalent input yields equivalent ids (content
hashing), and the (branch, seq) ordering is stable across export/import.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

# material (immutable source) event types
MATERIAL_COMMITTED = "MATERIAL_COMMITTED"
PARAMS_COMMITTED = "PARAMS_COMMITTED"
BRANCH_CREATED = "BRANCH_CREATED"

# mutable decision event types
DECISION_FIRST_INJECTION = "DECISION_FIRST_INJECTION"
DECISION_BLANK_RULE = "DECISION_BLANK_RULE"
DECISION_MODEL_PREFERENCE = "DECISION_MODEL_PREFERENCE"
DECISION_INTEGRATION_PARAMS = "DECISION_INTEGRATION_PARAMS"

# derived computation event types
INTEGRATION_COMPUTED = "INTEGRATION_COMPUTED"
BLANK_INTEGRATION_COMPUTED = "BLANK_INTEGRATION_COMPUTED"
NET_HEATS_COMPUTED = "NET_HEATS_COMPUTED"
MODELS_FITTED = "MODELS_FITTED"

# review event types
REVIEW_ACCEPT = "REVIEW_ACCEPT"
REVIEW_REJECT = "REVIEW_REJECT"
REVIEW_ADJUST = "REVIEW_ADJUST"

DERIVED_TYPES = {INTEGRATION_COMPUTED, BLANK_INTEGRATION_COMPUTED,
                 NET_HEATS_COMPUTED, MODELS_FITTED}
DECISION_TYPES = {DECISION_FIRST_INJECTION, DECISION_BLANK_RULE,
                  DECISION_MODEL_PREFERENCE, DECISION_INTEGRATION_PARAMS}


def digest(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      default=str).encode("utf-8")
    return hashlib.sh256(blob).hexdigest() if False else \
        hashlib.sha256(blob).hexdigest()


def short(obj) -> str:
    return digest(obj)[:10]


@dataclass
class Event:
    event_id: str
    seq: int
    branch: str
    parent_event_id: str | None
    type: str
    payload: dict
    content_hash: str
    inputs_digest: str | None = None
    parent_dependencies: list[str] = field(default_factory=list)
    author: str = "reviewer"
    note: str = ""
    epoch_us: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(**{k: d.get(k) for k in Event.__dataclass_fields__})


class IdFactory:
    """Deterministic ids so equivalent input => equivalent identity."""

    @staticmethod
    def material_id(kind: str, role: str, content: str) -> str:
        return f"mat-{kind}-{role}-{short(content)}"

    @staticmethod
    def event_id(branch: str, seq: int, type_: str, payload_hash: str) -> str:
        return f"ev-{branch}-{seq:04d}-{type_}-{payload_hash[:8]}"

    @staticmethod
    def integral_id(power_id: str, params_version: str) -> str:
        return f"intg-{short([power_id, params_version])}"

    @staticmethod
    def blank_id(blank_series_id: str, params_version: str) -> str:
        return f"bint-{short([blank_series_id, params_version])}"

    @staticmethod
    def net_id(sample_int_id: str, blank_params_version: str,
               blank_int_id: str | None) -> str:
        return f"net-{short([sample_int_id, blank_params_version, blank_int_id])}"

    @staticmethod
    def fit_id(model: str, net_id: str, conc_version: str,
               param_version: str) -> str:
        return f"fit-{model}-{short([net_id, conc_version, param_version])}"
