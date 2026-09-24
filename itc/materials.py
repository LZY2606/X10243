"""Immutable source materials, each stream versioned independently.

Streams (the "direct sources"):

``power``, ``baseline``, ``blank_power``, ``blank_baseline``,
``injections``, ``concentration``, ``temperature``, ``cell_volume``.

Every material revision is immutable, stores the *original* representation,
and carries a SHA-1 content hash. A revision never mutates; corrections append
a new revision in the same stream chain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic, SemanticImpasse
from .series import SeriesData, parse_series
from .units import Quantity, parse_quantity, require

SERIES_STREAMS = {"power", "baseline", "blank_power", "blank_baseline"}
SCALAR_STREAMS = {
    "concentration": "concentration",       # keyed scalar file
    "temperature": "temperature",
    "cell_volume": "volume",
}
ALL_STREAMS = sorted(SERIES_STREAMS | {"injections"} | set(SCALAR_STREAMS))

STREAM_ORDER = ["power", "injections", "concentration", "temperature",
                "cell_volume", "baseline", "blank_power", "blank_baseline"]


def content_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return "sha1:" + hashlib.sha1(blob).hexdigest()


@dataclass(frozen=True)
class Injection:
    id: str
    row: int
    time: Quantity
    volume: Quantity

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "row": self.row,
                "time": self.time.to_dict(), "volume": self.volume.to_dict()}


@dataclass(frozen=True)
class InjectionSchedule:
    file: str
    injections: tuple[Injection, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file,
                "diagnostics": [d.to_dict() for d in self.diagnostics],
                "injections": [i.to_dict() for i in self.injections]}


def parse_injections(text: str, file: str) -> InjectionSchedule:
    from .series import parse_table
    table = parse_table(text, file)
    diags: list[Diagnostic] = []

    def col(*names: str) -> int:
        for n in names:
            idx = table.column_index(n)
            if idx >= 0:
                return idx
        return -1

    ti = col("time", "t", "inject_time")
    vi = col("volume", "v", "inj_volume", "vol")
    idi = col("injection_id", "id", "shot", "n")
    missing = [n for n, i in (("time", ti), ("volume", vi)) if i < 0]
    if missing:
        raise SemanticImpasse(
            f"{file}: missing column(s) {missing}",
            [Diagnostic("missing_column", "required injection column absent",
                        file, 1, detail={"missing": missing,
                                         "header": table.header})])
    seen_ids: set[str] = set()
    injections: list[Injection] = []
    prev_t: float | None = None
    for i, row in enumerate(table.rows):
        physical_row = i + 2
        auto_id = f"inj-{i + 1:02d}"
        inj_id = row[idi].strip() if idi >= 0 and idi < len(row) and row[idi].strip() else auto_id
        if inj_id in seen_ids:
            raise SemanticImpasse(
                f"{file}: duplicate injection id {inj_id!r}",
                [Diagnostic("duplicate_injection_id", "injection ids must be unique",
                            file, physical_row, detail={"id": inj_id})])
        seen_ids.add(inj_id)
        tq = parse_quantity(row[ti], "time", default_unit="s",
                            file=file, row=physical_row, column="time")
        vq = parse_quantity(row[vi], "volume", default_unit="uL",
                            file=file, row=physical_row, column="volume")
        for q, name in ((tq, "time"), (vq, "volume")):
            if not q.convertible:
                diags.append(q.diagnostic or Diagnostic(
                    "unconvertible_injection", f"injection {name} unconvertible",
                    file, physical_row, name, detail={"raw": q.raw}))
        if tq.convertible:
            if prev_t is not None and require(tq) <= prev_t:
                raise SemanticImpasse(
                    f"{file}: injection times must be strictly increasing",
                    [Diagnostic("injection_order",
                                f"time {tq.raw!r} is not after previous injection",
                                file, physical_row, "time",
                                detail={"previous_s": prev_t})])
            prev_t = require(tq)
        if vq.convertible and require(vq) <= 0:
            raise SemanticImpasse(
                f"{file}: injection volume must be positive",
                [Diagnostic("nonpositive_volume", "volume must be > 0",
                            file, physical_row, "volume", detail={"raw": vq.raw})])
        injections.append(Injection(inj_id, physical_row, tq, vq))
    if not injections:
        raise SemanticImpasse(f"{file}: injection schedule is empty",
                              [Diagnostic("empty_schedule", "no injections", file)])
    return InjectionSchedule(file=file, injections=tuple(injections),
                             diagnostics=tuple(diags))


@dataclass(frozen=True)
class ScalarFile:
    file: str
    stream: str
    entries: tuple[tuple[str, Quantity], ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "stream": self.stream,
                "diagnostics": [d.to_dict() for d in self.diagnostics],
                "entries": [{"key": k, "quantity": q.to_dict()}
                            for k, q in self.entries]}


_CONCENTRATION_KEYS = {
    "syringe": "syringe_concentration",
    "syringe_concentration": "syringe_concentration",
    "ligand": "syringe_concentration",
    "cell": "cell_concentration",
    "cell_concentration": "cell_concentration",
    "macromolecule": "cell_concentration",
}


def parse_scalar(text: str, file: str, stream: str) -> ScalarFile:
    from .series import parse_table
    table = parse_table(text, file)
    ki = table.column_index("key")
    vi = table.column_index("value")
    if ki < 0 or vi < 0:
        raise SemanticImpasse(
            f"{file}: scalar files need 'key' and 'value' columns",
            [Diagnostic("missing_column", "key/value columns required",
                        file, 1, detail={"header": table.header})])
    entries: list[tuple[str, Quantity]] = []
    diags: list[Diagnostic] = []
    seen: set[str] = set()
    for i, row in enumerate(table.rows):
        physical_row = i + 2
        key = row[ki].strip().lower()
        if stream == "concentration":
            canonical_key = _CONCENTRATION_KEYS.get(key)
            if canonical_key is None:
                raise SemanticImpasse(
                    f"{file}: unknown concentration key {key!r}",
                    [Diagnostic("unknown_key",
                                "expected syringe/cell concentration",
                                file, physical_row, "key", detail={"key": key})])
            key = canonical_key
            dim = "concentration"
        elif stream == "temperature":
            if key not in ("temperature", "temp", "t"):
                raise SemanticImpasse(
                    f"{file}: temperature file carries {key!r}",
                    [Diagnostic("unknown_key", "expected 'temperature'",
                                file, physical_row, "key", detail={"key": key})])
            key = "temperature"
            dim = "temperature"
        elif stream == "cell_volume":
            if key not in ("cell_volume", "v0", "volume"):
                raise SemanticImpasse(
                    f"{file}: cell volume file carries {key!r}",
                    [Diagnostic("unknown_key", "expected 'cell_volume'",
                                file, physical_row, "key", detail={"key": key})])
            key = "cell_volume"
            dim = "volume"
        else:  # pragma: no cover - guarded by callers
            raise SemanticImpasse(f"unknown scalar stream {stream!r}")
        if key in seen:
            raise SemanticImpasse(f"{file}: duplicate key {key!r}",
                                  [Diagnostic("duplicate_key", "duplicate scalar key",
                                              file, physical_row, "key")])
        seen.add(key)
        q = parse_quantity(row[vi], dim, file=file, row=physical_row, column="value")
        if not q.convertible:
            diags.append(q.diagnostic or Diagnostic(
                "unconvertible_scalar", "scalar value unconvertible",
                file, physical_row, "value", detail={"raw": row[vi]}))
        entries.append((key, q))
    if not entries:
        raise SemanticImpasse(f"{file}: scalar file is empty",
                              [Diagnostic("empty_scalar", "no entries", file)])
    return ScalarFile(file=file, stream=stream, entries=tuple(entries),
                      diagnostics=tuple(diags))


@dataclass(frozen=True)
class Material:
    id: str
    stream: str
    revision: int
    file: str
    kind: str                       # series | injections | scalar
    payload: dict[str, Any]         # structured, original representations kept
    raw_text: str
    sha1: str
    span: tuple[float, float] | None = None
    diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "stream": self.stream, "revision": self.revision,
               "file": self.file, "kind": self.kind, "sha1": self.sha1,
               "payload": self.payload, "raw_text": self.raw_text,
               "diagnostics": list(self.diagnostics)}
        if self.span is not None:
            out["span"] = list(self.span)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Material":
        span = tuple(d["span"]) if d.get("span") is not None else None
        return cls(id=d["id"], stream=d["stream"], revision=d["revision"],
                   file=d["file"], kind=d["kind"], payload=d["payload"],
                   raw_text=d.get("raw_text", ""), sha1=d["sha1"],
                   span=span, diagnostics=tuple(d.get("diagnostics", [])))


def build_material(stream: str, raw_text: str, file: str | None = None,
                   revision: int = 1) -> Material:
    """Parse one stream's raw text into an immutable material revision."""
    fname = file or f"{stream}.csv"
    if stream in SERIES_STREAMS:
        if stream.startswith("blank_"):
            inner = stream[len("blank_"):]
            value_col = "power" if inner == "power" else "power"
            data = parse_series(raw_text, fname, stream, time_column="t",
                                value_column=value_col, time_unit="s",
                                value_unit="ucal/s")
        else:
            value_col = "power"
            data = parse_series(raw_text, fname, stream, time_column="t",
                                value_column=value_col, time_unit="s",
                                value_unit="ucal/s")
        payload = data.to_dict()
        span = data.span
        diags = tuple(d.to_dict() for d in data.diagnostics)
        kind = "series"
    elif stream == "injections":
        sched = parse_injections(raw_text, fname)
        payload = sched.to_dict()
        span = None
        diags = sched.diagnostics and tuple(d.to_dict() for d in sched.diagnostics)
        kind = "injections"
    elif stream in SCALAR_STREAMS:
        scalar = parse_scalar(raw_text, fname, stream)
        payload = scalar.to_dict()
        span = None
        diags = tuple(d.to_dict() for d in scalar.diagnostics)
        kind = "scalar"
    else:
        raise SemanticImpasse(
            f"unknown material stream {stream!r}",
            [Diagnostic("unknown_stream", "stream not in the versioned set",
                        fname, detail={"stream": stream, "known": ALL_STREAMS})])
    sha = content_hash({"stream": stream, "revision": revision,
                        "file": fname, "payload": payload})
    mid = f"mat:{stream}:{revision:04d}"
    return Material(id=mid, stream=stream, revision=revision, file=fname,
                    kind=kind, payload=payload, raw_text=raw_text, sha1=sha,
                    span=span, diagnostics=diags)
