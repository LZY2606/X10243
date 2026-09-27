"""Versioned, immutable *materials* (the evidence layer).

Material kinds, each versioned independently:

  power_series     raw sample power vs time (+ declared gap table)
  injection_schedule injection volumes and times
  concentrations   syringe / cell concentration (and per-injection revisions)
  conditions       temperature, cell volume (experimental conditions)
  baseline         baseline model/series applied to a power series
  blank_series     blank/control power curve (kept separate, not merged)

A material record is content-addressed (ids.content_id). Revisions create a NEW
record with a `replaces` link; nothing is ever mutated in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .ids import content_id

KINDS = ("power_series", "injection_schedule", "concentrations",
         "conditions", "baseline", "blank_series")


@dataclass(frozen=True)
class Material:
    kind: str
    mid: str
    role: str                 # e.g. "sample" / "blank"
    version: int
    replaces: Optional[str]
    raw: dict[str, Any]       # verbatim source fragment
    data: dict[str, Any]      # normalized view (values may stay None)
    diagnostics: tuple = ()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "mid": self.mid, "role": self.role,
                "version": self.version, "replaces": self.replaces,
                "raw": self.raw, "data": self.data,
                "diagnostics": [d.to_dict() for d in self.diagnostics]}

    @staticmethod
    def from_dict(d: dict) -> "Material":
        from .diagnostics import unpack
        return Material(kind=d["kind"], mid=d["mid"], role=d.get("role", ""),
                        version=d.get("version", 1),
                        replaces=d.get("replaces"), raw=d["raw"],
                        data=d["data"],
                        diagnostics=tuple(unpack(d.get("diagnostics", []))))


def make_material(kind: str, raw: dict[str, Any], data: dict[str, Any],
                  diagnostics=(), role: str = "sample",
                  version: int = 1, replaces: Optional[str] = None) -> Material:
    mid = content_id("mat", {"k": kind, "r": role, "v": version,
                             "raw": raw, "repl": replaces})
    return Material(kind=kind, mid=mid, role=role, version=version,
                    replaces=replaces, raw=raw, data=data,
                    diagnostics=tuple(diagnostics))


def latest(materials: list[Material], kind: str,
           role: str = "sample") -> Optional[Material]:
    candidates = [m for m in materials if m.kind == kind and m.role == role]
    if not candidates:
        return None
    return max(candidates, key=lambda m: (m.version, m.mid))


def lineage(materials: list[Material], mid: str) -> list[Material]:
    """Follow replaces-links from the given material back to its origin."""
    by_id = {m.mid: m for m in materials}
    out: list[Material] = []
    cur = by_id.get(mid)
    seen: set[str] = set()
    while cur is not None and cur.mid not in seen:
        seen.add(cur.mid)
        out.append(cur)
        cur = by_id.get(cur.replaces) if cur.replaces else None
    return list(reversed(out))
