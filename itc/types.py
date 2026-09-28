"""Core domain datatypes for the ITC review application.

Immutable materials keep their *raw representation*. Parsed objects carry the
canonical (normalised) value plus the original token and unit, so nothing is
silently rewritten on import. Unconvertible cells stay in place with a
diagnostic instead of becoming a fake default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class Diagnostic:
    code: str                       # e.g. PARSE_UNIT, SEMANTIC_GAP_CROSSED
    severity: str                   # "error" | "warning"
    message: str
    location: str = ""              # e.g. "power.csv:12" or "injection[3].end_s"
    value: Optional[str] = None     # raw value that caused the diagnostic

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Diagnostic":
        return Diagnostic(**d)


@dataclass(frozen=True)
class Sample:
    t_s: float
    p_w: float
    raw_t: str
    raw_p: str
    line: int


@dataclass
class PowerSeries:
    kind: str                       # "sample" | "blank"
    unit_time: str                  # declared unit token, e.g. "s"
    unit_power: str                 # declared unit token, e.g. "uW"
    samples: list[Sample] = field(default_factory=list)
    gap_starts_s: list[float] = field(default_factory=list)
    gap_ends_s: list[float] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def times(self) -> list[float]:
        return [s.t_s for s in self.samples]

    def powers(self) -> list[float]:
        return [s.p_w for s in self.samples]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "unit_time": self.unit_time,
            "unit_power": self.unit_power,
            "samples": [asdict(s) for s in self.samples],
            "gap_starts_s": self.gap_starts_s,
            "gap_ends_s": self.gap_ends_s,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PowerSeries":
        return PowerSeries(
            kind=d["kind"],
            unit_time=d["unit_time"],
            unit_power=d["unit_power"],
            samples=[Sample(**s) for s in d.get("samples", [])],
            gap_starts_s=list(d.get("gap_starts_s", [])),
            gap_ends_s=list(d.get("gap_ends_s", [])),
            diagnostics=[Diagnostic.from_dict(x) for x in d.get("diagnostics", [])],
        )


@dataclass
class Injection:
    index: int                      # 1-based ordinal as given in the file
    time_s: float                   # canonical seconds
    volume_l: float                 # canonical litres
    raw_time: str
    raw_volume: str
    line: int
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Injection":
        return Injection(
            index=d["index"],
            time_s=d["time_s"],
            volume_l=d["volume_l"],
            raw_time=d["raw_time"],
            raw_volume=d["raw_volume"],
            line=d["line"],
            diagnostics=[Diagnostic.from_dict(x) for x in d.get("diagnostics", [])],
        )


@dataclass
class InjectionSchedule:
    unit_time: str
    unit_volume: str
    injections: list[Injection] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_time": self.unit_time,
            "unit_volume": self.unit_volume,
            "injections": [i.to_dict() for i in self.injections],
            diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "InjectionSchedule":
        return InjectionSchedule(
            unit_time=d["unit_time"],
            unit_volume=d["unit_volume"],
            injections=[Injection.from_dict(i) for i in d.get("injections", [])],
            diagnostics=[Diagnostic.from_dict(x) for x in d.get("diagnostics", [])],
        )


@dataclass
class Scalar:
    """A scalar material (concentration / temperature / cell volume / ...).

    Keeps both the canonical numeric value (or None when unconvertible) and
    the exact raw representation.
    """
    kind: str
    value: Optional[float]          # canonical value, None if unconvertible
    canonical_unit: Optional[str]
    raw: str
    declared_unit: Optional[str]
    diagnostic: Optional[Diagnostic] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scalar":
        diag = d.get("diagnostic")
        return Scalar(
            kind=d["kind"],
            value=d.get("value"),
            canonical_unit=d.get("canonical_unit"),
            raw=d["raw"],
            declared_unit=d.get("declared_unit"),
            diagnostic=Diagnostic.from_dict(diag) if diag else None,
        )


@dataclass
class Baseline:
    kind: str = "baseline_linear"   # "baseline_linear" | "baseline_constant"
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "diagnostics": [d.to_dict() for d in self.diagnostics]}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Baseline":
        return Baseline(
            kind=d.get("kind", "baseline_linear"),
            diagnostics=[Diagnostic.from_dict(x) for x in d.get("diagnostics", [])],
        )


@dataclass
class SessionBundle:
    """All materials of one import, as parsed (before normalisation-driven calc)."""
    power: Optional[PowerSeries] = None
    blank: Optional[PowerSeries] = None
    schedule: Optional[InjectionSchedule] = None
    baseline: Optional[Baseline] = None
    scalars: dict[str, Scalar] = field(default_factory=dict)
    raw_files: dict[str, str] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def diagnostics_all(self) -> list[Diagnostic]:
        out: list[Diagnostic] = list(self.diagnostics)
        if self.power:
            out += self.power.diagnostics
        if self.blank:
            out += self.blank.diagnostics
        if self.schedule:
            out += self.schedule.diagnostics
            for inj in self.schedule.injections:
                out += inj.diagnostics
        for s in self.scalars.values():
            if s.diagnostic:
                out.append(s.diagnostic)
        if self.baseline:
            out += self.baseline.diagnostics
        return out
