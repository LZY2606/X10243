"""Immutable, independently versioned lab materials.

Each kind is versioned on its own so a concentration revision never rewrites a
power trace and a local curve patch does not invalidate injection metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

MATERIAL_KINDS = (
    "power",          # sample-cell power sampling vs experiment time
    "schedule",       # injection volume + time per shot
    "concentration",  # syringe/cell concentration revision stream
    "temperature",    # bath temperature
    "cell_volume",    # cell volume
    "baseline",       # baseline curve, separately versioned from power
    "blank",          # blank (dilution) curve, separately versioned
)


@dataclass(frozen=True)
class Series:
    """Sampled curve on an explicit time basis.

    samples stores [time_s, value_canonical, raw_time_text, raw_value_text,
    valid]. Unconvertible points keep their raw text with valid=False instead
    of being dropped.
    """

    points: list[list] = field(default_factory=list)
    value_dimension: str = "power"
    value_unit: str = "W"
    time_unit: str = "s"
    origin: str = ""
    time_basis: str = "experiment_seconds"

    def arrays(self, *, valid_only: bool = True):
        import math

        import numpy as np

        rows = [p for p in self.points if (p[4] or not valid_only)]
        t = np.array([p[0] for p in rows], dtype=float)
        v = np.array([p[1] for p in rows], dtype=float)
        if valid_only:
            mask = np.isfinite(t) & np.isfinite(v)
            t, v = t[mask], v[mask]
        order = np.argsort(t)
        return t[order], v[order]

    def all_rows(self):
        return [list(p) for p in self.points]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Series":
        allowed = {f for f in Series.__dataclass_fields__}
        return Series(**{k: v for k, v in d.items() if k in allowed})


@dataclass(frozen=True)
class Injection:
    index: int
    time_s: float
    volume_l: float
    duration_s: float
    raw_time: str
    raw_volume: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Injection":
        return Injection(**d)


@dataclass(frozen=True)
class MaterialVersion:
    kind: str
    version: int
    origin: str
    # kind-specific payload:
    series: Series | None = None
    injections: tuple[Injection, ...] = ()
    quantity: float | None = None
    unit: str | None = None
    raw: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "origin": self.origin,
            "series": self.series.to_dict() if self.series else None,
            "injections": [i.to_dict() for i in self.injections],
            "quantity": self.quantity,
            "unit": self.unit,
            "raw": self.raw,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(d: dict) -> "MaterialVersion":
        return MaterialVersion(
            kind=d["kind"],
            version=d["version"],
            origin=d["origin"],
            series=Series.from_dict(d["series"]) if d.get("series") else None,
            injections=tuple(Injection.from_dict(i) for i in d.get("injections", [])),
            quantity=d.get("quantity"),
            unit=d.get("unit"),
            raw=d.get("raw"),
            extra=d.get("extra", {}),
        )
