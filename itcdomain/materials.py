"""Immutable, separately-versioned material value objects.

The immutable layer is split by direct source so that revising one kind of
material never silently rewrites another:

* ``power``      - raw sampled power curve (time + power coordinate series)
* ``injections`` - injection times and volumes
* ``concentration`` - syringe / cell analyte concentrations
* ``temperature``   - bath temperature
* ``cell_volume``   - calorimeter cell volume
* ``baseline``      - baseline curve (its own versioned material)
* ``blank``         - blank (dilution-only) run, curve + injections

Every version carries: its raw parsed payload (quantities remember raw
tokens), a content digest, the direct-source label and ordinal.  The same
content imported twice resolves to the same id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json, digest

POWER = "power"
INJECTIONS = "injections"
CONCENTRATION = "concentration"
TEMPERATURE = "temperature"
CELL_VOLUME = "cell_volume"
BASELINE = "baseline"
BLANK = "blank"

MATERIAL_KINDS = (POWER, INJECTIONS, CONCENTRATION, TEMPERATURE, CELL_VOLUME, BASELINE, BLANK)


@dataclass(frozen=True)
class Series:
    """An (x, y) coordinate series with explicit coordinate bases."""

    points: tuple[tuple[str, str, str | None], ...]
    # each point: (raw_x_token, raw_y_token, optional per-point flag/comment)
    x_unit: str | None
    y_unit: str | None
    x_basis: str
    y_basis: str

    def __len__(self) -> int:
        return len(self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [list(p) for p in self.points],
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "x_basis": self.x_basis,
            "y_basis": self.y_basis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Series":
        return cls(
            points=tuple(tuple(p) for p in data["points"]),  # type: ignore[arg-type]
            x_unit=data.get("x_unit"),
            y_unit=data.get("y_unit"),
            x_basis=data["x_basis"],
            y_basis=data["y_basis"],
        )


@dataclass(frozen=True)
class InjectionRecord:
    """One scheduled injection: raw tokens for time and volume, plus ordinal."""

    ordinal: int
    time_raw: str
    time_unit: str | None
    volume_raw: str
    volume_unit: str | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "time_raw": self.time_raw,
            "time_unit": self.time_unit,
            "volume_raw": self.volume_raw,
            "volume_unit": self.volume_unit,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InjectionRecord":
        return cls(
            ordinal=int(data["ordinal"]),
            time_raw=str(data["time_raw"]),
            time_unit=data.get("time_unit"),
            volume_raw=str(data["volume_raw"]),
            volume_unit=data.get("volume_unit"),
            note=data.get("note"),
        )


@dataclass(frozen=True)
class MaterialVersion:
    """One immutable version of one direct-source material."""

    material_id: str
    kind: str
    ordinal: int
    run_id: str
    source_ref: str
    payload: dict[str, Any]
    content_digest: str

    @staticmethod
    def build_id(kind: str, run_id: str, ordinal: int) -> str:
        return f"mat_{kind}_{run_id}_v{ordinal:02d}"

    @classmethod
    def create(
        cls,
        kind: str,
        run_id: str,
        ordinal: int,
        payload: dict[str, Any],
        *,
        source_ref: str,
    ) -> "MaterialVersion":
        if kind not in MATERIAL_KINDS:
            raise ValueError(f"unknown material kind {kind!r}")
        content = digest({"kind": kind, "run_id": run_id, "payload": payload})
        return cls(
            material_id=cls.build_id(kind, run_id, ordinal),
            kind=kind,
            ordinal=ordinal,
            run_id=run_id,
            source_ref=source_ref,
            payload=payload,
            content_digest=content,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "kind": self.kind,
            "ordinal": self.ordinal,
            "run_id": self.run_id,
            "source_ref": self.source_ref,
            "payload": self.payload,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialVersion":
        return cls(
            material_id=data["material_id"],
            kind=data["kind"],
            ordinal=int(data["ordinal"]),
            run_id=data["run_id"],
            source_ref=data["source_ref"],
            payload=data["payload"],
            content_digest=data["content_digest"],
        )


def series_payload(series: Series, *, source_ref: str) -> dict[str, Any]:
    return {"series": series.to_dict(), "source_ref": source_ref}


def injections_payload(records: list[InjectionRecord], *, source_ref: str) -> dict[str, Any]:
    return {
        "injections": [r.to_dict() for r in records],
        "source_ref": source_ref,
    }


def scalar_payload(quantity: dict[str, Any], *, source_ref: str) -> dict[str, Any]:
    return {"quantity": quantity, "source_ref": source_ref}


def concentration_payload(
    syringe: dict[str, Any],
    cell: dict[str, Any] | None,
    *,
    source_ref: str,
) -> dict[str, Any]:
    return {
        "syringe_concentration": syringe,
        "cell_concentration": cell,
        "source_ref": source_ref,
    }


def material_semantic_key(material: MaterialVersion) -> str:
    """Content-only key: same bytes -> same key regardless of ordinal/id."""
    return canonical_json({"kind": material.kind, "payload": material.payload})
