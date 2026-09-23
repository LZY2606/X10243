"""Immutable material / result domain objects.

Materials are the seven separately-versioned *immutable* sources:
power samples, injections (volume+time), concentrations, temperature,
cell volume, baseline and blank curves.  Results (integrations, fits) are
also immutable: they record input summaries, parameter versions and parent
dependencies, so old conclusions are never overwritten by newer runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Material kinds -- each is independently versioned.
POWER = "power"
INJECTIONS = "injections"
CONCENTRATION = "concentration"
TEMPERATURE = "temperature"
CELL_VOLUME = "cell_volume"
BASELINE = "baseline"
BLANK = "blank"

MATERIAL_KINDS = (
    POWER, INJECTIONS, CONCENTRATION, TEMPERATURE, CELL_VOLUME, BASELINE, BLANK,
)

# Result kinds.
INTEGRATION_BATCH = "integration_batch"
FIT = "fit"


@dataclass
class Sample:
    index: int
    raw_time: str
    raw_value: str
    time_si: float | None                 # seconds
    value_si: float | None                # watts (power)
    time_unit: str
    value_unit: str
    convertible: bool = True
    diagnostic: dict[str, Any] | None = None
    is_gap: bool = False                  # explicit missing datum ("NA"/"--")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Injection:
    index: int                            # 1-based
    time_si: float | None                 # seconds from time origin
    volume_si: float | None               # litres
    raw_time: str
    raw_volume: str
    time_unit: str
    volume_unit: str
    convertible: bool = True
    diagnostic: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """A power/baseline/blank trace with a shared time basis."""
    samples: list[Sample]
    origin: str = "t0"                    # coordinate/time reference label
    gaps: list[tuple[float | None, float | None]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    valid_range: tuple[float | None, float | None] = (None, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "origin": self.origin,
            "gaps": [list(g) for g in self.gaps],
            "diagnostics": list(self.diagnostics),
            "valid_range": list(self.valid_range),
        }


@dataclass
class Material:
    kind: str
    role: str                             # "primary" | "blank"
    content: str                          # verbatim original text/bytes-as-text
    file_name: str
    media_type: str
    content_sha256: str
    parsed: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrationPoint:
    injection_index: int
    start_si: float
    end_si: float
    heat_si: float | None                 # joules, after dilution when applied
    raw_heat_si: float                    # joules, before dilution subtraction
    blank_heat_si: float | None = None
    dilution_rule: str | None = None      # correspondence rule statement/version
    baseline_material: str | None = None
    n_samples: int = 0
    status: str = "ok"                    # ok | warning | rejected
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FitCandidate:
    model: str                            # one_site | two_site | dilution
    aic: float
    aicc: float
    n_params: int
    params: dict[str, float]
    param_units: dict[str, str]
    predicted: list[float]
    residuals: list[float]
    tied: bool = False
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
