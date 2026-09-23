"""Immutable, separately versioned source materials.

Every material class is an append-only version.  Seven independent streams are
kept apart (never merged into one generic table):

    power          power samples (time series)
    injections     injection volumes and times
    concentration  syringe / cell concentrations
    temperature    experimental temperature
    cell_volume    cell (pool) volume
    baseline       baseline series
    blank          blank (dilution) power series

Each scalar value carries its raw representation and unit; series values carry
a coordinate base (time origin + unit) and the column unit string.  Values that
cannot be converted remain in place with diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import Diagnostic, FormatCorrupt, SemanticUnsolvable
from .identity import content_hash
from .units import parse_quantity

MATERIAL_KINDS = (
    "power", "injections", "concentration",
    "temperature", "cell_volume", "baseline", "blank",
)


@dataclass
class SamplePoint:
    t_raw: str
    p_raw: str
    t_s: float | None
    p_ucal_s: float | None
    flags: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Series:
    """A time-indexed power or baseline series with its coordinate base."""
    time_unit_raw: str
    power_unit_raw: str
    time_origin: str           # documented reference, e.g. "first_injection"
    points: list[SamplePoint]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_unit_raw": self.time_unit_raw,
            "power_unit_raw": self.power_unit_raw,
            "time_origin": self.time_origin,
            "points": [p.to_dict() for p in self.points],
            "diagnostics": self.diagnostics,
            "gaps": self.gaps,
        }


@dataclass
class Injection:
    index: int
    time_raw: str
    volume_raw: str
    time_s: float | None
    volume_uL: float | None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scalar:
    raw: str
    unit_raw: str
    value: float | None
    canonical_unit: str | None
    dimension: str | None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConcentrationSet:
    syringe: Scalar
    cell: Scalar

    def to_dict(self) -> dict[str, Any]:
        return {"syringe": self.syringe.to_dict(), "cell": self.cell.to_dict()}


@dataclass
class MaterialVersion:
    kind: str
    version_id: str
    label: str
    source: dict[str, Any]        # provenance: format / line map / checksum
    body: dict[str, Any]
    supersedes: str | None
    content_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _diagnostic(severity: str, code: str, message: str,
                location: dict[str, Any], raw: Any = None) -> dict[str, Any]:
    return Diagnostic(severity, code, message, location, raw).to_dict()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _require_unit_header(header: str, line: int, what: str) -> str:
    if "[" not in header or "]" not in header:
        raise FormatCorrupt(
            f"{what} header must declare a unit in square brackets",
            locations=[{"line": line, "column": 1, "section": what}],
            diagnostics=[{"header": header}])
    return header[header.index("[") + 1:header.rindex("]")].strip()


def _parse_series(rows: list[str], start_line: int, section: str,
                  time_unit: str, power_unit: str, origin: str) -> Series:
    points: list[SamplePoint] = []
    diagnostics: list[dict[str, Any]] = []
    prev_t: float | None = None
    median_dt: float | None = None
    dts: list[float] = []

    for offset, row in enumerate(rows):
        line = start_line + offset
        if not row.strip():
            diagnostics.append(_diagnostic(
                "warning", "blank_row", "Empty row ignored.",
                {"line": line, "section": section}))
            continue
        if row.strip().startswith("inject:"):
            # Blank runs carry their own injection schedule inside the series.
            continue
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 2:
            raise FormatCorrupt(
                f"{section} rows must contain exactly two columns",
                locations=[{"line": line, "column": 3 if len(parts) > 2 else 1,
                            "section": section}],
                diagnostics=[{"columns": len(parts), "row": row}])
        tq = parse_quantity(f"{parts[0]} {time_unit}", expected_dimension="time",
                            location={"line": line, "column": 1, "section": section})
        pq = parse_quantity(f"{parts[1]} {power_unit}", expected_dimension="power",
                            location={"line": line, "column": 2, "section": section})
        point = SamplePoint(
            parts[0], parts[1], tq.value, pq.value,
            diagnostics=[d.to_dict() for d in tq.diagnostics + pq.diagnostics
                         if d.severity == "error"])
        if not point.diagnostics:
            assert tq.value is not None
            if prev_t is not None:
                dts.append(tq.value - prev_t)
                if tq.value <= prev_t:
                    point.flags.append("non_monotonic_time")
                    point.diagnostics.append(_diagnostic(
                        "error", "non_monotonic_time",
                        "Sample time is not strictly greater than the previous one.",
                        {"line": line, "section": section}, parts[0]))
            prev_t = tq.value
        points.append(point)

    gaps: list[dict[str, float]] = []
    good = [p for p in points if p.t_s is not None and p.p_ucal_s is not None
            and "non_monotonic_time" not in p.flags]
    if len(dts) >= 2:
        ordered = sorted(dts)
        median_dt = ordered[len(ordered) // 2]
        for i in range(1, len(good)):
            dt = good[i].t_s - good[i - 1].t_s  # type: ignore[operator]
            if median_dt > 0 and dt > 2.5 * median_dt:
                gaps.append({"start_s": good[i - 1].t_s,  # type: ignore[dict-item]
                             "end_s": good[i].t_s,        # type: ignore[dict-item]
                             "missing_dt_s": dt - median_dt})

    return Series(time_unit, power_unit, origin, points, diagnostics, gaps)


def _parse_injections(rows: list[str], start_line: int,
                      time_unit: str, volume_unit: str) -> list[Injection]:
    injections: list[Injection] = []
    for offset, row in enumerate(rows):
        line = start_line + offset
        if not row.strip():
            continue
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 2:
            raise FormatCorrupt(
                "injection rows must contain exactly two columns (time, volume)",
                locations=[{"line": line, "section": "injections"}],
                diagnostics=[{"columns": len(parts), "row": row}])
        tq = parse_quantity(f"{parts[0]} {time_unit}", expected_dimension="time",
                            location={"line": line, "column": 1, "section": "injections"})
        vq = parse_quantity(f"{parts[1]} {volume_unit}", expected_dimension="volume",
                            location={"line": line, "column": 2, "section": "injections"})
        injections.append(Injection(
            len(injections) + 1, parts[0], parts[1], tq.value, vq.value,
            [d.to_dict() for d in tq.diagnostics + vq.diagnostics
             if d.severity == "error"]))
    return injections


def _parse_scalar(raw: Any, unit: str, dimension: str, where: str) -> Scalar:
    q = parse_quantity(f"{raw} {unit}", expected_dimension=dimension,
                       location={"section": where})
    return Scalar(str(raw), unit, q.value, q.unit_canonical, q.dimension,
                  [d.to_dict() for d in q.diagnostics if d.severity == "error"])


# Section-based structured text. JSON manifests are handled in io_bundle.
def parse_sectioned_text(text: str) -> dict[str, Any]:
    """Parse the lab interchange text format into raw material fragments.

    Grammar::

        @section power [time_unit] [power_unit] origin=<...>
        <csv rows>
        @section injections [time_unit] [volume_unit]
        @section concentration [unit]   rows: syringe=<...>, cell=<...>
        @section temperature [unit]     single row
        @section cell_volume [unit]     single row
        @section baseline / blank       same shape as power
    """
    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None
    current_rows: list[str] = []
    current_line = 0
    header_line: dict[str, int] = {}

    def flush() -> None:
        if current is not None:
            sections[current]["rows"] = list(current_rows)

    lines = text.splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if current is not None:
                current_rows.append(raw_line)
            continue
        if line.startswith("@"):
            flush()
            tokens = line[1:].split()
            name = tokens[0].lower()
            if name not in ("power", "injections", "concentration", "temperature",
                            "cell_volume", "baseline", "blank"):
                raise FormatCorrupt(
                    f"Unknown section {name!r}",
                    locations=[{"line": lineno, "column": 1}],
                    diagnostics=[{"section": name, "known": MATERIAL_KINDS}])
            if name in sections:
                raise FormatCorrupt(
                    f"Duplicate section {name!r}",
                    locations=[{"line": lineno}],
                    diagnostics=[{"first_line": header_line.get(name)}])
            current = name
            header_line[name] = lineno
            sections[name] = {"header": tokens[1:], "rows": [], "line": lineno}
            current_rows = []
            current_line = lineno
        else:
            if current is None:
                raise FormatCorrupt(
                    "Data row appears before any @section header",
                    locations=[{"line": lineno, "column": 1}],
                    diagnostics=[{"row": line}])
            current_rows.append(raw_line)
    flush()

    materials: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []

    for name, block in sections.items():
        header = block["header"]
        rows = [r for r in block["rows"] if r.strip() and not r.strip().startswith("#")]
        line0 = block["line"] + 1
        if name in ("power", "baseline", "blank"):
            if len(header) < 2:
                raise FormatCorrupt(
                    f"@{name} needs [time unit] and [power unit] in header",
                    locations=[{"line": block["line"]}])
            origin = "experiment_start"
            extra = header[2:]
            for item in extra:
                if item.startswith("origin="):
                    origin = item.split("=", 1)[1]
            series = _parse_series(rows, line0, name, header[0], header[1], origin)
            body = series.to_dict()
            if name == "blank":
                blank_injections = []
                for offset, raw in enumerate(block["rows"]):
                    text = raw.strip()
                    if text.startswith("inject:"):
                        payload = text[len("inject:"):].strip()
                        bits = [b.strip() for b in payload.split(",")]
                        if len(bits) != 2:
                            raise FormatCorrupt(
                                "blank inject: directive needs time,volume",
                                locations=[{"line": line0 + offset,
                                            "section": "blank"}],
                                diagnostics=[{"raw": raw}])
                        tq = parse_quantity(f"{bits[0]} {header[0]}",
                                            expected_dimension="time",
                                            location={"line": line0 + offset,
                                                      "section": "blank.inject"})
                        vq = parse_quantity(f"{bits[1]} {header[2] if len(header) > 2 else ''}",
                                            expected_dimension="volume",
                                            location={"line": line0 + offset,
                                                      "section": "blank.inject"})
                        blank_injections.append({
                            "index": len(blank_injections) + 1,
                            "time_raw": bits[0], "volume_raw": bits[1],
                            "time_s": tq.value, "volume_uL": vq.value,
                            "diagnostics": [d.to_dict() for d in
                                            tq.diagnostics + vq.diagnostics
                                            if d.severity == "error"]})
                body["blank_injections"] = blank_injections
            materials[name] = body
        elif name == "injections":
            if len(header) < 2:
                raise FormatCorrupt(
                    "@injections needs [time unit] and [volume unit] in header",
                    locations=[{"line": block["line"]}])
            injections = _parse_injections(rows, line0, header[0], header[1])
            materials[name] = {"time_unit_raw": header[0],
                               "volume_unit_raw": header[1],
                               "injections": [i.to_dict() for i in injections]}
        elif name == "concentration":
            unit = _require_unit_header("@".join(header), block["line"], "concentration") \
                if header else ""
            if not header:
                raise FormatCorrupt("@concentration needs a [unit] header",
                                    locations=[{"line": block["line"]}])
            values = {}
            for offset, row in enumerate(rows):
                if "=" not in row:
                    raise FormatCorrupt(
                        "concentration rows must be key=value",
                        locations=[{"line": line0 + offset}],
                        diagnostics=[{"row": row}])
                key, _, value = row.partition("=")
                values[key.strip()] = (value.strip(), line0 + offset)
            if "syringe" not in values or "cell" not in values:
                raise SemanticUnsolvable(
                    "concentration needs both syringe and cell values",
                    locations=[{"section": "concentration"}],
                    diagnostics=[{"found": sorted(values)}])
            materials[name] = {
                "unit_raw": unit,
                "syringe": _parse_scalar(values["syringe"][0], unit,
                                         "concentration", "concentration.syringe").to_dict(),
                "cell": _parse_scalar(values["cell"][0], unit,
                                      "concentration", "concentration.cell").to_dict(),
            }
        elif name in ("temperature", "cell_volume"):
            if not header:
                raise FormatCorrupt(f"@{name} needs a [unit] header",
                                    locations=[{"line": block["line"]}])
            if len(rows) != 1:
                raise SemanticUnsolvable(
                    f"@{name} must contain exactly one value, found {len(rows)}",
                    locations=[{"section": name}],
                    diagnostics=[{"rows": len(rows)}])
            dimension = "temperature" if name == "temperature" else "volume"
            scalar = _parse_scalar(rows[0], header[0], dimension, name)
            materials[name] = scalar.to_dict()

    missing = [k for k in ("power", "injections", "concentration",
                           "temperature", "cell_volume") if k not in materials]
    if missing:
        raise SemanticUnsolvable(
            "The interchange document is incomplete.",
            locations=[{"document": True}],
            diagnostics=[{"missing_sections": missing}])

    return {"materials": materials, "diagnostics": diagnostics}
