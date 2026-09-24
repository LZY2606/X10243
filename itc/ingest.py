"""Raw material ingestion.

Materials are the *immutable* layer.  Seven independently versioned material
kinds cover every source datum the specification calls out:

``power_curve``  sampled power vs time (may contain gaps / unreadable cells)
``injections``   injection volumes and times
``concentration`` syringe and/or cell concentration
``temperature``  bath / cell temperature
``cell_volume``  sample-cell volume
``baseline``     baseline curve, versioned separately from sample power
``blank_curve``  blank (dilution) control power curve, versioned separately

Ingestion preserves the original textual representation.  Normalization to
canonical units happens on the parsed material, never by overwriting raw text.
Unconvertible cells are retained with a located diagnostic; gap cells are
explicit ``{"missing": true}`` samples and adjacent-time gaps are detected.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    Diagnostic,
    FormatCorrupted,
    SemanticImpossible,
    diag,
)
from .units import CANONICAL, Quantity, TimeBasis, parse_quantity

MATERIAL_KINDS = (
    "power_curve",
    "injections",
    "concentration",
    "temperature",
    "cell_volume",
    "baseline",
    "blank_curve",
)

# Filename hint -> kind (used by the offline case pack and by upload guessing).
FILENAME_HINTS: dict[str, str] = {
    "power": "power_curve",
    "powercurve": "power_curve",
    "trace": "power_curve",
    "injection": "injections",
    "injections": "injections",
    "schedule": "injections",
    "concentration": "concentration",
    "concentrations": "concentration",
    "temperature": "temperature",
    "temp": "temperature",
    "cell": "cell_volume",
    "cellvolume": "cell_volume",
    "baseline": "baseline",
    "blank": "blank_curve",
    "blankcurve": "blank_curve",
}


@dataclass
class Sample:
    index: int
    t: Quantity
    y: Quantity | None
    raw_t: str
    raw_y: str
    missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "t": self.t.to_dict(),
            "y": self.y.to_dict() if self.y is not None else None,
            "raw_t": self.raw_t,
            "raw_y": self.raw_y,
            "missing": self.missing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        return cls(
            index=data["index"],
            t=Quantity.from_dict(data["t"]),
            y=Quantity.from_dict(data["y"]) if data.get("y") else None,
            raw_t=data.get("raw_t", ""),
            raw_y=data.get("raw_y", ""),
            missing=data.get("missing", False),
        )


@dataclass
class Gap:
    """A detected interval with no usable power samples (canonical seconds)."""

    start: float
    end: float
    reason: str
    first_index: int | None = None
    last_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
            "first_index": self.first_index,
            "last_index": self.last_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Gap":
        return cls(**data)


@dataclass
class Material:
    """An immutable, self-describing source document."""

    kind: str
    label: str
    source_name: str
    time_basis: TimeBasis
    columns: dict[str, str]
    samples: list[Sample] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    fields: dict[str, Quantity] = field(default_factory=dict)
    injections: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    diagnostics: list[Diagnostic] = field(default_factory=list)
    version_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "source_name": self.source_name,
            "time_basis": self.time_basis.to_dict(),
            "columns": dict(self.columns),
            "samples": [s.to_dict() for s in self.samples],
            "gaps": [g.to_dict() for g in self.gaps],
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "injections": list(self.injections),
            "raw_text": self.raw_text,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "version_note": self.version_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Material":
        return cls(
            kind=data["kind"],
            label=data.get("label", ""),
            source_name=data.get("source_name", ""),
            time_basis=TimeBasis.from_dict(data.get("time_basis")),
            columns=dict(data.get("columns", {})),
            samples=[Sample.from_dict(s) for s in data.get("samples", [])],
            gaps=[Gap.from_dict(g) for g in data.get("gaps", [])],
            fields={k: Quantity.from_dict(v) for k, v in data.get("fields", {}).items()},
            injections=list(data.get("injections", [])),
            raw_text=data.get("raw_text", ""),
            diagnostics=[Diagnostic(**d) for d in data.get("diagnostics", [])],
            version_note=data.get("version_note", ""),
        )


# ---------------------------------------------------------------------------
# header parsing
# ---------------------------------------------------------------------------

def _read_header(lines: list[str]) -> tuple[dict[str, str], int]:
    meta: dict[str, str] = {}
    idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            idx = i + 1
            continue
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if ":" in body:
                key, value = body.split(":", 1)
                meta[key.strip().lower()] = value.strip()
            idx = i + 1
            continue
        idx = i
        break
    return meta, idx


def _decode(raw: bytes | str, *, source_name: str) -> str:
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FormatCorrupted(
                f"{source_name}: file is not valid UTF-8 text",
                [diag("encoding_error", str(exc), path=source_name)],
            ) from exc
    return raw


def _table(lines: list[str], start: int, *, source_name: str) -> list[dict[str, str]]:
    rest = "\n".join(lines[start:]).strip()
    if not rest:
        raise FormatCorrupted(
            f"{source_name}: no tabular rows found",
            [diag("empty_table", "file contains metadata but no data rows",
                  path=source_name)],
        )
    try:
        reader = csv.DictReader(io.StringIO(rest))
    except csv.Error as exc:
        raise FormatCorrupted(
            f"{source_name}: malformed CSV ({exc})",
            [diag("csv_error", str(exc), path=source_name)],
        ) from exc
    if reader.fieldnames is None:
        raise FormatCorrupted(
            f"{source_name}: missing CSV header row",
            [diag("missing_header", "expected a column header row",
                  path=source_name)],
        )
    rows = list(reader)
    if not rows:
        raise FormatCorrupted(
            f"{source_name}: header present but no data rows",
            [diag("empty_table", "no data rows", path=source_name)],
        )
    return rows


def _column_map(meta: dict[str, str], *names: str) -> dict[str, str] | None:
    found: dict[str, str] = {}
    for logical, candidates in names:
        token = meta.get(logical)
        if token is None:
            return None
        found[logical] = token.strip()
    return found


# ---------------------------------------------------------------------------
# curve materials: power / baseline / blank
# ---------------------------------------------------------------------------

_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "missing", "-", "--"}


def _is_missing(token: str) -> bool:
    return token.strip().lower() in _MISSING_TOKENS


def _parse_curve(
    kind: str,
    lines: list[str],
    start: int,
    meta: dict[str, str],
    *,
    source_name: str,
) -> Material:
    time_col = meta.get("time_column")
    value_col = meta.get("value_column")
    time_unit = meta.get("time_unit", "s")
    value_dimension = "power"
    value_unit = meta.get("power_unit")
    if kind == "power_curve":
        value_unit = meta.get("power_unit")
    elif kind == "blank_curve":
        value_unit = meta.get("power_unit")
    elif kind == "baseline":
        value_unit = meta.get("baseline_unit", meta.get("power_unit"))
    if not time_col or not value_col:
        raise FormatCorrupted(
            f"{source_name}: {kind} requires time_column and value_column metadata",
            [diag(
                "missing_metadata",
                f"need '# time_column:' and '# value_column:' headers for {kind}",
                path=source_name,
            )],
        )
    if value_unit is None:
        raise FormatCorrupted(
            f"{source_name}: missing power/baseline unit metadata",
            [diag("missing_metadata",
                  f"need '# power_unit:' header for {kind}", path=source_name)],
        )

    rows = _table(lines, start, source_name=source_name)
    if time_col not in rows[0] or value_col not in rows[0]:
        missing = [c for c in (time_col, value_col) if c not in rows[0]]
        raise FormatCorrupted(
            f"{source_name}: declared column(s) {missing} absent from header",
            [diag("unknown_column", f"columns not found: {missing}",
                  path=source_name)],
        )

    label = meta.get("label", kind)
    origin = meta.get("time_origin", "run_start")
    basis = TimeBasis(origin=origin)
    material = Material(
        kind=kind,
        label=label,
        source_name=source_name,
        time_basis=basis,
        columns={"time": time_col, "value": value_col},
        raw_text="\n".join(lines),
        version_note=meta.get("version_note", ""),
    )

    previous_t: float | None = None
    missing_run_start: int | None = None
    median_dt: float | None = None
    raw_dts: list[float] = []
    parsed: list[tuple[Sample, float | None]] = []

    for i, row in enumerate(rows):
        path = f"{source_name}:row{i + 1}"
        raw_t = (row.get(time_col) or "").strip()
        raw_y = (row.get(value_col) or "").strip()
        tq = parse_quantity(raw_t, time_unit, "time", path=path, ref=time_col)
        if not tq.valid:
            material.diagnostics.append(tq.diagnostic)  # type: ignore[arg-type]
            parsed.append((
                Sample(i, tq, None, raw_t, raw_y, missing=True), None,
            ))
            continue
        tv = tq.value
        assert tv is not None
        if previous_t is not None:
            if tv <= previous_t:
                raise SemanticImpossible(
                    f"{source_name}: time column must be strictly increasing",
                    [diag(
                        "non_monotonic_time",
                        f"row {i + 1}: t={raw_t} does not exceed previous value",
                        path=path, ref=time_col,
                    )],
                )
            raw_dts.append(tv - previous_t)
        previous_t = tv
        if _is_missing(raw_y):
            parsed.append((
                Sample(i, tq, None, raw_t, raw_y, missing=True), tv,
            ))
        else:
            yq = parse_quantity(
                raw_y, value_unit, value_dimension, path=path, ref=value_col
            )
            if not yq.valid:
                # Unconvertible value stays in place with its raw token.
                material.diagnostics.append(yq.diagnostic)  # type: ignore[arg-type]
                parsed.append((
                    Sample(i, tq, yq, raw_t, raw_y, missing=True), tv,
                ))
            else:
                parsed.append((
                    Sample(i, tq, yq, raw_t, raw_y, missing=False), tv,
                ))

    for sample, _tv in parsed:
        material.samples.append(sample)

    if raw_dts:
        raw_dts.sort()
        median_dt = raw_dts[len(raw_dts) // 2]

    # Build gaps from explicit missing runs / unreadable values.
    gap_start: float | None = None
    gap_first: int | None = None
    gap_last: int | None = None
    for sample, tv in parsed:
        if tv is None:
            continue
        if sample.missing:
            if gap_start is None:
                gap_start = tv
                gap_first = sample.index
            gap_last = sample.index
        elif gap_start is not None:
            material.gaps.append(
                Gap(
                    start=gap_start,
                    end=tv,
                    reason="missing_samples",
                    first_index=gap_first,
                    last_index=gap_last,
                )
            )
            gap_start = None
            gap_first = gap_last = None
    if gap_start is not None:
        # Missing run reaches the end of the recorded samples; close against
        # the last timestamp we know (the gap extends to end-of-trace).
        last_t = parsed[-1][1]
        material.gaps.append(
            Gap(
                start=gap_start,
                end=last_t if last_t is not None else gap_start,
                reason="missing_samples_to_end",
                first_index=gap_first,
                last_index=gap_last,
            )
        )

    # Detect acquisition jumps (a time step far above the nominal spacing).
    if median_dt and median_dt > 0:
        prev_t: float | None = None
        for sample, tv in parsed:
            if tv is None:
                continue
            if prev_t is not None and (tv - prev_t) > 10.0 * median_dt:
                material.gaps.append(
                    Gap(
                        start=prev_t,
                        end=tv,
                        reason="time_jump",
                        first_index=sample.index - 1,
                        last_index=sample.index,
                    )
                )
            prev_t = tv
    # stable gap order
    material.gaps.sort(key=lambda g: (g.start, g.end, g.reason))
    return material


# ---------------------------------------------------------------------------
# injection schedule
# ---------------------------------------------------------------------------


def _parse_injections(
    lines: list[str],
    start: int,
    meta: dict[str, str],
    *,
    source_name: str,
) -> Material:
    required = ("injection_column", "time_column", "volume_column")
    for key in required:
        if not meta.get(key):
            raise FormatCorrupted(
                f"{source_name}: injections need {key} metadata",
                [diag("missing_metadata", f"need '# {key}:' header",
                      path=source_name)],
            )
    id_col = meta["injection_column"]
    time_col = meta["time_column"]
    vol_col = meta["volume_column"]
    time_unit = meta.get("time_unit", "s")
    vol_unit = meta.get("volume_unit")
    if not vol_unit:
        raise FormatCorrupted(
            f"{source_name}: injections need volume_unit metadata",
            [diag("missing_metadata", "need '# volume_unit:' header",
                  path=source_name)],
        )

    rows = _table(lines, start, source_name=source_name)
    header = rows[0]
    for col in (id_col, time_col, vol_col):
        if col not in header:
            raise FormatCorrupted(
                f"{source_name}: declared column {col!r} not in header",
                [diag("unknown_column", f"{col!r} not found", path=source_name)],
            )

    basis = TimeBasis(origin=meta.get("time_origin", "run_start"))
    material = Material(
        kind="injections",
        label=meta.get("label", "injection schedule"),
        source_name=source_name,
        time_basis=basis,
        columns={"id": id_col, "time": time_col, "volume": vol_col},
        raw_text="\n".join(lines),
        version_note=meta.get("version_note", ""),
    )
    previous_t: float | None = None
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        path = f"{source_name}:row{i + 1}"
        raw_id = (row.get(id_col) or "").strip()
        raw_t = (row.get(time_col) or "").strip()
        raw_v = (row.get(vol_col) or "").strip()
        if raw_id in seen_ids:
            raise SemanticImpossible(
                f"{source_name}: duplicate injection id {raw_id!r}",
                [diag("duplicate_injection_id",
                      f"injection id {raw_id!r} repeats", path=path, ref=id_col)],
            )
        seen_ids.add(raw_id)
        tq = parse_quantity(raw_t, time_unit, "time", path=path, ref=time_col)
        vq = parse_quantity(raw_v, vol_unit, "volume", path=path, ref=vol_col)
        if not tq.valid:
            material.diagnostics.append(tq.diagnostic)  # type: ignore[arg-type]
        else:
            assert tq.value is not None
            if previous_t is not None and tq.value < previous_t:
                raise SemanticImpossible(
                    f"{source_name}: injection times must be non-decreasing",
                    [diag("non_monotonic_time",
                          f"row {i + 1}: injection time {raw_t} precedes prior",
                          path=path, ref=time_col)],
                )
            previous_t = tq.value
        if not vq.valid:
            material.diagnostics.append(vq.diagnostic)  # type: ignore[arg-type]
        elif vq.value is not None and vq.value <= 0:
            raise SemanticImpossible(
                f"{source_name}: injection volume must be positive",
                [diag("non_positive_volume",
                      f"row {i + 1}: volume {raw_v} {vol_unit}",
                      path=path, ref=vol_col)],
            )
        material.injections.append({
            "id": raw_id,
            "time": tq.to_dict(),
            "volume": vq.to_dict(),
            "raw_id": raw_id,
            "raw_time": raw_t,
            "raw_volume": raw_v,
            "index": i,
        })
    return material


# ---------------------------------------------------------------------------
# scalar materials: concentration / temperature / cell volume
# ---------------------------------------------------------------------------

_SCALAR_SPECS = {
    "concentration": {
        "fields": ("syringe", "cell"),
        "dimension": "concentration",
        "unit_meta": ("syringe_unit", "cell_unit"),
        "required_unit": False,
    },
    "temperature": {
        "fields": ("temperature",),
        "dimension": "temperature",
        "unit_meta": ("temperature_unit",),
        "required_unit": True,
    },
    "cell_volume": {
        "fields": ("cell_volume",),
        "dimension": "volume",
        "unit_meta": ("volume_unit",),
        "required_unit": True,
    },
}


def _parse_scalar(
    kind: str,
    lines: list[str],
    start: int,
    meta: dict[str, str],
    *,
    source_name: str,
) -> Material:
    spec = _SCALAR_SPECS[kind]
    rows = _table(lines, start, source_name=source_name)
    material = Material(
        kind=kind,
        label=meta.get("label", kind),
        source_name=source_name,
        time_basis=TimeBasis(),
        columns={},
        raw_text="\n".join(lines),
        version_note=meta.get("version_note", ""),
    )
    row = rows[0]
    for field_name, unit_meta in zip(spec["fields"], spec["unit_meta"]):
        col = meta.get(f"{field_name}_column", field_name)
        if col not in row:
            # A concentration material may legitimately carry only one of
            # syringe/cell; but a named field whose column is missing entirely
            # is a structural problem.
            if kind == "concentration" and field_name == "cell":
                continue
            raise FormatCorrupted(
                f"{source_name}: {kind} column {col!r} missing",
                [diag("unknown_column", f"{col!r} not in header",
                      path=source_name)],
            )
        raw = (row.get(col) or "").strip()
        unit = meta.get(unit_meta)
        if not unit:
            unit = _infer_unit_from_header(col, row)
        if not unit:
            q = parse_quantity(
                raw, "", spec["dimension"],
                path=f"{source_name}:row1", ref=col,
            )
        else:
            q = parse_quantity(
                raw, unit, spec["dimension"],
                path=f"{source_name}:row1", ref=col,
            )
        if not q.valid:
            material.diagnostics.append(q.diagnostic)  # type: ignore[arg-type]
        material.fields[field_name] = q
    if len(rows) > 1:
        material.diagnostics.append(diag(
            "extra_rows_ignored",
            f"{kind} expected a single row; {len(rows) - 1} extra row(s) ignored",
            path=source_name,
        ))
    return material


def _infer_unit_from_header(col: str, row: dict[str, str]) -> str:
    # We do not silently guess dimension-bearing units; this helper only
    # recognizes the explicit bracketed form "syringe (mM)".
    if "(" in col and col.endswith(")"):
        return col[col.rfind("(") + 1:-1].strip()
    return ""


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def guess_kind(source_name: str) -> str | None:
    stem = source_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    stem = stem.rsplit(".", 1)[0].replace("_", "").replace("-", "")
    for token, kind in FILENAME_HINTS.items():
        if token.replace("_", "") in stem:
            return kind
    return None


def parse_material(
    raw: bytes | str,
    *,
    kind: str | None = None,
    source_name: str = "material.csv",
) -> Material:
    """Parse one source document into an immutable :class:`Material`.

    Raises :class:`FormatCorrupted` for unparseable text and
    :class:`SemanticImpossible` for parseable-but-contradictory data.
    Unconvertible *cells* do not raise: they are retained as invalid
    quantities with located diagnostics (curves) or material diagnostics
    (scalars), so downstream decisions can report precise failure positions.
    """

    text = _decode(raw, source_name=source_name)
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        raise FormatCorrupted(
            f"{source_name}: empty file",
            [diag("empty_file", "no content", path=source_name)],
        )
    meta, start = _read_header(lines)
    declared = (meta.get("kind") or "").strip() or None
    resolved = kind or declared or guess_kind(source_name)
    if resolved is None:
        raise FormatCorrupted(
            f"{source_name}: cannot determine material kind",
            [diag(
                "unknown_kind",
                "add a '# kind: power_curve|injections|concentration|temperature|"
                "cell_volume|baseline|blank_curve' header",
                path=source_name,
            )],
        )
    if resolved not in MATERIAL_KINDS:
        raise FormatCorrupted(
            f"{source_name}: unknown material kind {resolved!r}",
            [diag("unknown_kind", f"{resolved!r} is not a supported material kind",
                  path=source_name)],
        )
    if declared and kind and declared != kind:
        raise SemanticImpossible(
            f"{source_name}: declared kind {declared!r} conflicts with {kind!r}",
            [diag("kind_conflict",
                  f"header says {declared!r}, import requested {kind!r}",
                  path=source_name)],
        )

    if resolved in ("power_curve", "baseline", "blank_curve"):
        material = _parse_curve(resolved, lines, start, meta, source_name=source_name)
    elif resolved == "injections":
        material = _parse_injections(lines, start, meta, source_name=source_name)
    else:
        material = _parse_scalar(resolved, lines, start, meta, source_name=source_name)

    _semantic_checks(material)
    return material


def _semantic_checks(material: Material) -> None:
    if material.kind in ("power_curve", "baseline", "blank_curve"):
        if not material.samples:
            raise SemanticImpossible(
                f"{material.source_name}: curve has no samples",
                [diag("empty_curve", "at least one time sample required",
                      path=material.source_name)],
            )
        good = [s for s in material.samples if not s.missing]
        if not good:
            material.diagnostics.append(diag(
                "all_samples_missing",
                "every value sample is missing or unconvertible",
                path=material.source_name,
            ))
    elif material.kind == "injections":
        if not material.injections:
            raise SemanticImpossible(
                f"{material.source_name}: injection schedule is empty",
                [diag("empty_schedule", "at least one injection required",
                      path=material.source_name)],
            )
    elif material.kind == "concentration":
        if "syringe" not in material.fields:
            raise SemanticImpossible(
                f"{material.source_name}: syringe concentration required",
                [diag("missing_field", "syringe concentration column is required",
                      path=material.source_name)],
            )
    elif material.kind in ("temperature", "cell_volume"):
        required_field = {
            "temperature": "temperature",
            "cell_volume": "cell_volume",
        }[material.kind]
        q = material.fields.get(required_field)
        if q is None:
            raise SemanticImpossible(
                f"{material.source_name}: {required_field} missing",
                [diag("missing_field", f"{required_field} value required",
                      path=material.source_name)],
            )
        if q.valid and q.value is not None:
            if material.kind == "temperature" and not (200.0 < q.value < 400.0):
                material.diagnostics.append(diag(
                    "implausible_temperature",
                    f"{q.raw} {q.unit} normalizes to {q.value} K, outside 200-400 K",
                    path=material.source_name,
                ))
            if material.kind == "cell_volume" and not (1e-6 < q.value < 1e-1):
                material.diagnostics.append(diag(
                    "implausible_cell_volume",
                    f"{q.raw} {q.unit} normalizes to {q.value} L, outside 1 uL-100 mL",
                    path=material.source_name,
                ))


def material_summary(material: Material) -> dict[str, Any]:
    """Compact input summary recorded on every derived result."""

    summary: dict[str, Any] = {
        "kind": material.kind,
        "label": material.label,
        "source_name": material.source_name,
        "version_note": material.version_note,
        "time_basis": material.time_basis.to_dict(),
        "n_samples": len(material.samples),
        "n_diagnostics": len(material.diagnostics),
        "gap_count": len(material.gaps),
    }
    if material.kind == "injections":
        summary["n_injections"] = len(material.injections)
    return summary
