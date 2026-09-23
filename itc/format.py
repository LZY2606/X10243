"""ITC bundle text format (*.itc): a domain-specific text encoding.

The format is line based so diagnostics can carry 1-based line numbers.
Original text is preserved verbatim; unit normalisation happens elsewhere.

Grammar::

    ITC-BUNDLE 1.0
    kind: sample | blank
    title: ...
    temperature: <value><unit>
    cell_volume: <value><unit>
    macromolecule_concentration: <value><unit>   # cell
    ligand_concentration: <value><unit>          # syringe
    power_unit: uW
    time_unit: s
    time_origin: 0 s
    @@injections
    index,time_s,volume_uL
    1,120,5
    @@baseline
    start_s,end_s,offset_uW
    0,100,0
    @@power
    # free text comments are kept, not interpreted
    t_s,p_uW
    0.0,0.01
    ...

A missing power stream / injections table is a *semantic* error, distinct from
lexical corruption ("FormatCorrupt") and from an exhausted resource budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .units import parse_quantity

MAGIC = "ITC-BUNDLE"
SUPPORTED_VERSION = "1.0"
HEADER_KIND = "header"


class FormatCorrupt(Exception):
    """Raised when the text cannot even be tokenised as an ITC bundle."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


class BudgetExhausted(Exception):
    """Raised when parsing would exceed an explicit resource budget."""

    def __init__(self, message: str, *, limit: int, used: int) -> None:
        super().__init__(message)
        self.limit = limit
        self.used = used


@dataclass
class ParsedBundle:
    kind: str
    title: str
    headers: dict[str, str]
    quantities: dict[str, dict[str, Any]]
    injections: list[dict[str, Any]]
    power: list[list[float | None]]
    baseline: list[dict[str, Any]]
    comments: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not any(d.get("severity") == "error" for d in self.diagnostics)


_NUMERIC_COLUMNS = {
    "injections": {"index": int, "time_s": float, "volume_uL": float},
    "baseline": {"start_s": float, "end_s": float, "offset_uW": float},
}


def parse_bundle(
    text: str,
    *,
    max_bytes: int = 8_000_000,
    max_power_rows: int = 2_000_000,
    max_injections: int = 100_000,
    source_name: str = "<text>",
) -> ParsedBundle:
    if not isinstance(text, str):
        raise FormatCorrupt("Input must be decoded text (utf-8).", line=None)
    used_bytes = len(text.encode("utf-8"))
    if used_bytes > max_bytes:
        raise BudgetExhausted(
            f"Byte budget exceeded: {used_bytes} > {max_bytes}",
            limit=max_bytes,
            used=used_bytes,
        )
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise FormatCorrupt("Empty input: missing ITC-BUNDLE header.", line=1)
    first = lines[0].strip().split()
    if len(first) < 2 or first[0] != MAGIC:
        raise FormatCorrupt(
            f"First line must start with {MAGIC!r} and a version.", line=1
        )
    version = first[1]
    if version != SUPPORTED_VERSION:
        raise FormatCorrupt(
            f"Unsupported bundle version {version!r}; expected {SUPPORTED_VERSION!r}.",
            line=1,
        )

    diag: list[dict[str, Any]] = []
    headers: dict[str, str] = {}
    section: str | None = None
    section_header_cols: list[str] | None = None
    injections: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    power: list[list[float | None]] = []
    comments: list[dict[str, Any]] = []
    power_rows = 0
    injection_rows = 0

    quantity_specs = {
        "temperature": "temperature",
        "cell_volume": "volume",
        "macromolecule_concentration": "concentration",
        "ligand_concentration": "concentration",
    }

    def add_err(code: str, line: int, message: str, **extra: Any) -> None:
        entry = {"code": code, "severity": "error", "line": line,
                 "source": source_name, "message": message}
        entry.update(extra)
        diag.append(entry)

    def add_note(code: str, line: int, message: str, **extra: Any) -> None:
        entry = {"code": code, "severity": "info", "line": line,
                 "source": source_name, "message": message}
        entry.update(extra)
        diag.append(entry)

    for lineno, raw_line in enumerate(lines[1:], start=2):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append({"line": lineno, "text": raw_line})
            continue
        if line.startswith("@@"):
            name = line[2:].strip().lower()
            if name not in ("injections", "baseline", "power"):
                add_err("unknown_section", lineno, f"Unknown section @@{name}.")
            section = name
            section_header_cols = None
            continue
        if section is None:
            if ":" not in line:
                add_err("bad_header", lineno,
                        "Header lines must be 'key: value'.")
                continue
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()
            continue
        # inside a section: first non-comment row is the column header.
        if section_header_cols is None:
            section_header_cols = [c.strip() for c in line.split(",")]
            expected = {
                "injections": ["index", "time_s", "volume_uL"],
                "baseline": ["start_s", "end_s", "offset_uW"],
                "power": ["t_s", "p_uW"],
            }[section]
            if section_header_cols != expected:
                add_err(
                    "bad_columns", lineno,
                    f"Columns {section_header_cols!r} do not match {expected!r}.",
                )
            continue

        cells = [c.strip() for c in line.split(",")]
        if section == "injections":
            injection_rows += 1
            if injection_rows > max_injections:
                raise BudgetExhausted(
                    f"Injection row budget exceeded ({max_injections})",
                    limit=max_injections, used=injection_rows,
                )
            row = _parse_row(cells, _NUMERIC_COLUMNS["injections"], lineno,
                             add_err)
            if row is not None:
                injections.append(row)
        elif section == "baseline":
            row = _parse_row(cells, _NUMERIC_COLUMNS["baseline"], lineno,
                             add_err)
            if row is not None:
                baseline.append(row)
        elif section == "power":
            power_rows += 1
            if power_rows > max_power_rows:
                raise BudgetExhausted(
                    f"Power row budget exceeded ({max_power_rows})",
                    limit=max_power_rows, used=power_rows,
                )
            power.append(_parse_power_row(cells, lineno, add_err, add_note))

    kind = headers.get("kind", "").strip()
    if kind not in ("sample", "blank"):
        add_err("bad_kind", 1, "Header 'kind' must be 'sample' or 'blank'.")
    if "title" not in headers:
        add_err("missing_title", 1, "Header 'title' is required.")
    if not injections:
        add_err("missing_stream", 0, "Section @@injections has no data rows.",
                stream="injections")
    if not power:
        add_err("missing_stream", 0, "Section @@power has no data rows.",
                stream="power")

    quantities: dict[str, dict[str, Any]] = {}
    for key, qkind in quantity_specs.items():
        raw_value = headers.get(key)
        if raw_value is None:
            add_err("missing_header", 0, f"Header {key!r} is required.",
                    header=key)
            continue
        q = parse_quantity(raw_value, qkind, location=f"header:{key}")
        for d in q.diagnostics:
            d["line"] = 0
            d["source"] = source_name
            d["severity"] = "error" if d["code"] in (
                "unknown_unit", "unparseable_number", "non_finite_number",
                "missing_value") else "warning"
            diag.append(d)
        quantities[key] = q.to_dict()
        quantities[key]["raw_header"] = f"{key}: {raw_value}"

    # Injection monotonicity/order checks (semantic, not lexical).
    last_time = float("-inf")
    for row in injections:
        t = row["time_s"]
        if t < last_time:
            add_err("injection_order", 0,
                    "Injection times must be non-decreasing.",
                    injection_index=row["index"], time_s=t)
        last_time = t
        if row["volume_uL"] <= 0:
            add_err("bad_volume", 0, "Injection volume must be positive.",
                    injection_index=row["index"])

    if power:
        t0 = power[0][0]
        tlast = power[-1][0]
        if t0 is None or tlast is None:
            add_err("power_anchor_missing", 0,
                    "First/last power timestamps are required.")
        else:
            for row in injections:
                if not (t0 <= row["time_s"] <= tlast):
                    add_err("injection_outside_stream", 0,
                            "Injection time lies outside the power stream.",
                            injection_index=row["index"],
                            time_s=row["time_s"],
                            stream_start=t0, stream_end=tlast)

    return ParsedBundle(
        kind=kind,
        title=headers.get("title", ""),
        headers=headers,
        quantities=quantities,
        injections=injections,
        power=power,
        baseline=baseline,
        comments=comments,
        diagnostics=diag,
        raw=text,
    )


def _parse_row(cells: list[str], columns: dict[str, type], lineno: int,
               add_err: Any) -> dict[str, Any] | None:
    if len(cells) != len(columns):
        add_err("bad_cell_count", lineno,
                f"Expected {len(columns)} columns, got {len(cells)}.")
        return None
    row: dict[str, Any] = {}
    for (name, caster), cell in zip(columns.items(), cells):
        try:
            row[name] = caster(cell)
        except ValueError:
            add_err("bad_number", lineno,
                    f"Column {name!r} value {cell!r} is not numeric.",
                    column=name)
            return None
    return row


def _parse_power_row(cells: list[str], lineno: int, add_err: Any,
                     add_note: Any) -> list[float | None]:
    if len(cells) != 2:
        add_err("bad_cell_count", lineno,
                f"Power rows need 2 columns, got {len(cells)}.")
        return [None, None]
    t: float | None
    p: float | None
    try:
        t = float(cells[0])
    except ValueError:
        add_err("bad_number", lineno,
                f"Time value {cells[0]!r} is not numeric.", column="t_s")
        t = None
    if cells[1].upper() in ("NA", "N/A", "MISSING", ""):
        p = None
        add_note("missing_sample", lineno,
                 "Power sample missing; coordinate retained.", column="p_uW")
    else:
        try:
            p = float(cells[1])
        except ValueError:
            add_err("bad_number", lineno,
                    f"Power value {cells[1]!r} is not numeric.",
                    column="p_uW")
            p = None
    return [t, p]
