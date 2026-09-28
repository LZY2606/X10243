from __future__ import annotations

import csv
import io
from typing import Any

from .domain import ErrorKind, diagnostic, malformed
from .quantities import quantity_or_diagnostic


REQUIRED_COLUMNS = {
    "power": ["time", "power"],
    "baseline": ["time", "baseline_power"],
    "blank": ["time", "blank_power", "injection_ordinal"],
    "injections": ["ordinal", "start_time", "injection_volume", "end_time"],
    "concentration": ["role", "concentration"],
    "conditions": ["name", "value"],
}


def parse_csv(raw_text: str, kind: str) -> dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise malformed("Input CSV is empty")
    try:
        reader = csv.DictReader(io.StringIO(raw_text, newline=""))
    except csv.Error as exc:
        raise malformed(f"CSV structure is unreadable: {exc}") from exc
    if reader.fieldnames is None:
        raise malformed("CSV header is missing")
    headers = [h.strip() if h else h for h in reader.fieldnames]
    missing = [column for column in REQUIRED_COLUMNS[kind] if column not in headers]
    if missing:
        raise malformed("CSV is missing required columns", "$.header", missing)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for line_number, raw_row in enumerate(reader, start=2):
        row = {(key.strip() if key else key): value for key, value in raw_row.items()}
        if None in row:
            diagnostics.append(
                diagnostic(
                    ErrorKind.MALFORMED_INPUT.value,
                    "Row contains more fields than the header",
                    f"$.rows[{line_number - 2}]",
                    "error",
                    row[None],
                )
            )
            continue
        rows.append({"line": line_number, "index": len(rows), "raw": row, "diagnostics": []})
    result = {"kind": kind, "headers": headers, "rows": rows, "diagnostics": diagnostics}
    dispatch = {
        "power": lambda: _parse_series(result, "power"),
        "baseline": lambda: _parse_series(result, "baseline_power"),
        "blank": lambda: _parse_series(result, "blank_power", ordinal=True),
        "injections": lambda: _parse_injections(result),
        "concentration": lambda: _parse_concentrations(result),
        "conditions": lambda: _parse_conditions(result),
    }
    dispatch[kind]()
    result["diagnostics"].extend(diag for row in rows for diag in row["diagnostics"])
    return result


def _parse_series(result: dict[str, Any], value_column: str, *, ordinal: bool = False) -> None:
    points: list[dict[str, Any]] = []
    previous_time: float | None = None
    intervals: list[float] = []
    for row in result["rows"]:
        time_q, time_diag = quantity_or_diagnostic(
            row["raw"].get("time"), "time", f"$.rows[{row['index']}].time"
        )
        value_q, value_diag = quantity_or_diagnostic(
            row["raw"].get(value_column), "power", f"$.rows[{row['index']}].{value_column}"
        )
        ordinal_value = None
        ordinal_diag = None
        if ordinal:
            raw_ordinal = row["raw"].get("injection_ordinal", "")
            if raw_ordinal not in (None, ""):
                try:
                    ordinal_value = int(str(raw_ordinal).strip())
                    if ordinal_value < 1:
                        raise ValueError
                except ValueError:
                    ordinal_diag = diagnostic(
                        "unconvertible_ordinal",
                        "Blank injection ordinal must be a positive integer",
                        f"$.rows[{row['index']}].injection_ordinal",
                        "error",
                        raw_ordinal,
                    )
        for diag in (time_diag, value_diag, ordinal_diag):
            if diag:
                row["diagnostics"].append(diag)
        if time_q and value_q and ordinal_diag is None:
            t = time_q["value"]
            if previous_time is not None:
                if t <= previous_time:
                    row["diagnostics"].append(
                        diagnostic(
                            "non_increasing_time",
                            "Sample time must be strictly greater than the previous sample",
                            f"$.rows[{row['index']}].time",
                            "error",
                            time_q["raw"],
                        )
                    )
                else:
                    intervals.append(t - previous_time)
            previous_time = t
            points.append(
                {
                    "index": len(points),
                    "source_index": row["index"],
                    "line": row["line"],
                    "time": t,
                    "time_quantity": time_q,
                    "value": value_q["value"],
                    "value_quantity": value_q,
                    "injection_ordinal": ordinal_value,
                }
            )
    result["points"] = points
    result["nominal_interval_s"] = min(intervals) if intervals else None


def _parse_injections(result: dict[str, Any]) -> None:
    injections = []
    for row in result["rows"]:
        raw = row["raw"]
        start, start_diag = quantity_or_diagnostic(
            raw.get("start_time"), "time", f"$.rows[{row['index']}].start_time"
        )
        end, end_diag = quantity_or_diagnostic(
            raw.get("end_time"), "time", f"$.rows[{row['index']}].end_time"
        )
        volume, volume_diag = quantity_or_diagnostic(
            raw.get("injection_volume"),
            "volume",
            f"$.rows[{row['index']}].injection_volume",
            positive=True,
        )
        try:
            ordinal = int(str(raw.get("ordinal", "")).strip())
            ordinal_diag = None if ordinal >= 1 else ValueError()
        except (TypeError, ValueError):
            ordinal = None
            ordinal_diag = diagnostic(
                "invalid_ordinal",
                "Injection ordinal must be a positive integer",
                f"$.rows[{row['index']}].ordinal",
                "error",
                raw.get("ordinal"),
            )
        for diag in (start_diag, end_diag, volume_diag, ordinal_diag):
            if diag:
                row["diagnostics"].append(diag)
        if start and end and volume and ordinal is not None:
            if end["value"] <= start["value"]:
                row["diagnostics"].append(
                    diagnostic(
                        "invalid_integration_endpoint",
                        "Integration end must be after injection start",
                        f"$.rows[{row['index']}].end_time",
                        "error",
                        end["raw"],
                        f"injection-{ordinal}",
                    )
                )
            injections.append(
                {
                    "ordinal": ordinal,
                    "source_index": row["index"],
                    "line": row["line"],
                    "start_s": start["value"],
                    "end_s": end["value"],
                    "start_quantity": start,
                    "end_quantity": end,
                    "volume_uL": volume["value"],
                    "volume_quantity": volume,
                    "raw": raw,
                }
            )
    seen = [item["ordinal"] for item in injections]
    duplicates = {ordinal for ordinal in seen if seen.count(ordinal) > 1}
    for item in injections:
        item["duplicate"] = item["ordinal"] in duplicates
    injections.sort(key=lambda item: (item["start_s"], item["ordinal"]))
    result["injections"] = injections


def _parse_concentrations(result: dict[str, Any]) -> None:
    values = {}
    for row in result["rows"]:
        raw = row["raw"]
        role = (raw.get("role") or "").strip().lower()
        if role not in {"macromolecule", "ligand", "syringe_ligand"}:
            row["diagnostics"].append(
                diagnostic(
                    "unknown_concentration_role",
                    "Role must be macromolecule, ligand, or syringe_ligand",
                    f"$.rows[{row['index']}].role",
                    "error",
                    raw.get("role"),
                )
            )
            continue
        qty, diag = quantity_or_diagnostic(
            raw.get("concentration"),
            "concentration",
            f"$.rows[{row['index']}].concentration",
            positive=True,
        )
        if diag:
            row["diagnostics"].append(diag)
        if qty:
            values[role] = {"value_uM": qty["value"], "quantity": qty, "raw": raw}
    result["concentrations"] = values


def _parse_conditions(result: dict[str, Any]) -> None:
    values = {}
    parsers = {"temperature": ("temperature", False), "cell_volume": ("volume", True)}
    for row in result["rows"]:
        raw = row["raw"]
        name = (raw.get("name") or "").strip()
        if name not in parsers:
            row["diagnostics"].append(
                diagnostic(
                    "unknown_condition",
                    "Condition must be temperature or cell_volume",
                    f"$.rows[{row['index']}].name",
                    "warning",
                    name,
                )
            )
            continue
        dimension, positive = parsers[name]
        qty, diag = quantity_or_diagnostic(
            raw.get("value"),
            dimension,
            f"$.rows[{row['index']}].value",
            positive=positive,
        )
        if diag:
            row["diagnostics"].append(diag)
        if qty:
            values[name] = qty
    result["conditions"] = values
