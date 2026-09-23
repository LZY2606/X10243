"""Import raw ITC materials without normalizing away provenance.

The on-disk case format is a deliberately domain-specific text format (not a
generic spreadsheet).  Every numeric domain field keeps its original textual
representation and unit; only ``parsed.canonical_*`` derived fields are
converted.  Missing samples stay missing and are surfaced as gaps/diagnostics
rather than being replaced by zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostics, FormatCorrupted, Severity
from .identity import material_id
from .units import parse_quantity, parse_value

POWER_SUFFIX = ".itc-power.txt"
INJECT_SUFFIX = ".inject.csv"
CONC_SUFFIX = ".conc.txt"
COND_SUFFIX = ".cond.txt"
BASELINE_SUFFIX = ".baseline.csv"


@dataclass
class ParsedMaterial:
    kind: str
    role: str
    payload: dict[str, Any]
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    @property
    def material_id(self) -> str:
        return material_id(self.kind, self.payload)


def _kv_header(lines: list[str], path: str, diags: Diagnostics) -> dict[str, str]:
    header: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if line == "" or line.startswith("#") or line.startswith("["):
            continue
        if line.lower().startswith(("time,", "time\t")):
            break
        if ":" not in line:
            raise FormatCorrupted(
                f"header line missing ':' -> {line!r}", path=path, value=line
            )
        key, value = line.split(":", 1)
        header[key.strip().lower()] = value.strip()
    return header


def _series(path: str, text: str, diags: Diagnostics,
            explicit_gaps: list[dict[str, float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    in_series = False
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line.lower() == "[series]":
            in_series = True
            continue
        if not in_series or line == "" or line.startswith("#"):
            continue
        if line.lower().startswith("time,") or line.lower().startswith("time\t"):
            continue
        if line.lower() == "gap":
            # explicit structural gap marker; delimited by header fields
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        if len(parts) != 2:
            raise FormatCorrupted(
                f"series row must have 2 columns, got {len(parts)}",
                path=path, line=lineno, value=line,
            )
        tcell, pcell = parts
        try:
            tval = parse_value(tcell, path=f"{path}:{lineno}:1")
        except FormatCorrupted as exc:
            diags.add("FORMAT_CORRUPTED", str(exc), f"{path}:{lineno}:1",
                      value=exc.value)
            continue
        if tval is None:
            raise FormatCorrupted(
                "time sample cannot be missing", path=f"{path}:{lineno}:1",
                line=lineno, value=tcell,
            )
        if pcell == "" or pcell.lower() in {"na", "n/a", "missing", "--"}:
            rows.append({"time": tval, "power": None,
                         "raw": [tcell, pcell], "line": lineno})
            diags.add("POWER_SAMPLE_MISSING",
                      f"missing power sample at t={tval}",
                      f"{path}:{lineno}:2", Severity.WARNING, value=pcell)
            continue
        try:
            pval = parse_value(pcell, path=f"{path}:{lineno}:2")
        except FormatCorrupted as exc:
            diags.add("UNCONVERTIBLE_SAMPLE", str(exc), f"{path}:{lineno}:2",
                      Severity.WARNING, value=exc.value)
            rows.append({"time": tval, "power": None,
                         "raw": [tcell, pcell], "line": lineno})
            continue
        rows.append({"time": tval, "power": pval,
                     "raw": [tcell, pcell], "line": lineno})
    if not rows:
        raise FormatCorrupted("[series] contains no rows", path=path)
    times = [r["time"] for r in rows]
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            diags.add("NON_MONOTONIC_TIME",
                      f"time not strictly increasing at row {rows[i]['line']}",
                      f"{path}:{rows[i]['line']}:1", Severity.ERROR,
                      value=times[i])
    return {"rows": rows, "explicit_gaps": explicit_gaps}


def parse_power_curve(text: str, *, role: str, path: str) -> ParsedMaterial:
    diags = Diagnostics()
    lines = text.splitlines()
    try:
        header = _kv_header(lines, path, diags)
    except FormatCorrupted as exc:
        return _fail("power_curve", role, path, text, exc)

    explicit_gaps: list[dict[str, float]] = []
    for key, raw in header.items():
        if key == "gap":
            try:
                lo, hi = [float(x) for x in raw.split(",")]
            except (ValueError, AttributeError):
                diags.add("BAD_GAP_HEADER", f"bad gap declaration {raw!r}",
                          f"{path}#gap", value=raw)
                continue
            explicit_gaps.append({"start": lo, "end": hi})

    required = ["instrument_timebase", "power_unit", "temperature",
                "cell_volume"]
    for key in required:
        if key not in header:
            diags.add("MISSING_HEADER", f"power curve missing header {key!r}",
                      f"{path}#{key}")

    def q(kind: str, key: str) -> dict[str, Any]:
        qty = parse_quantity(header.get(key), kind, path=f"{path}#{key}")
        if not qty.ok:
            diags.add("UNCONVERTIBLE_HEADER",
                      f"cannot convert {key}={header.get(key)!r}",
                      f"{path}#{key}", Severity.ERROR, value=header.get(key))
        return qty.to_dict()

    try:
        series = _series(path, text, diags, explicit_gaps)
    except FormatCorrupted as exc:
        return _fail("power_curve", role, path, text, exc)

    payload = {
        "format": "itc-power/v1",
        "role": role,
        "path": path,
        "raw_text": text,
        "run": header.get("run", ""),
        "instrument_timebase": q("time", "instrument_timebase"),
        "power_unit": {"raw": header.get("power_unit", ""),
                       "unit": (header.get("power_unit", "") or "").strip()},
        "temperature": q("temperature", "temperature"),
        "cell_volume": q("volume", "cell_volume"),
        "series": series,
    }
    return ParsedMaterial("power_curve", role, payload, diags)


def parse_baseline(text: str, *, role: str, path: str) -> ParsedMaterial:
    diags = Diagnostics()
    lines = text.splitlines()
    try:
        header = _kv_header(lines, path, diags)
    except FormatCorrupted as exc:
        return _fail("baseline", role, path, text, exc)
    try:
        series = _series(path, text, diags, [])
    except FormatCorrupted as exc:
        return _fail("baseline", role, path, text, exc)
    payload = {
        "format": "itc-baseline/v1",
        "role": role,
        "path": path,
        "raw_text": text,
        "power_unit": {"raw": header.get("power_unit", ""),
                       "unit": (header.get("power_unit", "") or "").strip()},
        "kind_hint": header.get("baseline", "linear"),
        "series": series,
    }
    return ParsedMaterial("baseline", role, payload, diags)


def parse_injections(text: str, *, role: str, path: str) -> ParsedMaterial:
    diags = Diagnostics()
    rows: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        if parts[0].lower() in {"injection", "inj", "no"}:
            header_cols = parts
            continue
        if len(parts) < 3:
            return _fail("injection_schedule", role, path, text,
                         FormatCorrupted(
                             "injection row needs no,time,volume[,duration]",
                             path=f"{path}:{lineno}", line=lineno, value=line))
        no_raw, t_raw, v_raw = parts[0], parts[1], parts[2]
        dur_raw = parts[3] if len(parts) > 3 else None
        try:
            no = int(parse_value(no_raw, f"{path}:{lineno}:1"))  # type: ignore[arg-type]
        except (FormatCorrupted, TypeError, ValueError):
            return _fail("injection_schedule", role, path, text,
                         FormatCorrupted("injection number not an integer",
                                         path=f"{path}:{lineno}:1",
                                         line=lineno, value=no_raw))
        tq = parse_quantity(t_raw, "time", path=f"{path}:{lineno}:2")
        vq = parse_quantity(v_raw, "volume", path=f"{path}:{lineno}:3")
        dq = (parse_quantity(dur_raw, "time", path=f"{path}:{lineno}:4")
              if dur_raw else None)
        if not tq.ok:
            diags.add("UNCONVERTIBLE_INJECTION_TIME",
                      f"injection {no}: bad time {t_raw!r}",
                      f"{path}:{lineno}:2", value=t_raw)
        if not vq.ok:
            diags.add("UNCONVERTIBLE_INJECTION_VOLUME",
                      f"injection {no}: bad volume {v_raw!r}",
                      f"{path}:{lineno}:3", value=v_raw)
        rows.append({"injection": no, "time": tq.to_dict(),
                     "volume": vq.to_dict(),
                     "duration": dq.to_dict() if dq else None,
                     "raw": [no_raw, t_raw, v_raw] + ([dur_raw] if dur_raw else []),
                     "line": lineno})
    if not rows:
        return _fail("injection_schedule", role, path, text,
                     FormatCorrupted("no injection rows", path=path))
    nos = [r["injection"] for r in rows]
    if sorted(nos) != list(range(1, len(nos) + 1)):
        diags.add("INJECTION_NUMBERING",
                  "injection numbers must be 1..N with no gaps",
                  f"{path}#injection", Severity.ERROR, value=nos)
    payload = {"format": "itc-injections/v1", "role": role, "path": path,
               "raw_text": text, "injections": rows}
    return ParsedMaterial("injection_schedule", role, payload, diags)


def parse_concentration(text: str, *, role: str, path: str) -> ParsedMaterial:
    diags = Diagnostics()
    lines = text.splitlines()
    try:
        header = _kv_header(lines, path, diags)
    except FormatCorrupted as exc:
        return _fail("concentration", role, path, text, exc)

    def q(key: str) -> dict[str, Any]:
        qty = parse_quantity(header.get(key), "concentration",
                             path=f"{path}#{key}")
        if not qty.ok:
            diags.add("UNCONVERTIBLE_CONCENTRATION",
                      f"cannot convert {key}={header.get(key)!r}",
                      f"{path}#{key}", Severity.ERROR, value=header.get(key))
        return qty.to_dict()

    payload = {
        "format": "itc-concentration/v1", "role": role, "path": path,
        "raw_text": text,
        "syringe_concentration": q("syringe_concentration"),
        "cell_concentration": q("cell_concentration"),
        "raw_fields": {k: v for k, v in header.items()
                       if "concentration" in k},
    }
    return ParsedMaterial("concentration", role, payload, diags)


def parse_conditions(text: str, *, role: str, path: str) -> ParsedMaterial:
    diags = Diagnostics()
    try:
        header = _kv_header(lines=text.splitlines(), path=path, diags=diags)
    except FormatCorrupted as exc:
        return _fail("conditions", role, path, text, exc)

    def q(kind: str, key: str) -> dict[str, Any]:
        qty = parse_quantity(header.get(key), kind, path=f"{path}#{key}")
        if not qty.ok:
            diags.add("UNCONVERTIBLE_CONDITION",
                      f"cannot convert {key}={header.get(key)!r}",
                      f"{path}#{key}", Severity.ERROR, value=header.get(key))
        return qty.to_dict()

    payload = {
        "format": "itc-conditions/v1", "role": role, "path": path,
        "raw_text": text,
        "temperature": q("temperature", "temperature"),
        "cell_volume": q("volume", "cell_volume"),
        "instrument_timebase": q("time", "instrument_timebase"),
        "reference_power": (q("power", "reference_power")
                            if "reference_power" in header else None),
    }
    return ParsedMaterial("conditions", role, payload, diags)


def _fail(kind: str, role: str, path: str, text: str,
          exc: FormatCorrupted) -> ParsedMaterial:
    diags = Diagnostics()
    diags.add("FORMAT_CORRUPTED", str(exc), exc.path, Severity.ERROR,
              exc.value, line=exc.line, column=exc.column)
    # raw material retained verbatim; downstream layer sees a blocked, not
    # silently dropped, source.
    payload = {"format": "unknown", "role": role, "path": path,
               "raw_text": text}
    return ParsedMaterial(kind, role, payload, diags)
