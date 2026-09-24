"""Raw-file import.

Parses the lab CSV formats into immutable MaterialVersion objects. The raw
text representation is preserved on every point/scalar. Import failures are
classified into format corruption vs semantic impossibility; per-row problems
become positioned diagnostics and never silent NaNs/defaults.
"""

from __future__ import annotations

import csv
import io

from .errors import FormatCorrupt, SemanticImpossible
from .materials import Injection, MaterialVersion, Series
from .units import CANONICAL, Quantity, known_units, to_canonical


def _read_rows(text: str, origin: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.reader(io.StringIO(text))
        raw = [r for r in reader if any(c.strip() for c in r)]
    except csv.Error as exc:
        raise FormatCorrupt(f"{origin}: malformed CSV ({exc})") from exc
    if len(raw) < 2:
        raise FormatCorrupt(f"{origin}: CSV needs a header row and at least one data row")
    header = [h.strip() for h in raw[0]]
    if len(set(header)) != len(header):
        raise FormatCorrupt(f"{origin}: duplicate header columns {header}")
    rows: list[dict[str, str]] = []
    for i, r in enumerate(raw[1:], start=2):
        if len(r) != len(header):
            raise FormatCorrupt(
                f"{origin}: row {i} has {len(r)} fields, header has {len(header)}"
            )
        rows.append({h: r[j].strip() for j, h in enumerate(header)})
    return header, rows


def _unit_from_header(header: list[str], dim: str, origin: str, fallback: str) -> str:
    for h in header:
        if "(" in h and h.endswith(")"):
            unit = h[h.rfind("(") + 1 : -1].strip()
            if unit in known_units(dim):
                return unit
    return fallback


def _series_kind(kind: str):
    return {
        "power": ("power", "mW"),
        "baseline": ("power", "mW"),
        "blank": ("power", "mW"),
    }[kind]


def import_series(kind: str, text: str, origin: str, version: int) -> tuple[MaterialVersion, list[dict]]:
    dim, default_unit = _series_kind(kind)
    header, rows = _read_rows(text, origin)
    tcol = next((h for h in header if h.lower().startswith(("time_s", "time(s)", "time"))), None)
    vcol = next((h for h in header if h.lower().startswith("power")), None)
    if tcol is None or vcol is None:
        raise FormatCorrupt(f"{origin}: needs time and power columns, got {header}")
    tunit = _unit_from_header([tcol], "time", origin, "s")
    vunit = _unit_from_header([vcol], dim, origin, default_unit)
    points: list[list] = []
    diagnostics: list[dict] = []
    for line, row in enumerate(rows, start=2):
        raw_t, raw_v = row[tcol], row[vcol]
        ok = True
        try:
            tq = Quantity.parse(raw_t, "time", default_unit=tunit)
            t = tq.value
        except SemanticImpossible:
            t, ok = float("nan"), False
            diagnostics.append(_diag(origin, line, "unconvertible_time", raw_t, "time"))
        try:
            vq = Quantity.parse(raw_v, dim, default_unit=vunit)
            v = vq.value
        except SemanticImpossible:
            v, ok = float("nan"), False
            diagnostics.append(_diag(origin, line, "unconvertible_value", raw_v, dim))
        points.append([t, v, raw_t, raw_v, ok])
    series = Series(points=points, value_dimension=dim, value_unit=CANONICAL[dim],
                    time_unit=CANONICAL["time"], origin=origin)
    return MaterialVersion(kind=kind, version=version, origin=origin, series=series), diagnostics


def _diag(origin: str, line: int, code: str, raw: str, dim: str) -> dict:
    return {
        "code": code,
        "origin": origin,
        "line": line,
        "raw": raw,
        "dimension": dim,
        "message": f"{origin}:{line} value '{raw}' cannot be converted; retained at origin",
    }


def import_schedule(text: str, origin: str, version: int) -> tuple[MaterialVersion, list[dict]]:
    header, rows = _read_rows(text, origin)
    required = ("shot", "time", "volume")
    for name in required:
        if not any(h.lower().startswith(name) for h in header):
            raise FormatCorrupt(f"{origin}: schedule needs {required} columns, got {header}")
    cshot = next(h for h in header if h.lower().startswith("shot"))
    ctime = next(h for h in header if h.lower().startswith("time"))
    cvol = next(h for h in header if h.lower().startswith("volume"))
    cdur = next((h for h in header if h.lower().startswith("duration")), None)
    tunit = _unit_from_header([ctime], "time", origin, "s")
    vunit = _unit_from_header([cvol], "volume", origin, "uL")
    dunit = _unit_from_header([cdur], "time", origin, "s") if cdur else "s"
    injections: list[Injection] = []
    diagnostics: list[dict] = []
    for line, row in enumerate(rows, start=2):
        try:
            idx = int(float(row[cshot]))
        except ValueError:
            raise FormatCorrupt(f"{origin}:{line} shot index '{row[cshot]}' is not integer")
        try:
            t = Quantity.parse(row[ctime], "time", default_unit=tunit).value
        except SemanticImpossible as exc:
            raise SemanticImpossible(
                f"{origin}:{line} injection time unusable", diagnostics=exc.diagnostics
            ) from exc
        try:
            vol = Quantity.parse(row[cvol], "volume", default_unit=vunit).value
        except SemanticImpossible as exc:
            raise SemanticImpossible(
                f"{origin}:{line} injection volume unusable", diagnostics=exc.diagnostics
            ) from exc
        dur = 0.0
        if cdur:
            try:
                dur = Quantity.parse(row[cdur], "time", default_unit=dunit).value
            except SemanticImpossible as exc:
                diagnostics.append({"code": "unconvertible_value", "origin": origin,
                                    "line": line, "raw": row[cdur], "dimension": "time"})
                dur = 0.0
        injections.append(Injection(idx, t, vol, dur, row[ctime], row[cvol]))
    injections = tuple(sorted(injections, key=lambda i: (i.time_s, i.index)))
    if len({i.index for i in injections}) != len(injections):
        raise SemanticImpossible(f"{origin}: duplicate shot indices")
    return (
        MaterialVersion(kind="schedule", version=version, origin=origin, injections=injections),
        diagnostics,
    )


def import_scalar(kind: str, dim: str, text: str, origin: str, version: int
                  ) -> tuple[MaterialVersion, list[dict]]:
    header, rows = _read_rows(text, origin)
    value_col = next((h for h in header if h.lower().startswith(("value", kind, "temperature",
                                                                  "concentration", "volume"))),
                     header[1] if len(header) > 1 else header[0])
    unit = _unit_from_header([value_col], dim, origin, CANONICAL[dim])
    diagnostics: list[dict] = []
    latest: tuple[float, str] | None = None
    for line, row in enumerate(rows, start=2):
        raw = row[value_col]
        try:
            q = Quantity.parse(raw, dim, default_unit=unit)
            latest = (q.value, raw)
        except SemanticImpossible:
            diagnostics.append(_diag(origin, line, "unconvertible_value", raw, dim))
    if latest is None:
        raise SemanticImpossible(
            f"{origin}: no convertible {kind} value",
            diagnostics=diagnostics or [{"code": "missing_value", "origin": origin}],
        )
    return (
        MaterialVersion(kind=kind, version=version, origin=origin,
                        quantity=latest[0], unit=CANONICAL[dim], raw=latest[1]),
        diagnostics,
    )


def import_concentration(text: str, origin: str, version: int
                         ) -> tuple[MaterialVersion, list[dict]]:
    """Concentration sheet: latest convertible syringe/cell pair wins.

    Invalid rows stay in the diagnostics with positions instead of silently
    falling back to prior values.
    """
    header, rows = _read_rows(text, origin)
    csyr = next((h for h in header if "syringe" in h.lower()), None)
    ccell = next((h for h in header if "cell" in h.lower()), None)
    if csyr is None or ccell is None:
        raise FormatCorrupt(f"{origin}: concentration sheet needs syringe_M and cell_M columns")
    sunit = _unit_from_header([csyr], "concentration", origin, "mM")
    cellunit = _unit_from_header([ccell], "concentration", origin, "mM")
    diagnostics: list[dict] = []
    syringe = cell = None
    for line, row in enumerate(rows, start=2):
        for col, dim_unit, slot in ((csyr, sunit, "syringe"), (ccell, cellunit, "cell")):
            raw = row[col]
            if raw == "":
                continue
            try:
                q = Quantity.parse(raw, "concentration", default_unit=dim_unit)
                if slot == "syringe":
                    syringe = q
                else:
                    cell = q
            except SemanticImpossible:
                diagnostics.append(_diag(origin, line, "unconvertible_value", raw, "concentration"))
    if syringe is None or cell is None:
        raise SemanticImpossible(
            f"{origin}: concentration sheet requires both syringe and cell values",
            diagnostics=diagnostics,
        )
    extra = {"syringe": syringe.value, "cell": cell.value,
             "raw_syringe": syringe.raw, "raw_cell": cell.raw,
             "unit": CANONICAL["concentration"]}
    return (
        MaterialVersion(kind="concentration", version=version, origin=origin,
                        quantity=syringe.value, unit=CANONICAL["concentration"],
                        raw=syringe.raw, extra=extra),
        diagnostics,
    )
