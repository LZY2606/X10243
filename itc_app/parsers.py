"""Parsers for the offline ITC domain file formats.

Domain formats (no generic spreadsheets):
  .itcpower     power sampling trace
  .itcinj       injection times + volumes
  .itcconc      ligand / macromolecule concentrations
  .itctemp      temperature reference
  .itccell      cell volume
  .itcbase      baseline trace (its own version line)
  .itcblank     blank control trace + blank injection schedule

Every input token is retained verbatim (``raw_*``).  Tokens that cannot be
converted stay on their row with a diagnostic; parsing never silently drops a
sample.  Structural damage raises FormatCorrupt; exceeding the sample budget
raises ResourceBudgetExhausted.
"""
from __future__ import annotations

from .errors import FormatCorrupt, ResourceBudgetExhausted, diag
from .models import Injection, Sample, Trace
from .units import parse_quantity

MAGIC = {
    ".itcpower": "ITC-POWER/1",
    ".itcinj": "ITC-INJECTIONS/1",
    ".itcconc": "ITC-CONCENTRATION/1",
    ".itctemp": "ITC-TEMPERATURE/1",
    ".itccell": "ITC-CELL/1",
    ".itcbase": "ITC-BASELINE/1",
    ".itcblank": "ITC-BLANK/1",
}

MISSING_TOKENS = {"na", "n/a", "--", "-", "nan", "?", "null", "none"}


def _lines(content: str):
    for lineno, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield lineno, raw_line, stripped


def _header(content: str, magic: str):
    """Return (headers dict, sections dict[name] -> list[(lineno,line)])."""
    first = True
    headers: dict[str, str] = {}
    sections: dict[str, list[tuple[int, str]]] = {}
    current = "_header"
    sections[current] = []
    saw_magic = False
    for lineno, raw_line, stripped in _lines(content):
        if first:
            if stripped.replace(" ", "") != magic.replace(" ", ""):
                raise FormatCorrupt(
                    f"expected magic header {magic!r}, found {stripped[:40]!r}",
                    [diag("bad_magic", "file does not start with the ITC domain "
                          f"magic {magic}", f"line {lineno}", raw=stripped[:80])],
                )
            saw_magic = True
            first = False
            continue
        first = False
        if stripped.startswith("@"):
            current = stripped[1:].strip().lower()
            sections.setdefault(current, [])
            continue
        sections[current].append((lineno, raw_line))
        if current == "_header" and ":" in stripped:
            key, _, value = stripped.partition(":")
            headers[key.strip().lower()] = value.strip()
    if not saw_magic:
        raise FormatCorrupt("empty file", [diag("empty", "no content")])
    return headers, sections


def _is_missing(token: str) -> bool:
    return token.strip().lower() in MISSING_TOKENS


def _parse_trace(headers, sections, max_samples: int, *, require_section="data"):
    time_unit = headers.get("time-unit", "s")
    value_unit = headers.get("power-unit", headers.get("value-unit", "uW"))
    origin = headers.get("time-origin", "t0")
    rows = sections.get(require_section)
    if rows is None:
        raise FormatCorrupt(
            f"missing @{require_section} section",
            [diag("missing_section", f"trace needs a @{require_section} section",
                  "sections")],
        )
    diagnostics: list[dict] = []
    samples: list[Sample] = []
    gaps: list[tuple[float | None, float | None]] = []
    valid_lo = headers.get("valid-from")
    valid_hi = headers.get("valid-to")

    def _bound(raw):
        if raw is None:
            return None
        q = parse_quantity(raw, "time")
        return q.si_value

    valid_range = (_bound(valid_lo), _bound(valid_hi))

    prev_time: float | None = None
    for lineno, raw_line in rows:
        if len(samples) >= max_samples:
            raise ResourceBudgetExhausted(
                f"sample budget {max_samples} exceeded at line {lineno}",
                [diag("budget", f"more than {max_samples} samples", f"line {lineno}",
                      raw=raw_line.strip()[:80],
                      suggestion="raise the parser sample budget or truncate input")],
            )
        parts = raw_line.split()
        if len(parts) != 2:
            raise FormatCorrupt(
                f"expected 2 columns at line {lineno}",
                [diag("bad_columns", "trace rows need exactly <time> <power>",
                      f"line {lineno}", raw=raw_line.strip()[:80])],
            )
        idx = len(samples)
        raw_t, raw_v = parts
        t_missing, v_missing = _is_missing(raw_t), _is_missing(raw_v)
        qt = parse_quantity(raw_t, "time", default_unit=time_unit)
        qv = parse_quantity(raw_v, "power", default_unit=value_unit)
        s = Sample(
            index=idx, raw_time=raw_t, raw_value=raw_v,
            time_si=None if t_missing else qt.si_value,
            value_si=None if v_missing else qv.si_value,
            time_unit=qt.unit or time_unit,
            value_unit=qv.unit or value_unit,
        )
        row_diag = None
        if t_missing:
            row_diag = diag("gap_time", "missing time sample; treated as a data gap",
                            f"line {lineno} col time", raw=raw_t,
                            suggestion="interpolation is not allowed across this point")
        elif qt.si_value is None:
            row_diag = diag("unconvertible_time",
                            f"cannot convert time token {raw_t!r} to seconds",
                            f"line {lineno} col time", raw=raw_t)
        if v_missing:
            d = diag("gap_power", "missing power sample; treated as a data gap",
                     f"line {lineno} col power", raw=raw_v,
                     suggestion="integration windows must not cross this point")
            s.is_gap = True
            diagnostics.append(d.to_dict())
        elif qv.si_value is None:
            d = diag("unconvertible_power",
                     f"cannot convert power token {raw_v!r} to watts; "
                     "retained in place, not dropped",
                     f"line {lineno} col power", raw=raw_v)
            diagnostics.append(d.to_dict())
        if row_diag is not None:
            s.diagnostic = row_diag.to_dict()
            diagnostics.append(row_diag.to_dict())
            if t_missing:
                s.is_gap = True
        if s.time_si is not None:
            if prev_time is not None and s.time_si < prev_time:
                diagnostics.append(diag(
                    "non_monotonic_time", "sample time is earlier than previous row",
                    f"line {lineno} col time", severity="warning", raw=raw_t).to_dict())
            prev_time = s.time_si
        samples.append(s)

    # Gap intervals: a missing power datum makes the interval between its
    # bracketing times unusable for integration.
    for s in samples:
        if s.is_gap and s.time_si is not None:
            gaps.append((s.time_si, s.time_si))
    trace = Trace(samples=samples, origin=origin, gaps=gaps,
                  diagnostics=diagnostics, valid_range=valid_range)
    return trace, {"time_unit": time_unit, "power_unit": value_unit,
                   "origin": origin, "headers": dict(headers)}


def parse_power(content: str, max_samples: int = 200_000) -> tuple[Trace, dict]:
    headers, sections = _header(content, MAGIC[".itcpower"])
    return _parse_trace(headers, sections, max_samples)


def parse_baseline(content: str, max_samples: int = 200_000):
    headers, sections = _header(content, MAGIC[".itcbase"])
    return _parse_trace(headers, sections, max_samples)


def parse_blank(content: str, max_samples: int = 200_000):
    headers, sections = _header(content, MAGIC[".itcblank"])
    trace, meta = _parse_trace(headers, sections, max_samples)
    injections = _parse_injection_rows(sections.get("injections", []), headers)
    meta["blank_injections"] = [i.to_dict() for i in injections]
    return trace, meta


def _parse_injection_rows(rows, headers):
    time_unit = headers.get("time-unit", "s")
    volume_unit = headers.get("volume-unit", "uL")
    injections: list[Injection] = []
    for lineno, raw_line in rows:
        parts = raw_line.split()
        if len(parts) != 3:
            raise FormatCorrupt(
                f"expected 3 columns at line {lineno}",
                [diag("bad_columns", "injection rows need <index> <time> <volume>",
                      f"line {lineno}", raw=raw_line.strip()[:80])],
            )
        raw_i, raw_t, raw_v = parts
        try:
            index = int(raw_i)
        except ValueError:
            raise FormatCorrupt(
                f"bad injection index at line {lineno}",
                [diag("bad_index", "injection index must be an integer",
                      f"line {lineno}", raw=raw_i)],
            )
        qt = parse_quantity(raw_t, "time", default_unit=time_unit)
        qv = parse_quantity(raw_v, "volume", default_unit=volume_unit)
        inj = Injection(index=index,
                        time_si=None if _is_missing(raw_t) else qt.si_value,
                        volume_si=None if _is_missing(raw_v) else qv.si_value,
                        raw_time=raw_t, raw_volume=raw_v,
                        time_unit=qt.unit or time_unit,
                        volume_unit=qv.unit or volume_unit)
        row_diags = []
        if inj.time_si is None:
            row_diags.append(diag("unconvertible_time",
                                  f"cannot convert injection time {raw_t!r}",
                                  f"line {lineno} col time", raw=raw_t))
        if inj.volume_si is None:
            row_diags.append(diag("unconvertible_volume",
                                  f"cannot convert injection volume {raw_v!r}",
                                  f"line {lineno} col volume", raw=raw_v))
        if row_diags:
            inj.convertible = False
            inj.diagnostic = row_diags[0].to_dict()
        injections.append(inj)
    return injections


def parse_injections(content: str):
    headers, sections = _header(content, MAGIC[".itcinj"])
    rows = sections.get("injections")
    if rows is None:
        raise FormatCorrupt("missing @injections section",
                            [diag("missing_section", "need @injections", "sections")])
    injections = _parse_injection_rows(rows, headers)
    return injections, {"time_unit": headers.get("time-unit", "s"),
                        "volume_unit": headers.get("volume-unit", "uL"),
                        "headers": dict(headers)}


def parse_concentration(content: str):
    headers, sections = _header(content, MAGIC[".itcconc"])
    raw_pairs = {}
    for lineno, raw_line in sections["_header"]:
        if ":" in raw_line:
            k, _, v = raw_line.partition(":")
            raw_pairs[k.strip()] = v.strip()
    ligand = headers.get("ligand_syringe", headers.get("ligand"))
    macro = headers.get("macromolecule_cell", headers.get("macromolecule"))
    ligand_unit = headers.get("ligand-unit", "mM")
    macro_unit = headers.get("macromolecule-unit", "uM")
    parsed: dict = {"raw_pairs": raw_pairs}
    diagnostics = []
    if ligand is not None:
        q = parse_quantity(f"{ligand} {ligand_unit}", "concentration")
        parsed["ligand_syringe_si"] = q.si_value
        parsed["ligand_syringe_raw"] = ligand
        if q.si_value is None:
            diagnostics.append(diag("unconvertible_concentration",
                                    f"cannot convert ligand concentration {ligand!r}",
                                    "header ligand_syringe", raw=ligand).to_dict())
    else:
        diagnostics.append(diag("missing_ligand", "ligand_syringe not declared",
                                "header").to_dict())
    if macro is not None:
        q = parse_quantity(f"{macro} {macro_unit}", "concentration")
        parsed["macromolecule_cell_si"] = q.si_value
        parsed["macromolecule_cell_raw"] = macro
        if q.si_value is None:
            diagnostics.append(diag("unconvertible_concentration",
                                    "cannot convert macromolecule concentration",
                                    "header macromolecule_cell", raw=macro).to_dict())
    else:
        diagnostics.append(diag("missing_macromolecule",
                                "macromolecule_cell not declared", "header").to_dict())
    parsed["ligand_unit"] = ligand_unit
    parsed["macromolecule_unit"] = macro_unit
    return parsed, diagnostics


def parse_temperature(content: str):
    headers, _ = _header(content, MAGIC[".itctemp"])
    raw_t = headers.get("temperature")
    unit = headers.get("unit", "C")
    diagnostics = []
    si = None
    if raw_t is None:
        diagnostics.append(diag("missing_temperature", "temperature not declared",
                                "header").to_dict())
    else:
        q = parse_quantity(f"{raw_t} {unit}", "temperature")
        si = q.si_value
        if si is None:
            diagnostics.append(diag("unconvertible_temperature",
                                    f"cannot convert temperature {raw_t!r} {unit!r}",
                                    "header temperature", raw=raw_t).to_dict())
    return {"temperature_si": si, "raw": raw_t, "unit": unit}, diagnostics


def parse_cell_volume(content: str):
    headers, _ = _header(content, MAGIC[".itccell"])
    raw_v = headers.get("cell-volume", headers.get("cell_volume"))
    unit = headers.get("unit", "mL")
    diagnostics = []
    si = None
    if raw_v is None:
        diagnostics.append(diag("missing_cell_volume", "cell-volume not declared",
                                "header").to_dict())
    else:
        q = parse_quantity(f"{raw_v} {unit}", "volume")
        si = q.si_value
        if si is None:
            diagnostics.append(diag("unconvertible_volume",
                                    f"cannot convert cell volume {raw_v!r} {unit!r}",
                                    "header cell-volume", raw=raw_v).to_dict())
    return {"cell_volume_si": si, "raw": raw_v, "unit": unit}, diagnostics


PARSERS_BY_EXT = {
    ".itcpower": "power",
    ".itcinj": "injections",
    ".itcconc": "concentration",
    ".itctemp": "temperature",
    ".itccell": "cell_volume",
    ".itcbase": "baseline",
    ".itcblank": "blank",
}
