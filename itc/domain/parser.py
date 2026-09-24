"""Raw material import / parsing.

Five separately-versioned material kinds are recognised:

* ``power``        - sampled power curve
* ``schedule``     - injection index / nominal time / volume
* ``concentration``- sample concentrations (cell, syringe, ...)
* ``conditions``   - temperature, cell (pool) volume, ...
* ``baseline``     - baseline curve
* ``blank``        - blank control curve and schedule

Raw representations (original unit strings, original numeric tokens) are
preserved on every object. Normalisation into canonical units happens only in
the derived layer. Unconvertible values stay in place with a positioned
diagnostic.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

from .diagnostics import FORMAT_CORRUPT, Diagnostic, diagnostic
from .units import CANONICAL, convert

PARSED = "PARSED"
WITH_DIAGNOSTICS = "WITH_DIAGNOSTICS"
CORRUPT = "CORRUPT"


@dataclass
class Sample:
    index: int
    raw_time: str
    raw_time_unit: str
    raw_power: str
    raw_power_unit: str
    time_s: float          # canonical; NaN when unconvertible/missing
    power: float           # canonical; NaN when unconvertible/missing
    flags: list[str] = field(default_factory=list)


@dataclass
class PowerSeries:
    kind: str                       # "power" | "baseline" | "blank"
    raw_text: str
    declared_power_unit: str
    n: int
    samples: list[Sample]
    median_dt_s: float
    gaps: list[dict]                # {start_index,end_index,start_time,end_time}
    status: str
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def canonical(self) -> tuple[list[float], list[float]]:
        return ([s.time_s for s in samples], [s.power for s in samples]) if False else (
            [s.time_s for s in self.samples],
            [s.power for s in self.samples],
        )


@dataclass
class Injection:
    index: int
    nominal_time_s: float
    raw_time: str
    raw_time_unit: str
    volume_uL: float
    raw_volume: str
    raw_volume_unit: str
    flags: list[str] = field(default_factory=list)


@dataclass
class MaterialBundle:
    power: PowerSeries | None = None
    baseline: PowerSeries | None = None
    blanks: dict[str, PowerSeries] = field(default_factory=dict)
    injections: list[Injection] = field(default_factory=list)
    schedule_status: str = CORRUPT
    schedule_diagnostics: list[Diagnostic] = field(default_factory=list)
    concentration: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _header_unit(fieldname: str, header_map: dict[str, str]) -> str:
    # e.g. power[ucal/s] or time[s]
    if "[" in fieldname and fieldname.endswith("]"):
        return fieldname[fieldname.index("[") + 1:-1]
    return header_map.get(fieldname.lower(), "")


def _read_rows(raw_text: str) -> tuple[list[str], list[dict], list[Diagnostic]]:
    diags: list[Diagnostic] = []
    text = raw_text.strip("\ufeff")
    try:
        reader = csv.reader(io.StringIO(text))
        rows = [r for r in reader]
    except csv.Error as exc:  # pragma: no cover - defensive
        return [], [], [diagnostic(FORMAT_CORRUPT, f"CSV unreadable: {exc}")]
    if not rows:
        return [], [], [diagnostic(FORMAT_CORRUPT, "empty file",
                                   location={"section": "header"})]
    header = [h.strip() for h in rows[0]]
    out: list[dict] = []
    for i, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue
        if len(row) != len(header):
            diags.append(diagnostic(
                FORMAT_CORRUPT,
                f"column count {len(row)} != header {len(header)}",
                location={"row": i}, raw=",".join(row),
            ))
            continue
        out.append({"_row": i, **{header[j].strip(): row[j].strip()
                                  for j in range(len(header))}})
    return header, out, diags


def _find_field(fields: list[str], *names: str) -> tuple[str | None, str | None]:
    for f in fields:
        base = f.split("[", 1)[0].strip().lower()
        if base in names:
            return f, base
    return None, None


def parse_power(raw_text: str, kind: str = "power") -> PowerSeries:
    header, rows, diags = _read_rows(raw_text)
    if not header:
        return PowerSeries(kind, raw_text, "", 0, [], 0.0, [], CORRUPT, diags)
    header_units: dict[str, str] = {}
    clean_fields: list[str] = []
    for f in header:
        base = f.split("[", 1)[0].strip().lower()
        unit = f[f.index("[") + 1:-1] if "[" in f and f.endswith("]") else ""
        header_units[base] = unit
        clean_fields.append(base)
    tf, _ = _find_field(header, "time", "t")
    pf, _ = _find_field(header, "power", "p")
    if tf is None or pf is None:
        diags.append(diagnostic(
            FORMAT_CORRUPT,
            f"{kind}: require time and power columns, got {header}",
            location={"section": "header"},
        ))
        return PowerSeries(kind, raw_text, header_units.get("power", ""), 0, [],
                           0.0, [], CORRUPT, diags)
    tunit = _header_unit(tf, {"time": "s", "t": "s"})
    if "[" not in tf:
        tunit = "s"
    punit = _header_unit(pf, {})
    if "[" not in pf:
        diags.append(diagnostic(
            FORMAT_CORRUPT,
            f"{kind}: power column must declare a unit e.g. power[ucal/s]",
            location={"column": pf},
        ))
        return PowerSeries(kind, raw_text, punit, 0, [], 0.0, [], CORRUPT, diags)

    samples: list[Sample] = []
    status = PARSED
    for row in rows:
        idx = len(samples)
        rt, rp = row[tf], row[pf]
        flags: list[str] = []
        tval = _num(rt)
        pval = _num(rp)
        if tval is None:
            diags.append(diagnostic(FORMAT_CORRUPT, f"non-numeric time {rt!r}",
                                    location={"row": row["_row"], "column": tf},
                                    raw=rt))
            flags.append("TIME_NONNUMERIC")
        if pval is None:
            diags.append(diagnostic(FORMAT_CORRUPT, f"non-numeric power {rp!r}",
                                    location={"row": row["_row"], "column": pf},
                                    raw=rp))
            flags.append("POWER_NONNUMERIC")
        ct = math.nan
        cp = math.nan
        if tval is not None:
            cr = convert(tval, tunit, "time")
            if cr.ok:
                ct = cr.value
            else:
                flags.append("TIME_UNCONVERTIBLE")
                status = WITH_DIAGNOSTICS
                diags.append(diagnostic(cr.diagnostic_code, cr.message,
                                        location={"row": row["_row"]}, raw=rt))
        if pval is not None:
            cr = convert(pval, punit, "power")
            if cr.ok:
                cp = cr.value
            else:
                flags.append("POWER_UNCONVERTIBLE")
                status = WITH_DIAGNOSTICS
                diags.append(diagnostic(cr.diagnostic_code, cr.message,
                                        location={"row": row["_row"]}, raw=rp))
        samples.append(Sample(idx, rt, tunit, rp, punit, ct, cp, flags))

    # ordering / median dt / gap detection on canonical time axis
    gaps: list[dict] = []
    dts = [samples[i].time_s - samples[i - 1].time_s
           for i in range(1, len(samples))
           if not math.isnan(samples[i].time_s)
           and not math.isnan(samples[i - 1].time_s)
           and samples[i].time_s > samples[i - 1].time_s]
    if dts:
        dts_sorted = sorted(dts)
        median_dt = dts_sorted[len(dts_sorted) // 2]
    else:
        median_dt = 0.0
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if math.isnan(a.time_s) or math.isnan(b.time_s):
            continue
        if median_dt > 0 and (b.time_s - a.time_s) > 2.5 * median_dt:
            gaps.append({"start_index": i - 1, "end_index": i,
                         "start_time": a.time_s, "end_time": b.time_s})
    if diags and status == PARSED:
        status = WITH_DIAGNOSTICS
    if any(d.code == FORMAT_CORRUPT for d in diags):
        status = CORRUPT
    return PowerSeries(kind, raw_text, punit, len(samples), samples,
                       median_dt, gaps, status, diags)


def _num(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def parse_schedule(raw_text: str) -> tuple[list[Injection], str, list[Diagnostic]]:
    header, rows, diags = _read_rows(raw_text)
    if not header:
        return [], CORRUPT, diags
    idx_f, _ = _find_field(header, "index", "inj", "injection")
    tf, _ = _find_field(header, "time", "nominal_time", "t")
    vf, _ = _find_field(header, "volume", "vol", "v")
    if not (idx_f and tf and vf):
        diags.append(diagnostic(
            FORMAT_CORRUPT,
            f"schedule requires index,time,volume columns, got {header}",
            location={"section": "header"}))
        return [], CORRUPT, diags
    tunit = tf[tf.index("[") + 1:-1] if "[" in tf else "s"
    vunit = vf[vf.index("[") + 1:-1] if "[" in vf else "uL"
    injections: list[Injection] = []
    status = PARSED
    for row in rows:
        ri, rt, rv = row[idx_f], row[tf], row[vf]
        iv, tv, vv = _num(ri), _num(rt), _num(rv)
        flags: list[str] = []
        if iv is None or tv is None or vv is None:
            diags.append(diagnostic(FORMAT_CORRUPT,
                                    "non-numeric schedule value",
                                    location={"row": row["_row"]},
                                    raw=(ri, rt, rv)))
            flags.append("NONNUMERIC")
            status = CORRUPT
            continue
        tc = convert(tv, tunit, "time")
        vc = convert(vv, vunit, "volume")
        t_can = tc.value if tc.ok else math.nan
        v_can = vc.value if vc.ok else math.nan
        if not tc.ok:
            flags.append("TIME_UNCONVERTIBLE"); status = WITH_DIAGNOSTICS
            diags.append(diagnostic(tc.diagnostic_code, tc.message,
                                    location={"row": row["_row"]}, raw=rt))
        if not vc.ok:
            flags.append("VOLUME_UNCONVERTIBLE"); status = WITH_DIAGNOSTICS
            diags.append(diagnostic(vc.diagnostic_code, vc.message,
                                    location={"row": row["_row"]}, raw=rv))
        injections.append(Injection(int(iv), t_can, rt, tunit, v_can, rv,
                                    vunit, flags))
    injections.sort(key=lambda x: (x.nominal_time_s
                                   if not math.isnan(x.nominal_time_s)
                                   else float("inf"), x.index))
    if any(d.code == FORMAT_CORRUPT for d in diags):
        status = CORRUPT
    elif diags and status == PARSED:
        status = WITH_DIAGNOSTICS
    return injections, status, diags


def _parse_keyvalue(raw_text: str, section: str
                    ) -> tuple[dict, str, list[Diagnostic]]:
    """Parse `name,value,unit` tables (concentrations / conditions)."""
    header, rows, diags = _read_rows(raw_text)
    if not header:
        return {}, CORRUPT, diags
    nf, _ = _find_field(header, "name", "key", "parameter", "role")
    vf, _ = _find_field(header, "value", "val")
    uf, _ = _find_field(header, "unit")
    if not (nf and vf):
        diags.append(diagnostic(FORMAT_CORRUPT,
                                f"{section} requires name,value[,unit]",
                                location={"section": "header"}))
        return {}, CORRUPT, diags
    entries: dict[str, dict] = {}
    status = PARSED
    for row in rows:
        name, rval = row[nf], row[vf]
        runit = row[uf] if uf else ""
        val = _num(rval)
        entry = {"name": name, "raw_value": rval, "raw_unit": runit,
                 "value": math.nan, "canonical_unit": None,
                 "convertible": False, "row": row["_row"], "diagnostics": []}
        if val is None:
            d = diagnostic(FORMAT_CORRUPT, f"non-numeric {section} value",
                           location={"row": row["_row"], "name": name}, raw=rval)
            diags.append(d); entry["diagnostics"].append(d.to_dict())
            entries[name] = entry
            status = CORRUPT
            continue
        entry["value"] = val  # raw numeric retained
        entries[name] = entry
    if any(d.code == FORMAT_CORRUPT for d in diags):
        status = CORRUPT
    elif diags:
        status = WITH_DIAGNOSTICS
    return entries, status, diags


def normalise_keyvalues(entries: dict, kind_map: dict[str, str]
                        ) -> list[Diagnostic]:
    """Apply unit conversion for entries whose kind is known (derived layer)."""
    diags: list[Diagnostic] = []
    for name, e in entries.items():
        kind = kind_map.get(name.lower())
        if kind is None:
            continue
        cr = convert(e["value"], e["raw_unit"], kind)
        if cr.ok:
            e["canonical_value"] = cr.value
            e["canonical_unit"] = CANONICAL[kind]
            e["convertible"] = True
        else:
            e["canonical_value"] = None
            e["convertible"] = False
            d = diagnostic(cr.diagnostic_code, cr.message,
                           location={"row": e["row"], "name": name},
                           raw=f"{e['raw_value']} {e['raw_unit']}")
            diags.append(d); e["diagnostics"].append(d.to_dict())
    return diags
