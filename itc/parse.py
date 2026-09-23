"""Parsers for the ITC lab's own text representations.

This is deliberately *not* a generic CSV/table importer: the files are
domain artifacts (``.itcpower`` etc.) with explicit sections, units and
coordinate bases.  Everything typed is kept as raw text; conversion to
canonical units happens in Measured values alongside the raw representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic, FormatCorrupt, ResourceExhausted, SemanticUnresolvable
from .units import make_measured, parse_temperature

SECTION_KINDS = {
    "@power": "power",
    "@baseline": "baseline",
    "@blank": "blank",
    "@injections": "injections",
    "@concentrations": "concentrations",
    "@conditions": "conditions",
}

CANON_POINT_KINDS = {"power", "baseline", "blank"}


@dataclass
class Budgets:
    max_points: int = 200_000
    max_rows: int = 10_000


def _diag(code, msg, file, line, target=None, severity="error") -> dict:
    return Diagnostic(code, msg, file, line, None, target, severity).to_dict()


def parse_material(text: str, filename: str, budgets: Budgets | None = None) -> dict[str, Any]:
    """Parse one material file. Raises FormatCorrupt/ResourceExhausted;
    semantic issues are returned as attached diagnostics."""
    budgets = budgets or Budgets()
    kind: str | None = None
    meta: dict[str, str] = {}
    points: list[dict] = []
    gaps: list[dict] = []
    injections: list[dict] = []
    scalars: list[dict] = []
    diagnostics: list[dict] = []
    in_data = False
    point_count = 0
    last_t: float | None = None
    seen_sections: set[str] = set()
    lines = text.splitlines()

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            token = line.split(None, 1)[0].lower()
            if token not in SECTION_KINDS:
                raise FormatCorrupt(
                    f"unknown section header {token!r}",
                    [Diagnostic("unknown_section", f"unknown section {token!r}",
                                filename, idx)])
            if kind is not None:
                raise FormatCorrupt(
                    "multiple sections in one material file",
                    [Diagnostic("multiple_sections",
                                f"second section {token!r} after @{kind}", filename, idx)])
            kind = SECTION_KINDS[token]
            seen_sections.add(kind)
            in_data = False
            if " " in line:
                inline = line.split(None, 1)[1].strip()
                if inline:
                    meta["section_arg"] = inline
            continue
        if kind is None:
            raise FormatCorrupt(
                "content before any @section header",
                [Diagnostic("preamble", "data outside of a section", filename, idx)])
        if line.lower() == "data":
            in_data = True
            continue
        if in_data:
            parts = line.replace(",", " ").split()
            if parts and parts[0].lower() == "gap":
                if len(parts) != 3:
                    raise FormatCorrupt(
                        "GAP needs exactly two bounds",
                        [Diagnostic("bad_gap", "GAP start end expected", filename, idx)])
                g0 = make_measured(parts[1], meta.get("time_unit"), "time",
                                   file=filename, line=idx, target="gap")
                g1 = make_measured(parts[2], meta.get("time_unit"), "time",
                                   file=filename, line=idx, target="gap")
                gd = [d.to_dict() for v in (g0, g1) for d in v.diagnostics]
                gaps.append({"start": _m(g0), "end": _m(g1),
                             "raw": {"text": line, "line": idx}, "diagnostics": gd})
                if g0.value is not None and g1.value is not None and g1.value <= g0.value:
                    diagnostics.append(_diag("gap_reversed",
                                            "gap end must be greater than gap start",
                                            filename, idx, "gap"))
                continue
            if kind in CANON_POINT_KINDS:
                if len(parts) != 2:
                    raise FormatCorrupt(
                        f"{kind} data rows need two columns",
                        [Diagnostic("bad_row", f"expected 't p', got {line!r}",
                                    filename, idx)])
                point_count += 1
                if point_count > budgets.max_points:
                    raise ResourceExhausted(
                        f"point budget {budgets.max_points} exceeded",
                        [Diagnostic("points_budget",
                                    f"more than {budgets.max_points} points",
                                    filename, idx, severity="error")])
                t = make_measured(parts[0], meta.get("time_unit"), "time",
                                  file=filename, line=idx, target=f"point:{point_count}")
                p = make_measured(parts[1], meta.get("power_unit"), "power",
                                  file=filename, line=idx, target=f"point:{point_count}")
                pd = [d.to_dict() for v in (t, p) for d in v.diagnostics]
                tv = t.value
                if tv is not None:
                    if last_t is not None and tv < last_t:
                        diagnostics.append(_diag(
                            "time_not_monotonic",
                            "power sample time must be non-decreasing",
                            filename, idx, f"point:{point_count}"))
                    last_t = tv
                points.append({"seq": point_count, "t": _m(t), "p": _m(p),
                               "raw": {"line": idx, "text": line},
                               "diagnostics": pd})
            elif kind == "injections":
                if len(parts) < 3:
                    raise FormatCorrupt(
                        "injection rows need ordinal, time, volume",
                        [Diagnostic("bad_row", "expected 'ordinal time volume'",
                                    filename, idx)])
                point_count += 1
                if point_count > budgets.max_rows:
                    raise ResourceExhausted(
                        f"injection row budget {budgets.max_rows} exceeded",
                        [Diagnostic("rows_budget", "too many injections", filename, idx)])
                try:
                    ordinal = int(parts[0])
                except ValueError:
                    raise FormatCorrupt(
                        "injection ordinal must be an integer",
                        [Diagnostic("bad_ordinal", f"{parts[0]!r} not integer",
                                    filename, idx)])
                t = make_measured(parts[1], meta.get("time_unit"), "time",
                                  file=filename, line=idx, target=f"injection:{ordinal}")
                v = make_measured(parts[2], meta.get("volume_unit"), "volume",
                                  file=filename, line=idx, target=f"injection:{ordinal}")
                row_diag = [d.to_dict() for x in (t, v) for d in x.diagnostics]
                if v.value is not None and v.value <= 0:
                    row_diag.append(_diag("nonpositive_volume",
                                          "injection volume must be positive",
                                          filename, idx, f"injection:{ordinal}"))
                injections.append({"ordinal": ordinal, "time": _m(t), "volume": _m(v),
                                   "raw": {"line": idx, "text": line},
                                   "diagnostics": row_diag})
            else:
                raise FormatCorrupt(
                    f"DATA block not allowed for @{kind}",
                    [Diagnostic("bad_data", f"@{kind} takes key:value entries",
                                filename, idx)])
            continue
        # metadata line
        if ":" not in line:
            raise FormatCorrupt(
                f"unparseable line {line!r}",
                [Diagnostic("bad_line", "expected key: value or DATA row",
                            filename, idx)])
        key, _, val = line.partition(":")
        meta[key.strip().lower()] = val.strip()

    if kind is None:
        raise FormatCorrupt(
            "file contains no @section",
            [Diagnostic("no_section", "expected @power/@injections/...", filename, 1)])

    if kind == "injections":
        ordinals = [r["ordinal"] for r in injections]
        if len(set(ordinals)) != len(ordinals):
            diagnostics.append(_diag("duplicate_ordinal",
                                     "injection ordinals must be unique", filename, None,
                                     "injections"))
        if ordinals != sorted(ordinals):
            diagnostics.append(_diag("ordinals_unsorted",
                                     "injection rows must be ordered by ordinal",
                                     filename, None, "injections"))
        prev = None
        for r in injections:
            tv = r["time"]["canonical"]
            tvv = None if tv is None else tv["value"]
            if tvv is not None and prev is not None and tvv <= prev:
                diagnostics.append(_diag("injection_time_not_increasing",
                                         "injection times must be strictly increasing",
                                         None, None, f"injection:{r['ordinal']}"))
            if tvv is not None:
                prev = tvv
    if kind == "concentrations":
        for role in ("macromolecule", "ligand"):
            if role not in meta:
                diagnostics.append(_diag("concentration_role_missing",
                                         f"{role} concentration is required",
                                         filename, None, f"concentration:{role}"))
        for key, val in meta.items():
            if key in ("name", "note", "section_arg"):
                continue
            tokens = val.replace(",", " ").split()
            if len(tokens) < 2:
                diagnostics.append(_diag("scalar_without_unit",
                                         f"{key} must be 'value unit'", filename, None,
                                         f"concentration:{key}"))
                continue
            mv = make_measured(tokens[0], " ".join(tokens[1:]), "concentration",
                               file=filename, target=f"concentration:{key}")
            scalars.append({"role": key, "measured": _m(mv),
                            "raw": {"text": val}})
    if kind == "conditions":
        for key, ukind, parser in (
                ("temperature", "temperature", parse_temperature),
                ("cell_volume", "volume", lambda t, u, **k: make_measured(t, u, "volume", **k))):
            if key not in meta:
                diagnostics.append(_diag("condition_missing",
                                         f"{key} is required", filename, None, key))
                continue
            tokens = meta[key].replace(",", " ").split()
            if len(tokens) < 2:
                diagnostics.append(_diag("scalar_without_unit",
                                         f"{key} must be 'value unit'", filename, None, key))
                continue
            mv = parser(tokens[0], " ".join(tokens[1:]), file=filename, target=key)
            scalars.append({"role": key, "measured": _m(mv),
                            "raw": {"text": meta[key]}})

    material = {
        "kind": kind,
        "name": meta.get("name") or meta.get("section_arg") or filename,
        "coordinate_basis": meta.get("coordinate_basis", "t_seconds_from_acquisition_start"),
        "units": {k: v for k, v in meta.items() if k.endswith("_unit")},
        "meta": meta,
        "diagnostics": diagnostics,
    }
    if kind in CANON_POINT_KINDS:
        material.update({
            "points": points,
            "gaps": gaps,
            "baseline_ref": meta.get("baseline_ref"),
            "role": meta.get("role", "control" if kind == "blank" else "sample"),
        })
    elif kind == "injections":
        material["injections"] = injections
    elif kind == "concentrations":
        material["scalars"] = scalars
    elif kind == "conditions":
        material["scalars"] = scalars
    return material


def _m(mv) -> dict:
    """Measured -> wire dict."""
    return mv.to_dict()
