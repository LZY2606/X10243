"""Immutable ITC source materials and the .itcrun domain parser.

Seven independently versioned material slots:
power, injections, concentrations, temperature, cell_volume, baseline, blank.

Raw text representations are preserved on every value; normalisation happens
only in the derivation layer. Parse failures are *positioned* and fall into
three distinct classes (FormatCorrupt / SemanticNoSolution / BudgetExhausted
is produced later by the fitter).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

from .units import Quantity, parse_quantity, Diagnostic

SLOT_POWER = "power"
SLOT_INJECTIONS = "injections"
SLOT_CONCENTRATIONS = "concentrations"
SLOT_TEMPERATURE = "temperature"
SLOT_CELL_VOLUME = "cell_volume"
SLOT_BASELINE = "baseline"
SLOT_BLANK = "blank"
MATERIAL_SLOTS = (
    SLOT_POWER,
    SLOT_INJECTIONS,
    SLOT_CONCENTRATIONS,
    SLOT_TEMPERATURE,
    SLOT_CELL_VOLUME,
    SLOT_BASELINE,
    SLOT_BLANK,
)


class FormatCorrupt(Exception):
    """Syntax-level failure; always carries a position."""

    def __init__(self, message: str, position: str):
        super().__init__(f"{position}: {message}")
        self.message = message
        self.position = position


class SemanticNoSolution(Exception):
    """Well-formed input for which no legal domain solution exists."""

    def __init__(self, message: str, code: str = "E_SEMANTIC", position: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.position = position


class BudgetExhausted(Exception):
    """Computation used its full resource budget without converging."""

    def __init__(self, message: str, budget: int, code: str = "E_BUDGET"):
        super().__init__(message)
        self.message = message
        self.budget = budget
        self.code = code


@dataclass
class Sample:
    t: Quantity
    p: Quantity
    line: int  # 1-based line in the source file

    def to_dict(self) -> dict:
        return {"t": self.t.to_dict(), "p": self.p.to_dict(), "line": self.line}


@dataclass
class Gap:
    start: Quantity
    end: Quantity
    line: int

    def to_dict(self) -> dict:
        return {"start": self.start.to_dict(), "end": self.end.to_dict(), "line": self.line}


@dataclass
class Curve:
    """A sampled curve (run power, baseline or blank) with raw strings."""

    unit: str
    time_unit: str
    source: str
    samples: list[Sample] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    # blank-only: injection times against which dilution heat is matched
    injection_times: list[Quantity] = field(default_factory=list)

    def content_str(self) -> str:
        rows = [f"{s.t.raw} {s.p.raw}" for s in self.samples]
        gaps = [f"@gap {g.start.raw} {g.end.raw}" for g in self.gaps]
        inj = [f"@blank_injection {q.raw}" for q in self.injection_times]
        return "\n".join([self.unit, self.time_unit, self.source, *rows, *gaps, *inj])

    def to_dict(self) -> dict:
        return {
            "unit": self.unit,
            "time_unit": self.time_unit,
            "source": self.source,
            "samples": [s.to_dict() for s in self.samples],
            "gaps": [g.to_dict() for g in self.gaps],
            "injection_times": [q.to_dict() for q in self.injection_times],
        }


@dataclass
class Injection:
    index: int
    time: Quantity
    volume: Quantity
    blank_ref: int | None = None  # declared correspondence (index into blank)
    line: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "time": self.time.to_dict(),
            "volume": self.volume.to_dict(),
            "blank_ref": self.blank_ref,
            "line": self.line,
        }


@dataclass
class MaterialSet:
    power: Curve | None = None
    baseline: Curve | None = None
    blank: Curve | None = None
    injections: list[Injection] = field(default_factory=list)
    cell_conc: Quantity | None = None
    syringe_conc: Quantity | None = None
    temperature: Quantity | None = None
    cell_volume: Quantity | None = None
    blank_rule: str = "index"  # index | nearest | manual
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def slot_content(self, slot: str) -> str:
        if slot in (SLOT_POWER, SLOT_BASELINE, SLOT_BLANK):
            curve = {SLOT_POWER: self.power, SLOT_BASELINE: self.baseline, SLOT_BLANK: self.blank}[slot]
            return "" if curve is None else curve.content_str()
        if slot == SLOT_INJECTIONS:
            parts = [f"rule={self.blank_rule}"]
            for inj in self.injections:
                parts.append(
                    f"{inj.index} {inj.time.raw} {inj.time.unit} {inj.volume.raw} {inj.volume.unit} "
                    f"{'' if inj.blank_ref is None else inj.blank_ref}"
                )
            return "\n".join(parts)
        if slot == SLOT_CONCENTRATIONS:
            return f"cell={_q(self.cell_conc)} syringe={_q(self.syringe_conc)}"
        if slot == SLOT_TEMPERATURE:
            return _q(self.temperature)
        if slot == SLOT_CELL_VOLUME:
            return _q(self.cell_volume)
        raise KeyError(slot)


def _q(q: Quantity | None) -> str:
    return "" if q is None else f"{q.raw} {q.unit}"


def material_id(slot: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"mat:{slot}:{digest}"


@dataclass
class ParsedRun:
    materials: MaterialSet
    material_ids: dict[str, str]


def _tokens(line: str) -> list[str]:
    return line.split()


def parse_itcrun(text: str) -> ParsedRun:
    """Parse a .itcrun document. Raises FormatCorrupt on syntax problems."""
    mat = MaterialSet()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "ITC-RUN/1":
        raise FormatCorrupt("first line must be 'ITC-RUN/1'", "line 1")
    section: str | None = None
    curve_unit: dict[str, tuple[str, str, str]] = {}
    i = 1
    while i < len(lines):
        lineno = i + 1
        raw_line = lines[i]
        line = raw_line.strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        pos = f"line {lineno}"
        if line.startswith("@"):
            header = _tokens(line)
            tag = header[0]
            if tag == "@temperature" and len(header) == 3:
                mat.temperature = parse_quantity(header[1], header[2], "temperature", pos)
            elif tag == "@cell_volume" and len(header) == 3:
                mat.cell_volume = parse_quantity(header[1], header[2], "volume", pos)
            elif tag == "@cell_concentration" and len(header) == 3:
                mat.cell_conc = parse_quantity(header[1], header[2], "concentration", pos)
            elif tag == "@syringe_concentration" and len(header) == 3:
                mat.syringe_conc = parse_quantity(header[1], header[2], "concentration", pos)
            elif tag == "@blank_rule" and len(header) == 2:
                mode = header[1].split("=", 1)[-1]
                if mode not in ("index", "nearest", "manual"):
                    raise FormatCorrupt(f"unknown blank correspondence rule {mode!r}", pos)
                mat.blank_rule = mode
            elif tag in ("@power", "@baseline", "@blank"):
                if len(header) != 3:
                    raise FormatCorrupt(f"{tag} needs <power_unit> time=<time_unit>", pos)
                pu, tu = header[1], header[2]
                if not tu.startswith("time="):
                    raise FormatCorrupt("second token must be time=<unit>", pos)
                section = tag[1:]
                curve_unit[section] = (pu, tu[len("time="):], f"{section}@{lineno}")
            elif tag == "@injections":
                section = "injections"
            elif tag == "@end":
                section = None
            elif tag == "@gap" and section in ("power", "baseline", "blank"):
                if len(header) != 3:
                    raise FormatCorrupt("@gap needs <start> <end>", pos)
                pu, tu, _ = curve_unit[section]
                start = parse_quantity(header[1], tu, "time", pos)
                end = parse_quantity(header[2], tu, "time", pos)
                _curve_of(mat, section).gaps.append(Gap(start, end, lineno))
            elif tag == "@blank_injection" and section == "blank":
                if len(header) != 2:
                    raise FormatCorrupt("@blank_injection needs <time>", pos)
                _, tu, _ = curve_unit["blank"]
                _curve_of(mat, "blank").injection_times.append(
                    parse_quantity(header[1], tu, "time", pos)
                )
            else:
                raise FormatCorrupt(f"unrecognised declaration {line!r}", pos)
            continue
        if section in ("power", "baseline", "blank"):
            parts = _tokens(line)
            if len(parts) != 2:
                raise FormatCorrupt(
                    f"sample row must contain exactly '<time> <power>', got {len(parts)} tokens",
                    pos,
                )
            pu, tu, _ = curve_unit[section]
            tq = parse_quantity(parts[0], tu, "time", f"{pos} col 1")
            pq = parse_quantity(parts[1], pu, "power", f"{pos} col 2")
            _curve_of(mat, section).samples.append(Sample(tq, pq, lineno))
        elif section == "injections":
            parts = _tokens(line)
            if len(parts) not in (5, 6):
                raise FormatCorrupt(
                    "injection row must be '<index> <time> <time_unit> <volume> <volume_unit> [blank_ref]'",
                    pos,
                )
            idx: Any
            try:
                idx = int(parts[0])
            except ValueError:
                raise FormatCorrupt(f"injection index must be an integer, got {parts[0]!r}", pos)
            tq = parse_quantity(parts[1], parts[2], "time", f"{pos} col 2")
            vq = parse_quantity(parts[3], parts[4], "volume", f"{pos} col 4")
            ref = None
            if len(parts) == 6:
                try:
                    ref = int(parts[5])
                except ValueError:
                    raise FormatCorrupt(f"blank_ref must be an integer, got {parts[5]!r}", pos)
            mat.injections.append(Injection(idx, tq, vq, ref, lineno))
        else:
            raise FormatCorrupt(f"data outside of any section: {line!r}", pos)

    _post_validate(mat)
    ids = {slot: material_id(slot, mat.slot_content(slot)) for slot in MATERIAL_SLOTS}
    return ParsedRun(mat, ids)


def _curve_of(mat: MaterialSet, name: str) -> Curve:
    attr = mat.power if name == "power" else mat.baseline if name == "baseline" else mat.blank
    if attr is None:
        pu = {"power": ("uW", "s"), "baseline": ("uW", "s"), "blank": ("uW", "s")}[name]
        attr = Curve(unit=pu[0], time_unit=pu[1], source=name)
        setattr(mat, name, attr)
    return attr


def _post_validate(mat: MaterialSet) -> None:
    """Structural semantic checks (positioned). Raises SemanticNoSolution only
    when the document as a whole admits no legal interpretation."""
    if mat.power is None or not mat.power.samples:
        raise SemanticNoSolution("power curve has no samples", "E_NO_POWER", "@power")
    times = [s.t.canonical for s in mat.power.samples]
    if any(t is None for t in times):
        diags = [s.t.diagnostics[0] for s in mat.power.samples if s.t.canonical is None]
        raise FormatCorrupt(diags[0].message, diags[0].position or "@power")
    ps = [s.p for s in mat.power.samples]
    for p in ps:
        if not p.convertible:
            mat.diagnostics.extend(p.diagnostics)
    for prev, nxt in zip(times, times[1:]):
        if nxt <= prev:
            raise SemanticNoSolution(
                f"power samples must be strictly increasing in time ({prev} -> {nxt})",
                "E_TIME_ORDER",
            )
    if not mat.injections:
        raise SemanticNoSolution("no injections declared", "E_NO_INJECTIONS", "@injections")
    idxs = [inj.index for inj in mat.injections]
    if len(set(idxs)) != len(idxs):
        raise SemanticNoSolution("duplicate injection index", "E_DUP_INDEX", "@injections")
    itimes = [inj.time.canonical for inj in mat.injections]
    if any(t is None for t in itimes):
        bad = next(inj for inj in mat.injections if inj.time.canonical is None)
        raise FormatCorrupt(bad.time.diagnostics[0].message, bad.time.diagnostics[0].position)
    for prev, nxt in zip(itimes, itimes[1:]):
        if nxt <= prev:
            raise SemanticNoSolution("injection times must be strictly increasing", "E_INJ_ORDER", "@injections")
    for inj in mat.injections:
        if inj.volume.canonical is not None and inj.volume.canonical <= 0:
            mat.diagnostics.append(
                Diagnostic(
                    "E_NONPOSITIVE_VOLUME",
                    "error",
                    f"injection {inj.index} has non-positive volume",
                    f"line {inj.line}",
                )
            )
    for name in ("cell_conc", "syringe_conc", "temperature", "cell_volume"):
        q = getattr(mat, name)
        if q is not None and q.convertible and q.canonical <= 0:
            mat.diagnostics.append(
                Diagnostic("E_NONPOSITIVE", "error", f"{name} must be positive", None)
            )


def materials_to_dict(mat: MaterialSet) -> dict:
    return {
        "power": None if mat.power is None else mat.power.to_dict(),
        "baseline": None if mat.baseline is None else mat.baseline.to_dict(),
        "blank": None if mat.blank is None else mat.blank.to_dict(),
        "injections": [inj.to_dict() for inj in mat.injections],
        "cell_conc": None if mat.cell_conc is None else mat.cell_conc.to_dict(),
        "syringe_conc": None if mat.syringe_conc is None else mat.syringe_conc.to_dict(),
        "temperature": None if mat.temperature is None else mat.temperature.to_dict(),
        "cell_volume": None if mat.cell_volume is None else mat.cell_volume.to_dict(),
        "blank_rule": mat.blank_rule,
        "diagnostics": [d.to_dict() for d in mat.diagnostics],
    }
