"""Parser for the domain-specific ``.itc`` raw bundle text format.

The format is purpose-built for titration thermograms (not a generic table):

.. code-block:: text

    @run main_titration
    @kind sample
    @source main_titration.itc
    [meta]
    temperature = 298.15 K
    cell_volume = 202.4 uL
    syringe_concentration = 0.50 mM
    cell_concentration = 10 uM
    [injections]
    # ordinal, time, volume
    1, 120 s, 2.00 uL
    [power]
    # x_unit=s y_unit=uW
    0.0, 0.10
    [baseline]
    # x_unit=s y_unit=uW
    0.0, 0.02

Blank runs use ``@kind blank``.  Raw tokens are kept verbatim; unconvertible
values stay on the value object with a diagnostic instead of being dropped.
Structural problems raise :class:`~itcdomain.exceptions.FormatCorrupt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .diagnostics import DiagnosticBag
from .exceptions import FormatCorrupt
from .materials import InjectionRecord, Series
from .units import parse_quantity, parse_temperature

HEADER_PREFIX = "@"
SECTION_PREFIX = "["


@dataclass
class ParsedBundle:
    run_id: str
    kind: str
    source_ref: str
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    injections: list[InjectionRecord] = field(default_factory=list)
    power: Series | None = None
    baseline: Series | None = None
    diagnostics: DiagnosticBag = field(default_factory=DiagnosticBag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "source_ref": self.source_ref,
            "meta": self.meta,
            "injections": [r.to_dict() for r in self.injections],
            "power": self.power.to_dict() if self.power else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "diagnostics": [d.to_dict() for d in self.diagnostics.diagnostics],
        }


def _strip_comment(line: str) -> str:
    # ``#`` starts a comment everywhere; data may not contain ``#``.
    idx = line.find("#")
    return line if idx < 0 else line[:idx]


def _parse_header_directives(line: str) -> tuple[str, str]:
    body = line.strip()[1:].strip()
    if "=" in body:
        key, value = body.split("=", 1)
        return key.strip(), value.strip()
    parts = body.split(None, 1)
    if len(parts) != 2:
        raise FormatCorrupt(f"malformed directive {line.strip()!r}")
    return parts[0].strip(), parts[1].strip()


def _split_value_unit(token: str, *, location: str, bag: DiagnosticBag,
                      dimension: str, temperature: bool = False) -> dict[str, Any]:
    text = token.strip()
    parts = text.split()
    if len(parts) >= 2:
        raw_value, unit = parts[0], parts[1]
    elif len(parts) == 1:
        raw_value, unit = parts[0], None
        bag.error(
            "missing_unit",
            f"{dimension} value {raw_value!r} has no declared unit",
            location=location, fragment=text,
        )
    else:
        raw_value, unit = "", None
        bag.error("empty_value", f"empty {dimension} value", location=location, fragment=text)
    if temperature:
        q = parse_temperature(raw_value, unit, location=location)
    else:
        q = parse_quantity(raw_value, dimension, unit, location=location)
    if q.diagnostic:
        bag.error("unconvertible_quantity", q.diagnostic, location=location, fragment=text)
    return q.to_dict()


def _read_series(section: str, lines: list[tuple[int, str]], bag: DiagnosticBag) -> Series:
    x_unit: str | None = None
    y_unit: str | None = None
    points: list[tuple[str, str, str | None]] = []
    x_basis = "time"
    y_basis = "power"
    for lineno, raw_line in lines:
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.lower().startswith("x_unit="):
            x_unit = line.split("=", 1)[1].strip()
            continue
        if line.lower().startswith("y_unit="):
            y_unit = line.split("=", 1)[1].strip()
            continue
        if line.lower().startswith("x_basis="):
            x_basis = line.split("=", 1)[1].strip()
            continue
        if line.lower().startswith("y_basis="):
            y_basis = line.split("=", 1)[1].strip()
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2:
            bag.error(
                "series_bad_row",
                f"{section} row needs x,y at line {lineno}",
                line=lineno, fragment=line, location=f"{section}:{lineno}",
            )
            continue
        flag = cells[2] if len(cells) >= 3 and cells[2] else None
        x_token, y_token = cells[0], cells[1]
        for name, token, unit, dim in (
            ("x", x_token, x_unit, "time"),
            ("y", y_token, y_unit, "power"),
        ):
            try:
                float(token)
            except ValueError:
                bag.error(
                    "series_non_numeric",
                    f"{section} {name} value {token!r} is not numeric; kept at source position",
                    line=lineno, fragment=token, location=f"{section}:{lineno}:{name}",
                )
        points.append((x_token, y_token, flag))
    if not points:
        bag.error("series_empty", f"section [{section}] contains no points", location=section)
    return Series(
        points=tuple(points), x_unit=x_unit, y_unit=y_unit,
        x_basis=x_basis, y_basis=y_basis,
    )


def parse_itc(text: str, *, default_source: str = "<memory>") -> ParsedBundle:
    """Parse ``.itc`` text into a :class:`ParsedBundle`."""

    if not isinstance(text, str):
        raise FormatCorrupt("ITC input must be text")
    directives: dict[str, str] = {}
    sections: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            if current:
                sections[current].append((lineno, raw))
            continue
        if line.startswith("#"):
            if current:
                sections[current].append((lineno, raw))
            continue
        if line.startswith(HEADER_PREFIX):
            key, value = _parse_header_directives(line)
            if key in directives:
                raise FormatCorrupt(f"duplicate directive @{key} at line {lineno}")
            directives[key] = value
            continue
        if line.startswith(SECTION_PREFIX):
            if not line.endswith("]"):
                raise FormatCorrupt(f"malformed section header at line {lineno}: {line!r}")
            name = line[1:-1].strip()
            if name in sections:
                raise FormatCorrupt(f"duplicate section [{name}] at line {lineno}")
            sections[name] = []
            current = name
            continue
        if current is None:
            raise FormatCorrupt(f"content outside of any section at line {lineno}: {line!r}")
        sections[current].append((lineno, raw))

    if "run" not in directives:
        raise FormatCorrupt("missing required @run directive")
    run_id = directives["run"]
    kind = directives.get("kind", "sample")
    if kind not in ("sample", "blank"):
        raise FormatCorrupt(f"@kind must be sample|blank, got {kind!r}")
    source_ref = directives.get("source", default_source)

    bag = DiagnosticBag()
    bundle = ParsedBundle(run_id=run_id, kind=kind, source_ref=source_ref, diagnostics=bag)

    # ---- [meta] -----------------------------------------------------------
    meta: dict[str, dict[str, Any]] = {}
    spec = {
        "temperature": ("temperature", True),
        "cell_volume": ("volume", False),
        "syringe_concentration": ("concentration", False),
        "cell_concentration": ("concentration", False),
    }
    for lineno, raw_line in sections.get("meta", []):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if "=" not in line:
            bag.error("meta_bad_row", f"meta line {lineno} needs key = value",
                      line=lineno, fragment=line, location=f"meta:{lineno}")
            continue
        key, value_token = (part.strip() for part in line.split("=", 1))
        if key in meta:
            bag.error("meta_duplicate", f"duplicate meta key {key!r}",
                      line=lineno, location=f"meta:{lineno}")
            continue
        if key not in spec:
            bag.warning("meta_unknown", f"unknown meta key {key!r} preserved verbatim",
                        line=lineno, location=f"meta:{lineno}", fragment=line)
            meta[key] = {"raw": value_token, "quantity": None}
            continue
        dimension, is_temp = spec[key]
        meta[key] = {"quantity": _split_value_unit(
            value_token, location=f"meta:{lineno}", bag=bag,
            dimension=dimension, temperature=is_temp,
        )}
    for required_key in ("temperature", "cell_volume", "syringe_concentration"):
        if required_key not in meta:
            bag.error(
                "meta_missing",
                f"required meta field {required_key!r} is absent",
                location="meta",
            )
    bundle.meta = meta

    # ---- [injections] -----------------------------------------------------
    seen_ordinals: set[int] = set()
    for lineno, raw_line in sections.get("injections", []):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if "=" in line and line.split("=", 1)[0].strip().isidentifier():
            bag.error(
                "injection_header_in_data",
                "injection sections do not accept key=value headers",
                line=lineno, fragment=line, location=f"injections:{lineno}",
            )
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 3:
            bag.error("injection_bad_row",
                      f"injection row needs ordinal, time, volume at line {lineno}",
                      line=lineno, fragment=line, location=f"injections:{lineno}")
            continue
        ordinal_token, time_token, volume_token = cells[0], cells[1], cells[2]
        note = cells[3] if len(cells) >= 4 and cells[3] else None
        try:
            ordinal = int(ordinal_token)
        except ValueError:
            bag.error("injection_bad_ordinal",
                      f"injection ordinal {ordinal_token!r} is not an integer",
                      line=lineno, fragment=ordinal_token, location=f"injections:{lineno}")
            continue
        if ordinal in seen_ordinals:
            bag.error("injection_duplicate_ordinal",
                      f"duplicate injection ordinal {ordinal}",
                      line=lineno, location=f"injections:{lineno}")
            continue
        seen_ordinals.add(ordinal)
        time_parts = time_token.split()
        volume_parts = volume_token.split()
        time_raw = time_parts[0] if time_parts else ""
        time_unit = time_parts[1] if len(time_parts) > 1 else None
        volume_raw = volume_parts[0] if volume_parts else ""
        volume_unit = volume_parts[1] if len(volume_parts) > 1 else None
        if time_unit is None:
            bag.error("injection_missing_time_unit",
                      f"injection {ordinal} time has no unit",
                      line=lineno, location=f"injections:{lineno}")
        if volume_unit is None:
            bag.error("injection_missing_volume_unit",
                      f"injection {ordinal} volume has no unit",
                      line=lineno, location=f"injections:{lineno}")
        qtime = parse_quantity(time_raw, "time", time_unit, location=f"injections:{lineno}")
        qvol = parse_quantity(volume_raw, "volume", volume_unit, location=f"injections:{lineno}")
        if qtime.diagnostic:
            bag.error("unconvertible_quantity", qtime.diagnostic,
                      line=lineno, location=f"injections:{lineno}:time")
        if qvol.diagnostic:
            bag.error("unconvertible_quantity", qvol.diagnostic,
                      line=lineno, location=f"injections:{lineno}:volume")
        bundle.injections.append(InjectionRecord(
            ordinal=ordinal,
            time_raw=time_raw, time_unit=time_unit,
            volume_raw=volume_raw, volume_unit=volume_unit,
            note=note,
        ))
    if not bundle.injections:
        bag.error("injections_empty", "no injections parsed", location="injections")

    # ---- [power] / [baseline] --------------------------------------------
    if "power" not in sections:
        bag.error("power_missing", "required section [power] is absent", location="power")
    else:
        bundle.power = _read_series("power", sections["power"], bag)
    if "baseline" in sections:
        bundle.baseline = _read_series("baseline", sections["baseline"], bag)

    unknown_sections = [s for s in sections if s not in {"meta", "injections", "power", "baseline"}]
    for name in unknown_sections:
        bag.warning("section_unknown", f"unknown section [{name}] ignored", location=name)

    return bundle
