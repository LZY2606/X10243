from __future__ import annotations

from .domain import Diagnostic, ParsedSeries, Quantity, Severity

ALIASES = {
    "s": ("s", 1.0),
    "sec": ("s", 1.0),
    "second": ("s", 1.0),
    "seconds": ("s", 1.0),
    "µcal/s": ("µJ/s", 4.184),
    "ucal/s": ("µJ/s", 4.184),
    "µcal/sec": ("µJ/s", 4.184),
    "ucal/sec": ("µJ/s", 4.184),
    "µj/s": ("µJ/s", 1.0),
    "uj/s": ("µJ/s", 1.0),
    "μj/s": ("µJ/s", 1.0),
    "µj/sec": ("µJ/s", 1.0),
    "uj/sec": ("µJ/s", 1.0),
    "μj/sec": ("µJ/s", 1.0),
    "cal/s": ("mJ/s", 4.184),
    "j/s": ("mJ/s", 1000.0),
    "mw": ("mW", 1.0),
    "w": ("mW", 1000.0),
    "µw": ("µW", 1.0),
    "uw": ("µW", 1.0),
    "μw": ("µW", 1.0),
    "m": ("mM", 1.0),
    "mm": ("mM", 1.0),
    "mmol/l": ("mM", 1.0),
    "mmolar": ("mM", 1.0),
    "k": ("K", 1.0),
    "°c": ("°C", 1.0),
    "c": ("°C", 1.0),
    "µl": ("µL", 1.0),
    "ul": ("µL", 1.0),
    "μl": ("µL", 1.0),
    "ml": ("µL", 1000.0),
    "l": ("µL", 1_000_000.0),
    "µj": ("µJ", 1.0),
    "uj": ("µJ", 1.0),
    "μj": ("µJ", 1.0),
    "mj": ("µJ", 1000.0),
    "j": ("µJ", 1_000_000.0),
    "kj/mol": ("kJ/mol", 1.0),
    "cal/mol": ("kJ/mol", 0.004184),
    "kcal/mol": ("kJ/mol", 4.184),
}


def normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "")


def parse_quantity(raw: str, target_unit: str, location: str) -> tuple[Quantity | None, Diagnostic | None]:
    text = str(raw).strip()
    if not text:
        return None, Diagnostic("missing_quantity", Severity.error, "Missing quantity", location, text, target_unit)
    head = ""
    for char in text:
        if char.isdigit() or char in "+-.eE":
            head += char
        else:
            break
    unit_text = text[len(head):].strip()
    try:
        number = float(head)
    except (ValueError, OverflowError):
        return None, Diagnostic("unconvertible_value", Severity.error, "Value cannot be converted", location, text, "numeric value")
    alias = ALIASES.get(normalize_unit(unit_text or target_unit))
    if not alias:
        return Quantity(number, unit_text or target_unit, text, target_unit), Diagnostic(
            "unconvertible_unit", Severity.error, "Unit cannot be normalized", location, unit_text, target_unit
        )
    normalized_unit, factor = alias
    if normalized_unit != target_unit:
        return Quantity(number, unit_text or target_unit, text, target_unit), Diagnostic(
            "incompatible_unit", Severity.error, "Unit is incompatible with coordinate basis", location, unit_text, target_unit
        )
    return Quantity(number * factor, normalized_unit, text, target_unit), None


def parse_series(values: list[Any], unit: str, location: str) -> ParsedSeries:
    quantities: list[Quantity] = []
    numeric: list[float] = []
    diagnostics: list[Diagnostic] = []
    invalid: list[int] = []
    for index, raw in enumerate(values):
        parsed, diagnostic = parse_quantity(raw, unit, f"{location}[{index}]")
        if diagnostic:
            diagnostics.append(diagnostic)
            invalid.append(index)
            numeric.append(float("nan"))
        if parsed:
            quantities.append(parsed)
            numeric.append(parsed.value)
    return ParsedSeries(numeric, quantities, diagnostics, invalid)
