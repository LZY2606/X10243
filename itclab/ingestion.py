"""Parser for the repository's own role-oriented ITC text format.

A file carries exactly one material role. Sections are line oriented;
``DATA ... END`` blocks hold aligned scalar rows and may contain ``GAP``
markers (power). Every token retains its original spelling. Positions are
1-based line numbers so diagnostics point reviewers at the raw material.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import BudgetExhausted, Diagnostic, FormatCorrupt
from .units import is_known_unit, parse_quantity

MAGIC = "#ITCTEXT/1"

DEFAULT_BUDGET = {
    "max_bytes": 8 * 1024 * 1024,
    "max_rows": 200_000,
}

_HEADER_DIMS = {
    "TIME_UNIT": "time",
    "VALUE_UNIT": "power",
    "VOLUME_UNIT": "volume",
    "CELL_UNIT": "concentration",
    "SYRINGE_UNIT": "concentration",
    "TEMP_UNIT": "temperature",
    "CELL_VOLUME_UNIT": "volume",
}

_REQUIRED_HEADERS = {
    "power": ["TIME_UNIT", "VALUE_UNIT"],
    "injections": ["TIME_UNIT", "VOLUME_UNIT"],
    "concentration": ["CELL_UNIT", "SYRINGE_UNIT", "CELL", "SYRINGE"],
    "conditions": ["TEMP_UNIT", "TEMP", "CELL_VOLUME_UNIT", "CELL_VOLUME"],
    "baseline": ["ANCHOR_PRE_S", "ANCHOR_POST_S"],
}


@dataclass
class ParsedFile:
    role: str
    scope: str | None
    headers: dict[str, str] = field(default_factory=dict)
    content: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _diag(code: str, message: str, lineno: int, **extra: Any) -> Diagnostic:
    return Diagnostic(code, message, {"line": lineno, **extra})


def _check_budget(text: str, budget: dict[str, int]) -> None:
    if len(text.encode("utf-8")) > budget["max_bytes"]:
        raise BudgetExhausted(
            f"input exceeds {budget['max_bytes']} byte budget",
            [
                Diagnostic(
                    "BYTE_BUDGET",
                    f"refused: byte budget {budget['max_bytes']} exceeded",
                    {"bytes": len(text.encode("utf-8"))},
                )
            ],
        )


def parse_itc_text(text: str, *, budget: dict[str, int] | None = None) -> ParsedFile:
    budget = {**DEFAULT_BUDGET, **(budget or {})}
    _check_budget(text, budget)

    lines = text.splitlines()
    if not lines or lines[0].strip() != MAGIC:
        raise FormatCorrupt(
            f"missing magic line {MAGIC!r}",
            [_diag("MAGIC_LINE", f"file must start with {MAGIC}", 1)],
        )

    headers: dict[str, str] = {}
    diagnostics: list[Diagnostic] = []
    i = 1
    role: str | None = None
    scope: str | None = None
    in_data = False
    data_rows: list[list[str]] = []
    gap_lines: set[int] = set()
    saw_end = False

    while i < len(lines):
        lineno = i + 1
        raw_line = lines[i]
        stripped = raw_line.strip()
        i += 1
        if not stripped:
            continue
        if stripped.startswith("#") and not in_data:
            continue

        if not in_data:
            if stripped == "DATA":
                in_data = True
                continue
            if ":" not in stripped:
                raise FormatCorrupt(
                    "header line must be KEY: VALUE",
                    [_diag("HEADER_SYNTAX", "expected KEY: VALUE", lineno, "text", stripped)],
                )
            key, _, value = stripped.partition(":")
            headers[key.strip()] = value.strip()
            if key.strip() == "ROLE":
                role = value.strip()
            if key.strip() == "CHANNEL":
                scope = value.strip()
            continue

        if stripped == "END":
            saw_end = True
            break
        if stripped == "GAP":
            gap_lines.add(lineno)
            continue
        if len(data_rows) >= budget["max_rows"]:
            raise BudgetExhausted(
                f"row budget {budget['max_rows']} exhausted",
                [_diag("ROW_BUDGET", f"refused: more than {budget['max_rows']} rows", lineno)],
            )
        if stripped.startswith("#"):
            continue
        data_rows.append(stripped.split())

    if not saw_end:
        raise FormatCorrupt(
            "DATA block without END",
            [_diag("UNTERMINATED_DATA", "DATA block was not closed by END", i)],
        )
    if role is None:
        raise FormatCorrupt(
            "ROLE header missing",
            [_diag("MISSING_ROLE", "ROLE must be declared", 1)],
        )
    if role not in _REQUIRED_HEADERS:
        raise FormatCorrupt(
            f"unknown ROLE {role!r}",
            [_diag("UNKNOWN_ROLE", f"ROLE must be one of {sorted(_REQUIRED_HEADERS)}", 1)],
        )
    if role in ("power", "injections", "baseline"):
        if scope not in ("sample", "blank"):
            raise FormatCorrupt(
                f"ROLE {role} requires CHANNEL sample|blank",
                [_diag("MISSING_CHANNEL", "CHANNEL must be sample or blank", 1)],
            )
    elif scope is not None:
        diagnostics.append(_diag("UNEXPECTED_CHANNEL", f"{role} has no channels", 1))

    for key in _REQUIRED_HEADERS[role]:
        if key not in headers:
            raise FormatCorrupt(
                f"ROLE {role} requires header {key}",
                [_diag("MISSING_HEADER", f"required header {key} absent", 1, key=key)],
            )

    for unit_key, dimension in _HEADER_DIMS.items():
        if unit_key in headers:
            unit = headers[unit_key]
            if not is_known_unit(unit, dimension):
                diagnostics.append(
                    _diag(
                        "UNKNOWN_UNIT",
                        f"unit {unit!r} not registered for {dimension}; raw values retained",
                        1,
                        unit=unit,
                        dimension=dimension,
                    )
                )

    content: dict[str, Any] = {
        "headers": headers,
        "gaps_at_lines": sorted(gap_lines),
    }
    if role in ("power",):
        content.update(_build_series(role, headers, data_rows, diagnostics, lineno_base=0))
        content["gap_marker_lines"] = sorted(gap_lines)
    elif role == "injections":
        content.update(_build_injections(headers, data_rows, diagnostics))
    elif role == "concentration":
        content.update(_build_scalars(role, headers, diagnostics))
    elif role == "conditions":
        content.update(_build_scalars(role, headers, diagnostics))
    elif role == "baseline":
        content.update(_build_baseline(headers, diagnostics))

    content["raw_row_count"] = len(data_rows)

    parsed = ParsedFile(role=role, scope=scope, headers=headers, content=content, diagnostics=diagnostics)
    parsed.diagnostics.extend(semantic_check(parsed))
    return parsed


def _as_float(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _build_series(role, headers, data_rows, diagnostics, *, lineno_base) -> dict[str, Any]:
    tu, vu = headers["TIME_UNIT"], headers["VALUE_UNIT"]
    times_raw: list[str] = []
    values_raw: list[str] = []
    times: list[float | None] = []
    values: list[float | None] = []
    last_t = None
    for idx, row in enumerate(data_rows):
        lineno = idx + 3
        if len(row) != 2:
            diagnostics.append(
                _diag("ROW_ARITY", f"power row needs 2 fields, got {len(row)}", lineno)
            )
            continue
        t_raw, v_raw = row
        times_raw.append(t_raw)
        values_raw.append(v_raw)
        t = _as_float(t_raw)
        v = _as_float(v_raw)
        if t is None:
            diagnostics.append(_diag("BAD_TIME", f"time {t_raw!r} retained but not numeric", lineno))
        if v is None:
            diagnostics.append(_diag("BAD_VALUE", f"power {v_raw!r} retained but not numeric", lineno))
        if t is not None and last_t is not None and t <= last_t:
            diagnostics.append(
                _diag("NON_MONOTONIC_TIME", f"time {t} <= previous {last_t}", lineno)
            )
        if t is not None:
            last_t = t
        times.append(t)
        values.append(v)
    return {
        "time_unit": tu,
        "value_unit": vu,
        "times_raw": times_raw,
        "values_raw": values_raw,
        "times": times,
        "values": values,
    }


def _build_injections(headers, data_rows, diagnostics) -> dict[str, Any]:
    rows = []
    for idx, row in enumerate(data_rows):
        lineno = idx + 3
        if len(row) != 3:
            diagnostics.append(
                _diag("ROW_ARITY", f"injection row needs index,time,volume; got {len(row)}", lineno)
            )
            continue
        i_raw, t_raw, v_raw = row
        rows.append(
            {
                "index_raw": i_raw,
                "time_raw": t_raw,
                "volume_raw": v_raw,
                "index": _as_float(i_raw),
                "time": _as_float(t_raw),
                "volume": _as_float(v_raw),
                "line": lineno,
            }
        )
        for name, token in (("index", i_raw), ("time", t_raw), ("volume", v_raw)):
            if _as_float(token) is None:
                diagnostics.append(
                    _diag("BAD_INJECTION_FIELD", f"{name} {token!r} not numeric", lineno)
                )
    return {"time_unit": headers["TIME_UNIT"], "volume_unit": headers["VOLUME_UNIT"], "rows": rows}


def _build_scalars(role, headers, diagnostics) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if role == "concentration":
        out["cell"] = parse_quantity(
            headers["CELL"], headers["CELL_UNIT"], "concentration", {"header": "CELL"}
        ).to_dict()
        out["syringe"] = parse_quantity(
            headers["SYRINGE"], headers["SYRINGE_UNIT"], "concentration", {"header": "SYRINGE"}
        ).to_dict()
    else:
        out["temperature"] = parse_quantity(
            headers["TEMP"], headers["TEMP_UNIT"], "temperature", {"header": "TEMP"}
        ).to_dict()
        out["cell_volume"] = parse_quantity(
            headers["CELL_VOLUME"], headers["CELL_VOLUME_UNIT"], "volume", {"header": "CELL_VOLUME"}
        ).to_dict()
    return out


def _build_baseline(headers, diagnostics) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("ANCHOR_PRE_S", "ANCHOR_POST_S"):
        q = parse_quantity(headers[key], "s", "time", {"header": key})
        if q.base is None:
            diagnostics.append(
                Diagnostic("BAD_BASELINE_WINDOW", f"{key} must be seconds", {"header": key})
            )
        out[key.lower()] = q.to_dict()
    order_raw = headers.get("ORDER", "1")
    try:
        out["order"] = int(order_raw)
    except ValueError:
        out["order"] = None
        diagnostics.append(
            Diagnostic("BAD_BASELINE_ORDER", "ORDER must be 0 (mean) or 1 (linear)", {"raw": order_raw})
        )
    out["note"] = headers.get("NOTE", "")
    return out


def semantic_check(parsed: ParsedFile) -> list[Diagnostic]:
    """Intra-material semantic checks (cross-material checks live in the store)."""
    diags: list[Diagnostic] = []
    content = parsed.content
    if parsed.role == "power":
        valid_t = [t for t in content.get("times", []) if t is not None]
        valid_v = [v for v in content.get("values", []) if v is not None]
        if len(valid_t) < 2:
            diags.append(Diagnostic("SEMANTIC_TOO_FEW_SAMPLES", "power curve needs >=2 valid samples", {}))
        if not valid_v:
            diags.append(Diagnostic("SEMANTIC_NO_VALID_POWER", "no convertible power samples", {}))
    elif parsed.role == "injections":
        rows = content.get("rows", [])
        if not rows:
            diags.append(Diagnostic("SEMANTIC_NO_INJECTIONS", "at least one injection is required", {}))
        indexes = [r["index"] for r in rows if r["index"] is not None]
        if len(indexes) != len(set(indexes)):
            diags.append(Diagnostic("SEMANTIC_DUP_INDEX", "injection indexes must be unique", {}))
        times = [r["time"] for r in rows if r["time"] is not None]
        if any(v is not None and v <= 0 for v in [r["volume"] for r in rows]):
            diags.append(Diagnostic("SEMANTIC_VOLUME", "injection volumes must be positive", {}))
        if times != sorted(times):
            diags.append(Diagnostic("SEMANTIC_INJECTION_ORDER", "injection times must increase", {}))
    elif parsed.role == "concentration":
        for name in ("cell", "syringe"):
            if content.get(name, {}).get("base") is None:
                diags.append(
                    Diagnostic(
                        "SEMANTIC_CONCENTRATION",
                        f"{name} concentration has no convertible value",
                        {"field": name},
                    )
                )
    elif parsed.role == "conditions":
        for name in ("temperature", "cell_volume"):
            if content.get(name, {}).get("base") is None:
                diags.append(
                    Diagnostic(f"SEMANTIC_{name.upper()}", f"{name} has no convertible value", {"field": name})
                )
    elif parsed.role == "baseline":
        for key in ("anchor_pre_s", "anchor_post_s"):
            if content.get(key, {}).get("base") is None or content[key]["base"] <= 0:
                diags.append(Diagnostic("SEMANTIC_BASELINE_WINDOW", f"{key} must be positive", {}))
        if content.get("order") not in (0, 1):
            diags.append(Diagnostic("SEMANTIC_BASELINE_ORDER", "ORDER must be 0 or 1", {}))
    return diags
