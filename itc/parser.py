"""Parser for the offline domain material format (``.itc`` bundles).

Format sketch::

    ## MATERIAL: run=E1 kind=power version=1 basis=time:experiment_t0
    # columns: t_s P_uW
    0.0 5.0
    ...

A bundle holds one or more versioned material sections.  Section headers carry
the material kind; key/value blocks (``SETUP``, ``BASELINE``) and whitespace
delimited tables are both supported.  Structural problems raise FormatCorrupt;
a declared sample budget is enforced with BudgetExhausted; individual values are
never dropped - they travel as raw strings for the materialiser to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import BudgetExhausted, FormatCorrupt

KNOWN_KINDS = {"power", "injections", "concentration", "temperature", "cell_volume", "baseline", "blank"}
TABLE_KINDS = {"power", "injections", "blank"}
SETUP_KINDS = {"concentration", "temperature", "cell_volume"}


@dataclass
class RawBlock:
    kind: str
    run: str
    version: int
    basis: str | None
    lines: list[str]                 # raw, comment-stripped non-empty lines
    raw_header: str
    starts_at: int
    diagnostics: list[str] = field(default_factory=list)  # structural warnings

    @property
    def is_table(self) -> bool:
        return self.kind in TABLE_KINDS or (
            self.lines and ("=" not in self.lines[0] and len(self.lines[0].split()) >= 2)
        )


@dataclass
class ParsedDocument:
    blocks: list[RawBlock]
    source_name: str
    max_samples: int | None

    def blocks_of(self, kind: str) -> list[RawBlock]:
        return [b for b in self.blocks if b.kind == kind]


def _parse_header(line: str, lineno: int) -> tuple[str, str, int, str | None]:
    body = line[2:].strip()
    parts = body.split()
    attrs: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise FormatCorrupt(f"malformed header attribute {part!r}", f"line {lineno}", line)
        key, _, val = part.partition("=")
        attrs[key.strip()] = val.strip()
    kind = parts[0] if parts else ""
    kind_key, kind_val = ("kind", kind)
    if "kind" in attrs:  # explicit kind= wins
        kind_val = attrs["kind"]
    else:
        # first token without '=' is treated as kind
        first = parts[0] if parts else ""
        if "=" in first:
            raise FormatCorrupt("material header missing kind", f"line {lineno}", line)
    if kind_val not in KNOWN_KINDS:
        raise FormatCorrupt(f"unknown material kind {kind_val!r}", f"line {lineno}", line)
    run = attrs.get("run", attrs.get("label", "run"))
    version = 1
    if "version" in attrs:
        try:
            version = int(attrs["version"])
        except ValueError as exc:
            raise FormatCorrupt("material version must be an integer", f"line {lineno}", line) from exc
    basis = attrs.get("basis")
    return kind_val, run, version, basis


def parse_document(text: str, source_name: str = "<input>", max_samples: int | None = None) -> ParsedDocument:
    blocks: list[RawBlock] = []
    current: RawBlock | None = None
    data_rows = 0
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("##"):
            kind, run, version, basis = _parse_header(line, idx)
            current = RawBlock(
                kind=kind, run=run, version=version, basis=basis,
                lines=[], raw_header=line, starts_at=idx,
            )
            blocks.append(current)
            continue
        if current is None:
            raise FormatCorrupt(
                "content appears before any '## MATERIAL' header", f"line {idx}", raw_line
            )
        current.lines.append(line)
        if current.kind in TABLE_KINDS and "=" not in line:
            data_rows += 1
            if max_samples is not None and data_rows > max_samples:
                raise BudgetExhausted(
                    f"sample budget exhausted: more than {max_samples} data rows",
                    f"line {idx}",
                    raw_line,
                )
    if not blocks:
        raise FormatCorrupt("no '## MATERIAL' sections found", "document")
    return ParsedDocument(blocks=blocks, source_name=source_name, max_samples=max_samples)
