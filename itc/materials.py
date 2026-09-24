"""Immutable, separately versioned source materials.

Seven material kinds are versioned independently - changing the concentration
revision never mutates a power series, and the derived layer can point at the
exact revision of every input it used:

``power``            sampled power vs acquisition time (per run/stream)
``injections``       injection ordinals, volumes and times (per run/stream)
``concentration``    syringe ligand stock concentration (per titration run)
``temperature``      bath temperature (per run)
``cell_volume``      active cell volume (per run)
``baseline``         baseline model parameters (per run)
``blank_link``       the dilution-blank run and ordinal correspondence rule

A material revision is immutable.  Its ``revision_id`` is content derived, so
byte-identical inputs always get identical, stable identities.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic, FormatCorrupt
from .units import RawValue, raw_value

MATERIAL_KINDS = (
    "power", "injections", "concentration", "temperature",
    "cell_volume", "baseline", "blank_link",
)
STREAM_KINDS = {"power", "injections"}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


def short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class MaterialRevision:
    material_id: str          # stable logical id, e.g. "power.titration"
    kind: str
    run_id: str               # titration / blank / flatrun
    version: int
    source: str               # direct source description (file path, patch id)
    data: dict[str, Any]      # structured payload containing RawValue objects
    raw: dict[str, Any]       # original imported representation, verbatim
    diagnostics: list[Diagnostic] = field(default_factory=list)
    revision_id: str = ""

    def __post_init__(self) -> None:
        if not self.revision_id:
            self.revision_id = self.compute_id()

    def compute_id(self) -> str:
        return f"{self.material_id}@v{self.version}:{digest_of(strip_raw(self.data))}"

    def ref(self) -> "MaterialRef":
        return MaterialRef(self.material_id, self.version, self.revision_id)

    def to_event_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "kind": self.kind,
            "run_id": self.run_id,
            "version": self.version,
            "source": self.source,
            "data": self.data,
            "raw": self.raw,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "revision_id": self.revision_id,
        }

    @staticmethod
    def from_event_dict(obj: dict) -> "MaterialRevision":
        diag = [Diagnostic(**d) for d in obj.get("diagnostics", [])]
        rev = MaterialRevision(
            obj["material_id"], obj["kind"], obj["run_id"], obj["version"],
            obj["source"], _rehydrate(obj["data"]), obj.get("raw", obj["data"]),
            diag, obj.get("revision_id", ""),
        )
        return rev


@dataclass(frozen=True)
class MaterialRef:
    material_id: str
    version: int
    revision_id: str

    def to_dict(self) -> dict:
        return {"material_id": self.material_id, "version": self.version,
                "revision_id": self.revision_id}

    @staticmethod
    def from_dict(obj: dict) -> "MaterialRef":
        return MaterialRef(obj["material_id"], obj["version"], obj["revision_id"])


def strip_raw(obj: Any) -> Any:
    if isinstance(obj, RawValue):
        return {"value": obj.value, "unit": obj.unit, "reference": obj.reference}
    if isinstance(obj, dict):
        return {k: strip_raw(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [strip_raw(v) for v in obj]
    return obj


def _rehydrate(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj.keys()) >= {"__raw__"}:
            inner = obj["__raw__"]
            rv = raw_value(inner)
            if obj.get("diagnostic"):
                rv.diagnostic = Diagnostic(**obj["diagnostic"])
            if obj.get("reference"):
                rv.reference = obj["reference"]
            return rv
        return {k: _rehydrate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rehydrate(v) for v in obj]
    return obj


def _dehydrate(obj: Any) -> Any:
    if isinstance(obj, RawValue):
        return {"__raw__": obj.raw, "reference": obj.reference,
                "diagnostic": obj.diagnostic.to_dict() if obj.diagnostic else None}
    if isinstance(obj, dict):
        return {k: _dehydrate(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dehydrate(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _rv(obj: Any, loc: str) -> RawValue:
    return raw_value(obj, location=loc)


def parse_power(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("samples",), source)
    samples = []
    diagnostics: list[Diagnostic] = []
    for i, row in enumerate(raw["samples"]):
        if not isinstance(row, dict) or "t" not in row or "p" not in row:
            raise FormatCorrupt(
                f"{source}: power sample #{i} must be an object with t and p",
                [Diagnostic("format:bad_sample", "sample needs t and p",
                            f"samples[{i}]", row)])
        tv = _rv(row["t"], f"samples[{i}].t")
        pv = _rv(row["p"], f"samples[{i}].p")
        samples.append({"t": tv, "p": pv, "flag": row.get("flag", "ok")})
    gaps = []
    for i, gap in enumerate(raw.get("gaps", [])):
        gaps.append([_rv(gap[0], f"gaps[{i}][0]"), _rv(gap[1], f"gaps[{i}][1]")])
    data = {
        "time_reference": raw.get("time_reference", "t=0 at acquisition start"),
        "samples": samples,
        "gaps": gaps,
    }
    return MaterialRevision(f"power.{run_id}", "power", run_id, version, source,
                            data, raw, diagnostics)


def parse_injections(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("injections",), source)
    items = []
    for i, row in enumerate(raw["injections"]):
        if not isinstance(row, dict) or "ordinal" not in row or "time" not in row or "volume" not in row:
            raise FormatCorrupt(
                f"{source}: injection #{i} needs ordinal/time/volume",
                [Diagnostic("format:bad_injection", "needs ordinal/time/volume",
                            f"injections[{i}]", row)])
        items.append({
            "ordinal": int(row["ordinal"]),
            "time": _rv(row["time"], f"injections[{i}].time"),
            "volume": _rv(row["volume"], f"injections[{i}].volume"),
            "discard_first": bool(row.get("discard_first", False)),
        })
    ordinals = [it["ordinal"] for it in items]
    diagnostics = []
    if len(set(ordinals)) != len(ordinals):
        diagnostics.append(Diagnostic("injections:duplicate_ordinal",
                                      "duplicate injection ordinal", "injections", ordinals))
    data = {"time_reference": raw.get("time_reference", "t=0 at acquisition start"),
            "injections": items}
    return MaterialRevision(f"injections.{run_id}", "injections", run_id, version,
                            source, data, raw, diagnostics)


def parse_concentration(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("syringe",), source)
    syr = raw["syringe"]
    data = {
        "species": syr.get("species", "ligand"),
        "concentration": _rv(syr["concentration"], "syringe.concentration"),
        "note": syr.get("note", ""),
    }
    cell = raw.get("cell")
    diagnostics: list[Diagnostic] = []
    if cell is None:
        diagnostics.append(Diagnostic(
            "concentration:missing_macromolecule",
            "cell/macromolecule concentration absent; binding models cannot be fitted",
            "cell", None, severity="warning"))
        data["cell_species"] = "macromolecule"
        data["cell_concentration"] = None
    else:
        data["cell_species"] = cell.get("species", "macromolecule")
        data["cell_concentration"] = _rv(cell["concentration"], "cell.concentration")
    return MaterialRevision(f"concentration.{run_id}", "concentration", run_id,
                            version, source, data, raw, diagnostics)


def parse_temperature(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("temperature",), source)
    data = {"temperature": _rv(raw["temperature"], "temperature"),
            "note": raw.get("note", "")}
    return MaterialRevision(f"temperature.{run_id}", "temperature", run_id,
                            version, source, data, raw)


def parse_cell_volume(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("cell_volume",), source)
    data = {"cell_volume": _rv(raw["cell_volume"], "cell_volume"),
            "note": raw.get("note", "")}
    return MaterialRevision(f"cell_volume.{run_id}", "cell_volume", run_id,
                            version, source, data, raw)


def parse_baseline(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("baseline",), source)
    bl = raw["baseline"]
    method = bl.get("method", "constant")
    if method not in ("constant", "linear"):
        raise FormatCorrupt(
            f"{source}: baseline method {method!r} unsupported",
            [Diagnostic("format:bad_baseline", "method must be constant|linear",
                        "baseline.method", method)])
    data = {"method": method,
            "level": _rv(bl["level"], "baseline.level"),
            "slope": _rv(bl["slope"], "baseline.slope") if "slope" in bl else None,
            "time_reference": bl.get("time_reference", "t=0 at acquisition start")}
    return MaterialRevision(f"baseline.{run_id}", "baseline", run_id, version,
                            source, data, raw)


def parse_blank_link(raw: dict, *, run_id: str, source: str, version: int) -> MaterialRevision:
    _require_keys(raw, ("blank_run",), source)
    rule = raw.get("correspondence", {})
    data = {
        "blank_run": raw["blank_run"],
        "rule": rule.get("rule", "ordinal"),
        "discard_first_in_blank": bool(rule.get("discard_first_in_blank", False)),
        "note": raw.get("note", ""),
    }
    return MaterialRevision(f"blank_link.{run_id}", "blank_link", run_id,
                            version, source, data, raw)


_PARSERS = {
    "power": parse_power,
    "injections": parse_injections,
    "concentration": parse_concentration,
    "temperature": parse_temperature,
    "cell_volume": parse_cell_volume,
    "baseline": parse_baseline,
    "blank_link": parse_blank_link,
}


def parse_material(kind: str, raw: dict, *, run_id: str, source: str,
                   version: int) -> MaterialRevision:
    if kind not in MATERIAL_KINDS:
        raise FormatCorrupt(
            f"{source}: unknown material kind {kind!r}",
            [Diagnostic("format:unknown_kind", "unknown material kind", "kind", kind)])
    return _PARSERS[kind](raw, run_id=run_id, source=source, version=version)


def _require_keys(raw: dict, keys: tuple[str, ...], source: str) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise FormatCorrupt(
            f"{source}: missing required keys {missing}",
            [Diagnostic("format:missing_key", f"missing {k}", k, None) for k in missing])


def dehydrate_revision(rev: MaterialRevision) -> dict:
    obj = rev.to_event_dict()
    obj["data"] = _dehydrate(obj["data"])
    return obj


def rehydrate_revision(obj: dict) -> MaterialRevision:
    obj = dict(obj)
    if "data" in obj:
        obj["data"] = _rehydrate(obj["data"])
    return MaterialRevision.from_event_dict(obj)


def logical_material_id(kind: str, run_id: str) -> str:
    return f"{kind}.{run_id}"
