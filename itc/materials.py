"""Immutable, separately-versioned material streams.

The seven streams are versioned independently of each other and of every
decision:

    power, injections, concentrations, temperature,
    cell_volume, baseline, blank

Parsing never destroys the source representation: every scalar is wrapped in a
:class:`~itc.units.Quantity` whose ``raw`` field is the exact imported token.
Values that cannot be normalized remain in place with a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic, FormatCorrupted
from .ident import digest
from .units import Quantity, parse_quantity

STREAM_KINDS = (
    "power", "injections", "concentrations", "temperature",
    "cell_volume", "baseline", "blank",
)


@dataclass
class MaterialVersion:
    case_id: str
    kind: str
    version: int
    content: dict[str, Any]
    digest: str
    label: str = ""
    source_event_id: str | None = None

    @property
    def id(self) -> str:
        return make_material_id(self.case_id, self.kind, self.digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "case_id": self.case_id, "kind": self.kind,
            "version": self.version, "content": self.content,
            "digest": self.digest, "label": self.label,
            "source_event_id": self.source_event_id,
        }


def make_material_id(case_id: str, kind: str, dig: str) -> str:
    return f"mat_{digest([case_id, kind, dig], length=18)}"


def _diag(content: dict, code: str, message: str, loc: str,
          severity: str = "warning") -> None:
    content.setdefault("diagnostics", []).append(
        Diagnostic(code, message, loc, severity).to_dict())


def _scalar(payload: dict, key: str, dimension: str, content: dict,
            loc: str, *, required: bool = True) -> None:
    if key not in payload:
        if required:
            _diag(content, "missing_value", f"{key} is absent", f"{loc}/{key}",
                  "error")
        return
    q = parse_quantity(payload[key], dimension, f"{loc}/{key}")
    content[key] = q.to_dict()


def _series(payload: Any, loc: str, content: dict, *, power: bool) -> None:
    if not isinstance(payload, list):
        raise FormatCorrupted("series must be a list", location=loc)
    samples: list[dict[str, Any]] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise FormatCorrupted("sample must be an object", location=f"{loc}/{i}")
        if "t" not in item:
            raise FormatCorrupted("sample missing 't'", location=f"{loc}/{i}")
        tq = parse_quantity(item["t"], "time", f"{loc}/{i}/t")
        pkey = "p" if power else "value"
        row: dict[str, Any] = {"t": tq.value, "raw_t": tq.raw}
        if power:
            if item.get("p") is None:
                row["p"] = None
                row["raw_p"] = item.get("p")
            else:
                pq = parse_quantity(item["p"], "power", f"{loc}/{i}/p")
                row["p"] = pq.value
                row["raw_p"] = pq.raw
                for d in pq.diagnostics:
                    _diag(content, d.code, d.message, d.location, d.severity)
        else:
            row["value"] = item.get(pkey)
        for d in tq.diagnostics:
            _diag(content, d.code, d.message, d.location, d.severity)
        samples.append(row)
    content["samples"] = samples


def parse_power(payload: dict, loc: str = "/power") -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FormatCorrupted("power section must be an object", location=loc)
    content: dict[str, Any] = {"time_origin": payload.get("time_origin",
                                                          "instrument_clock")}
    if "samples" in payload:
        _series(payload["samples"], f"{loc}/samples", content, power=True)
    elif "series" in payload:
        _series(payload["series"], f"{loc}/series", content, power=True)
    else:
        raise FormatCorrupted("power section needs 'samples'", location=loc)
    gaps = []
    for i, gap in enumerate(payload.get("gaps", [])):
        if not isinstance(gap, dict) or "start" not in gap or "end" not in gap:
            raise FormatCorrupted("gap needs start/end", location=f"{loc}/gaps/{i}")
        sq = parse_quantity(gap["start"], "time", f"{loc}/gaps/{i}/start")
        eq = parse_quantity(gap["end"], "time", f"{loc}/gaps/{i}/end")
        gaps.append({"start": sq.value, "end": eq.value,
                     "reason": gap.get("reason", ""),
                     "raw_start": sq.raw, "raw_end": eq.raw})
        for q in (sq, eq):
            for d in q.diagnostics:
                _diag(content, d.code, d.message, d.location, d.severity)
    content["gaps"] = gaps
    if "sampling_interval" in payload:
        sq = parse_quantity(payload["sampling_interval"], "time",
                            f"{loc}/sampling_interval")
        content["sampling_interval"] = sq.value
        content["raw_sampling_interval"] = sq.raw
    content["time_unit"] = "s"
    content["power_unit"] = "uW"
    return content


def parse_injections(payload: Any, loc: str = "/injections") -> dict[str, Any]:
    if not isinstance(payload, list):
        raise FormatCorrupted("injections section must be a list", location=loc)
    rows = []
    content: dict[str, Any] = {}
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise FormatCorrupted("injection must be an object",
                                  location=f"{loc}/{i}")
        idx = item.get("index", i + 1)
        row: dict[str, Any] = {"index": idx}
        if "volume" not in item:
            _diag(content, "missing_value", "injection volume absent",
                  f"{loc}/{i}/volume", "error")
        else:
            vq = parse_quantity(item["volume"], "volume", f"{loc}/{i}/volume")
            row["volume"] = vq.to_dict()
        if "time" not in item:
            _diag(content, "missing_value", "injection time absent",
                  f"{loc}/{i}/time", "error")
        else:
            tq = parse_quantity(item["time"], "time", f"{loc}/{i}/time")
            row["time"] = tq.to_dict()
        row["note"] = item.get("note", "")
        rows.append(row)
    content["rows"] = rows
    content["time_unit"] = "s"
    content["volume_unit"] = "uL"
    return content


def parse_concentrations(payload: dict, loc: str = "/concentrations") -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FormatCorrupted("concentrations section must be an object",
                              location=loc)
    content: dict[str, Any] = {}
    _scalar(payload, "syringe", "concentration", content, loc, required=False)
    _scalar(payload, "cell", "concentration", content, loc, required=False)
    content["concentration_unit"] = "uM"
    return content


def parse_temperature(payload: dict, loc: str = "/temperature") -> dict[str, Any]:
    content: dict[str, Any] = {}
    _scalar(payload if "value" in payload else {"value": payload},
            "value", "temperature", content, loc)
    content["temperature_unit"] = "degC"
    return content


def parse_cell_volume(payload: dict, loc: str = "/cell_volume") -> dict[str, Any]:
    content: dict[str, Any] = {}
    _scalar(payload if "value" in payload else {"value": payload},
            "value", "volume", content, loc)
    content["volume_unit"] = "uL"
    return content


def parse_baseline(payload: dict, loc: str = "/baseline") -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FormatCorrupted("baseline section must be an object", location=loc)
    kind = payload.get("kind", "constant")
    content: dict[str, Any] = {"kind": kind, "time_unit": "s",
                               "power_unit": "uW"}
    if kind == "constant":
        _scalar(payload, "value", "power", content, loc)
    elif kind == "linear":
        for key in ("intercept", "slope"):
            _scalar(payload, key, "power", content, f"{loc}/{key}")
    elif kind == "curve":
        sub: dict[str, Any] = {}
        _series(payload.get("samples", payload.get("series", [])),
                f"{loc}/samples", sub, power=True)
        content["samples"] = sub["samples"]
    else:
        _diag(content, "unknown_baseline_kind",
              f"baseline kind {kind!r} is not supported", f"{loc}/kind", "error")
    return content


def parse_blank(payload: dict | None, loc: str = "/blank") -> dict[str, Any]:
    if payload is None:
        return {"present": False, "diagnostics": [Diagnostic(
            "blank_absent", "no blank control has been imported", loc,
            "warning").to_dict()]}
    if not isinstance(payload, dict):
        raise FormatCorrupted("blank section must be an object", location=loc)
    content: dict[str, Any] = {"present": True}
    if payload.get("present") is False:
        content["present"] = False
        _diag(content, "blank_absent",
              payload.get("note", "blank explicitly absent"), loc, "warning")
        return content
    if "power" in payload:
        content["power"] = parse_power(payload["power"], f"{loc}/power")
    else:
        _diag(content, "missing_value", "blank power curve absent",
              f"{loc}/power", "error")
    if "injections" in payload:
        content["injections"] = parse_injections(
            payload["injections"], f"{loc}/injections")
    else:
        _diag(content, "missing_value", "blank injection timing absent",
              f"{loc}/injections", "error")
    content["match_rule"] = payload.get("match_rule", "")
    content["note"] = payload.get("note", "")
    return content


PARSERS = {
    "power": parse_power,
    "injections": parse_injections,
    "concentrations": parse_concentrations,
    "temperature": parse_temperature,
    "cell_volume": parse_cell_volume,
    "baseline": parse_baseline,
    "blank": parse_blank,
}


def parse_stream(kind: str, payload: Any) -> dict[str, Any]:
    if kind not in STREAM_KINDS:
        raise FormatCorrupted(f"unknown stream kind {kind!r}", location=f"/{kind}")
    return PARSERS[kind](payload)


def split_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Split a raw import bundle into ``{kind: raw_payload}``.

    Top-level aliases (``temp``, ``V0`` ...) are accepted but the canonical
    key is stored. Malformed bundle structure raises FormatCorrupted.
    """
    if not isinstance(bundle, dict):
        raise FormatCorrupted("import bundle must be a JSON object", location="/")
    aliases = {"temp": "temperature", "V0": "cell_volume", "v0": "cell_volume",
               "inj": "injections", "conc": "concentrations"}
    out: dict[str, Any] = {}
    for key, value in bundle.items():
        kind = aliases.get(key, key)
        if kind in STREAM_KINDS:
            out[kind] = value
    return out


def content_diagnostics(content: dict[str, Any]) -> list[dict[str, Any]]:
    return list(content.get("diagnostics", []))


def error_diagnostics(content: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in content.get("diagnostics", []) if d.get("severity") == "error"]
