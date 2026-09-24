"""原始材料导入：保留原始表示、行列位置；格式损坏/语义无解/预算耗尽是三种不同结果。

- FormatCorrupt: 结构性损坏（编码、列数不一致、缺列、JSON 非法），调用方必须整体拒绝。
- 单元值无法转换：不抛异常，保留 raw 到对应字段并附 diagnostics（行/列级）。
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

from .units import parse_quantity


class FormatCorrupt(Exception):
    def __init__(self, reason: str, location: str):
        super().__init__(f"{reason} @ {location}")
        self.reason = reason
        self.location = location


@dataclass
class IngestResult:
    payload: dict
    diagnostics: list[dict] = field(default_factory=list)


def read_csv(raw: str, required_cols: list[str], loc: str) -> list[dict]:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise FormatCorrupt("encoding_error: 文件不是 UTF-8 文本", loc) from exc
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise FormatCorrupt("empty_file: 没有任何行", loc)
    header = [h.strip() for h in rows[0]]
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise FormatCorrupt(f"missing_columns: 缺少列 {missing}", f"{loc}:1")
    width = len(header)
    out = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != width:
            raise FormatCorrupt(
                f"ragged_row: 期望 {width} 列实际 {len(r)} 列", f"{loc}:{i}")
        out.append({header[j]: r[j].strip() for j in range(width)})
    return out


def _q(raw, kind, diags, pos, field_name):
    q = parse_quantity("" if raw is None else str(raw), kind)
    if q.diagnostic:
        diags.append({"code": q.diagnostic.split(":")[0],
                      "message": q.diagnostic,
                      "location": f"{pos}#{field_name}"})
    return q.as_dict()


def ingest_power(raw: str, source: str) -> IngestResult:
    rows = read_csv(raw, ["time", "power"], source)
    points, diags = [], []
    prev_t = None
    for i, r in enumerate(rows, start=2):
        pos = f"{source}:{i}"
        tq = _q(r["time"], "time", diags, pos, "time")
        pq = _q(r["power"], "power", diags, pos, "power")
        gap_after = False
        if tq["si_value"] is not None and prev_t is not None and tq["si_value"] - prev_t > 5.0:
            gap_after = True  # 标注缺口（供积分层判定）；不丢点
        if tq["si_value"] is not None:
            prev_t = tq["si_value"]
        points.append({"time": tq, "power": pq, "row": i,
                       "gap_after_prev": gap_after})
    return IngestResult({"points": points}, diags)


def ingest_schedule(raw: str, source: str, is_blank: bool = False) -> IngestResult:
    rows = read_csv(raw, ["injection", "time", "volume"], source)
    inj, diags = [], []
    for i, r in enumerate(rows, start=2):
        pos = f"{source}:{i}"
        idx = r["injection"].strip()
        if not idx.isdigit():
            diags.append({"code": "unconvertible_value",
                          "message": f"unconvertible_value:injection_index:'{idx}'",
                          "location": f"{pos}#injection"})
        inj.append({
            "index": int(idx) if idx.isdigit() else None,
            "time": _q(r["time"], "time", diags, pos, "time"),
            "volume": _q(r["volume"], "volume", diags, pos, "volume"),
            "role": "blank" if is_blank else "sample",
            "row": i,
        })
    return IngestResult({"injections": inj}, diags)


def ingest_concentration(raw: str, source: str) -> IngestResult:
    rows = read_csv(raw, ["injection", "ligand_concentration",
                          "macromolecule_concentration"], source)
    out, diags = [], []
    for i, r in enumerate(rows, start=2):
        pos = f"{source}:{i}"
        idx = r["injection"].strip()
        out.append({
            "index": int(idx) if idx.isdigit() else None,
            "ligand": _q(r["ligand_concentration"], "concentration", diags, pos,
                         "ligand_concentration"),
            "macromolecule": _q(r["macromolecule_concentration"], "concentration",
                                diags, pos, "macromolecule_concentration"),
            "row": i,
        })
    return IngestResult({"concentrations": out}, diags)


def ingest_conditions(raw: str, source: str) -> IngestResult:
    rows = read_csv(raw, ["key", "value", "kind", "unit"], source)
    items, diags = [], []
    for i, r in enumerate(rows, start=2):
        pos = f"{source}:{i}"
        kind = r["kind"].strip()
        if kind not in {"temperature", "volume", "power", "time", "concentration",
                        "energy", "dimensionless"}:
            diags.append({"code": "unconvertible_value",
                          "message": f"unconvertible_value:kind:'{kind}'",
                          "location": f"{pos}#kind"})
            q = {"raw": r["value"], "value": None, "si_value": None,
                 "si_unit": None, "kind": kind, "unit": r["unit"],
                 "diagnostic": f"unconvertible_value:kind:'{kind}'"}
        elif kind == "dimensionless":
            q = {"raw": r["value"], "value": _num(r["value"]), "si_value": _num(r["value"]),
                 "si_unit": None, "kind": kind, "unit": None}
        else:
            full = f"{r['value'].strip()} {r['unit'].strip()}".strip()
            q = _q(full, kind, diags, pos, r["key"])
        items.append({"key": r["key"].strip(), "quantity": q, "row": i})
    return IngestResult({"conditions": items}, diags)


def _num(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def ingest_json(raw: str, source: str) -> IngestResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FormatCorrupt("json_error: JSON 结构损坏", f"{source}:{exc.lineno}") from exc
    return IngestResult(payload, [])
