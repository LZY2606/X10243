"""不可变原始材料层。

材料（material）一旦写入即不可变；同一逻辑项（key）的新版本是新的内容地址身份。
每种材料都保留原始文本 raw；结构化视图在派生时规范化，并逐格保留诊断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import FormatCorrupt, diagnostic
from .ids import material_id
from .units import parse_quantity, parse_unit_header

# 需要“分开版本化”的七类材料
MATERIAL_KINDS = (
    "power_trace",       # 功率采样
    "injections",        # 注射体积和时间
    "concentration",     # 浓度（针筒/池，逐 key 版本化）
    "temperature",       # 温度
    "cell_volume",       # 池体积
    "baseline_rule",     # 基线规则（版本化）
    "blank_trace",       # 空白对照曲线（版本化、与样品分开）
    "subtraction_rule",  # 稀释热对应规则（版本化）
)

_META_DIMENSIONS: dict[str, str] = {
    "concentration_syringe": "concentration",
    "concentration_cell": "concentration",
    "temperature": "temperature",
    "cell_volume": "volume",
}

_GAP_TOKENS = {"", "na", "n/a", "gap", "nan", "null", "none", "missing"}


@dataclass(frozen=True)
class Material:
    id: str
    kind: str
    key: str
    source: str
    raw: str
    view: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "key": self.key,
                "source": self.source, "raw": self.raw, "view": self.view}


def _make(kind: str, key: str, source: str, raw: str, view: dict[str, Any]) -> Material:
    return Material(material_id(kind, key, raw, source), kind, key, source, raw, view)


def _split_csv(raw: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # 保留尾空行信息但去掉完全无内容的末行
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) < 2:
        raise FormatCorrupt(
            "CSV 至少需要表头与一行数据",
            diagnostics=[diagnostic("$", "少于 2 行，结构无法解析")],
        )
    header = [c.strip() for c in lines[0].split(",")]
    rows = [[c.strip() for c in ln.split(",")] for ln in lines[1:]]
    return header, rows


def _parse_trace(kind: str, key: str, source: str, raw: str) -> Material:
    header, rows = _split_csv(raw)
    if len(header) < 2:
        raise FormatCorrupt("曲线 CSV 需要两列", diagnostics=[diagnostic("$.header", str(header))])
    t_name, t_unit_raw = parse_unit_header(header[0])
    p_name, p_unit_raw = parse_unit_header(header[1])
    if not t_unit_raw or not p_unit_raw:
        raise FormatCorrupt(
            "曲线列名必须在表头携带单位，例如 'time [s]' 与 'power [uJ/s]'",
            diagnostics=[diagnostic("$.header", f"得到 {header}")],
        )

    times: list[float | None] = []
    powers: list[float | None] = []
    cells: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        if len(row) < 2:
            diagnostics.append(diagnostic(f"$.rows[{i}]", "列数不足，该行保留为缺口",
                                          raw=",".join(row), severity="warning",
                                          code="short_row"))
            times.append(None)
            powers.append(None)
            continue
        t_cell = _parse_cell(row[0], "time", f"$.rows[{i}].{t_name}", t_unit_raw)
        p_cell = _parse_cell(row[1], "power", f"$.rows[{i}].{p_name}", p_unit_raw)
        cells.append({"row": i, "time": t_cell, "power": p_cell})
        times.append(t_cell["canonical"])
        powers.append(p_cell["canonical"])
        if t_cell["diagnostics"]:
            diagnostics.extend(t_cell["diagnostics"])
        if p_cell["diagnostics"]:
            diagnostics.extend(p_cell["diagnostics"])

    valid_t = [t for t in times if t is not None]
    view = {
        "columns": [
            {"name": t_name, "unit_raw": t_unit_raw, "canonical_unit": "s"},
            {"name": p_name, "unit_raw": p_unit_raw, "canonical_unit": "uJ/s"},
        ],
        "coordinate_basis": {"axis": "time", "unit": "s", "zero": "first_sample"},
        "n_samples": len(rows),
        "times": times,
        "powers": powers,
        "cells": cells,
        "diagnostics": diagnostics,
        "t_min": min(valid_t) if valid_t else None,
        "t_max": max(valid_t) if valid_t else None,
    }
    return _make(kind, key, source, raw, view)


def _has_unit_token(token: str) -> bool:
    m = re.match(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(.*)$", token.strip())
    return bool(m and m.group(1).strip())


def _parse_cell(raw: str, dimension: str, path: str, header_unit: str | None = None) -> dict[str, Any]:
    token = raw.strip()
    if token.lower() in _GAP_TOKENS:
        diag = diagnostic(path, "缺口标记：该采样点缺失并保留原位",
                          raw=raw, severity="warning", code="gap_marker")
        return {"raw": raw, "canonical": None, "diagnostics": [diag]}
    candidate = token
    if header_unit and not _has_unit_token(token):
        candidate = f"{token} {header_unit}"
    q = parse_quantity(candidate, dimension, path)
    out = {"raw": raw, "canonical": q.canonical,
           "unit_raw": q.unit, "diagnostics": list(q.diagnostics)}
    if q.canonical is None and not any(
            d.get("code") == "gap_marker" for d in q.diagnostics):
        out["diagnostics"].insert(0, diagnostic(
            path, "单元格无法规范化，保留原值于原位（不会被丢弃）",
            raw=raw, severity="warning", code="unconverted_cell"))
    return out


def parse_power_trace(raw: str, source: str, key: str = "sample") -> Material:
    return _parse_trace("power_trace", key, source, raw)


def parse_blank_trace(raw: str, source: str, key: str = "blank") -> Material:
    return _parse_trace("blank_trace", key, source, raw)


def parse_injections(raw: str, source: str, key: str = "injections") -> Material:
    header, rows = _split_csv(raw)
    names = [parse_unit_header(h)[0] for h in header]
    units = [parse_unit_header(h)[1] for h in header]
    required = {"inj_index", "time", "volume"}
    if not required.issubset(set(names)):
        raise FormatCorrupt(
            "注射时序 CSV 必须含 inj_index, time, volume 三列（单位写在表头）",
            diagnostics=[diagnostic("$.header", f"得到 {header}")],
        )
    idx_i = names.index("inj_index")
    time_i = names.index("time")
    vol_i = names.index("volume")

    injections: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        if max(idx_i, time_i, vol_i) >= len(row):
            raise FormatCorrupt(
                "注射行列数不足",
                diagnostics=[diagnostic(f"$.rows[{i}]", "列数不足", raw=",".join(row))],
            )
        index_raw = row[idx_i].strip()
        t_cell = _parse_cell(row[time_i], "time", f"$.rows[{i}].time", units[time_i])
        v_cell = _parse_cell(row[vol_i], "volume", f"$.rows[{i}].volume", units[vol_i])
        from .units import QToken
        tq = parse_quantity(f"{row[time_i]} {units[time_i]}".strip(), "time", f"$.rows[{i}].time")
        vq = parse_quantity(f"{row[vol_i]} {units[vol_i]}".strip(), "volume", f"$.rows[{i}].volume")
        try:
            index_val = int(float(index_raw))
        except ValueError:
            index_val = None
            diagnostics.append(diagnostic(f"$.rows[{i}].inj_index",
                                          "注射序号不是整数，保留原值", raw=index_raw,
                                          severity="warning", code="bad_index"))
        diagnostics.extend([dict(d, inj_index=index_raw) for d in tq.diagnostics])
        diagnostics.extend([dict(d, inj_index=index_raw) for d in vq.diagnostics])
        injections.append({
            "inj_index": index_val,
            "index_raw": index_raw,
            "time": tq.to_dict(),
            "volume": vq.to_dict(),
        })
    view = {
        "columns": [{"name": n, "unit_raw": u} for n, u in zip(names, units)],
        "injections": injections,
        "diagnostics": diagnostics,
    }
    return _make("injections", key, source, raw, view)


def parse_scalar(kind_key: str, raw: str, source: str) -> Material:
    """浓度 / 温度 / 池体积：单值材料，原始表示完整保留。"""
    if kind_key not in _META_DIMENSIONS:
        raise FormatCorrupt(f"未知标量材料键: {kind_key}")
    dimension = _META_DIMENSIONS[kind_key]
    kind = "temperature" if kind_key == "temperature" else (
        "cell_volume" if kind_key == "cell_volume" else "concentration")
    q = parse_quantity(raw.strip(), dimension, "$.value")
    view = {"quantity": q.to_dict(), "canonical_unit": q.canonical_unit}
    return _make(kind, kind_key, source, raw, view)


def _parse_rule(kind: str, key: str, source: str, raw: str,
                required_fields: set[str]) -> Material:
    fields: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            diagnostics.append(diagnostic(f"$.lines[{i}]", "规则行缺少 ':'，保留原文",
                                          raw=line, severity="warning", code="bad_rule_line"))
            continue
        name, value = line.split(":", 1)
        fields[name.strip()] = value.strip()
    missing = required_fields - set(fields)
    if missing:
        raise FormatCorrupt(
            f"{kind} 缺少必填字段: {sorted(missing)}",
            diagnostics=[diagnostic("$", f"缺少 {sorted(missing)}")],
        )
    view = {"fields": fields, "rule_version": fields.get("rule_version"),
            "diagnostics": diagnostics}
    return _make(kind, key, source, raw, view)


def parse_baseline_rule(raw: str, source: str, key: str = "baseline") -> Material:
    return _parse_rule("baseline_rule", key, source, raw,
                       {"rule_version", "method"})


def parse_subtraction_rule(raw: str, source: str, key: str = "subtraction") -> Material:
    return _parse_rule("subtraction_rule", key, source, raw,
                       {"rule_version", "correspondence"})
