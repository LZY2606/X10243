"""导入层：原始功率曲线、注射时序、空白对照、浓度/温度/池体积/基线。

约定的纯文本格式（领域专用，不是通用表格）：

    # kind: power
    # time_unit: s
    # power_unit: uJ/s
    time,power
    0.0,-12.10

解析阶段保留：
  * raw_text       —— 原始字节文本原样入库
  * raw_headers    —— 原始单位表示
规范化（s / uJ/s / uL / mM / K）只在派生字段发生；
无法转换/解析的值留在 payload 的 raw_fields 并附 Diagnostic。
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any

from .errors import (
    Diagnostic, FormatCorrupt, Outcome,
    STATUS_FORMAT_CORRUPT, STATUS_SEMANTIC_UNSOLVABLE,
)
from .units import normalize


def _comment_headers(lines: list[str]) -> tuple[dict[str, str], int]:
    headers: dict[str, str] = {}
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        body = lines[i].lstrip()[1:].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        i += 1
    return headers, i


def _read_rows(text: str, path: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    if text is None or not text.strip():
        raise FormatCorrupt(f"{path}: 空内容")
    lines = text.splitlines()
    headers, data_start = _comment_headers(lines)
    reader = csv.DictReader(io.StringIO("\n".join(lines[data_start:])))
    if reader.fieldnames is None:
        raise FormatCorrupt(f"{path}: 缺少表头行")
    rows = [r for r in reader]
    if not rows:
        raise FormatCorrupt(f"{path}: 表头之后没有任何数据行")
    return headers, rows


def _numbers(rows: list[dict[str, str]], col: str, path: str
             ) -> tuple[list[float], list[Diagnostic]]:
    values: list[float] = []
    diags: list[Diagnostic] = []
    for i, r in enumerate(rows):
        token = (r.get(col) or "").strip()
        try:
            values.append(float(token))
        except ValueError:
            diags.append(Diagnostic(
                "value_unparsable",
                f"{path}.rows[{i}].{col}={token!r} 不是数字，该行保留未用",
                f"{path}.rows[{i}].{col}"))
    return values, diags


def parse_power(text: str, kind: str = "power", path: str = "power"
                ) -> Outcome:
    """power / blank / baseline 曲线统一解析。"""
    try:
        headers, rows = _read_rows(text, path)
    except FormatCorrupt as exc:
        return Outcome.fail(STATUS_FORMAT_CORRUPT,
                            [Diagnostic("format_corrupt", str(exc), path)])
    cols = {c.strip().lower() for c in (rows[0].keys())}
    missing = {"time", "power"} - cols
    if missing:
        return Outcome.fail(STATUS_FORMAT_CORRUPT, [Diagnostic(
            "missing_column", f"{path}: 缺少列 {sorted(missing)}", path)])

    t_diag_unit = headers.get("time_unit")
    p_diag_unit = headers.get("power_unit")
    diagnostics: list[Diagnostic] = []
    if not t_diag_unit:
        diagnostics.append(Diagnostic(
            "unit_missing", f"{path}: 缺少 # time_unit 头，时间坐标基准未知",
            f"{path}.time_unit"))
    if not p_diag_unit:
        diagnostics.append(Diagnostic(
            "unit_missing", f"{path}: 缺少 # power_unit 头，功率坐标基准未知",
            f"{path}.power_unit"))

    t_raw, d1 = _numbers(rows, "time", path)
    p_raw, d2 = _numbers(rows, "power", path)
    diagnostics += d1 + d2
    n_bad = len(rows) - min(len(t_raw), len(p_raw))
    if len(t_raw) != len(p_raw):
        return Outcome.fail(STATUS_FORMAT_CORRUPT, [Diagnostic(
            "column_length_mismatch",
            f"{path}: time/power 有效行数不一致（time={len(t_raw)}, power={len(p_raw)}）",
            path)])
    if len(t_raw) < 3:
        return Outcome.fail(STATUS_SEMANTIC_UNSOLVABLE, [Diagnostic(
            "insufficient_samples", f"{path}: 可解析样本不足 3 个（{n_bad} 行损坏）",
            path)])

    def _norm_all(vals, unit, dim, pth):
        out, first_diag = [], None
        for v in vals:
            nv, d = normalize(v, unit, dim, pth)
            if d and first_diag is None:
                first_diag = d
            out.append(nv)
        return (out if first_diag is None else out), first_diag
    t_s, dt = _norm_all(t_raw, headers.get("time_unit"), "time",
                         f"{path}.time_unit")
    p_u, dp = _norm_all(p_raw, headers.get("power_unit"), "power",
                        f"{path}.power_unit")
    for d in (dt, dp):
        if d:
            diagnostics.append(d)

    payload: dict[str, Any] = {
        "kind": kind,
        "raw_headers": dict(headers),
        "raw_tokens": {"time": [r.get("time", "").strip() for r in rows],
                       "power": [r.get("power", "").strip() for r in rows]},
        "n_rows": len(rows),
        "basis": {},
    }
    if t_s is not None:
        payload["t"] = t_s
        payload["basis"]["time"] = "s"
    else:
        payload["t"] = None
        payload["t_raw"] = t_raw
    if p_u is not None:
        payload["p"] = p_u
        payload["basis"]["power"] = "uJ/s"
    else:
        payload["p"] = None
        payload["p_raw"] = p_raw
    if t_s is None or p_u is None:
        # 语义无解：坐标基准无法建立，不能进入数值积分；原始值仍在 payload
        return Outcome.fail(STATUS_SEMANTIC_UNSOLVABLE, diagnostics + [Diagnostic(
            "basis_unresolved",
            f"{path}: 无法建立时间/功率坐标基准，拒绝在无单位坐标上计算",
            path)])

    monotonic = all(
        t_s[i] < t_s[i + 1] for i in range(len(t_s) - 1))
    if not monotonic:
        diagnostics.append(Diagnostic(
            "time_not_monotonic",
            f"{path}: 时间列非严格递增，坐标基准不可靠", f"{path}.time", "warning"))
    return Outcome.good(payload, diagnostics)


def parse_schedule(text: str, path: str = "injections") -> Outcome:
    try:
        headers, rows = _read_rows(text, path)
    except FormatCorrupt as exc:
        return Outcome.fail(STATUS_FORMAT_CORRUPT,
                            [Diagnostic("format_corrupt", str(exc), path)])
    cols = {c.strip().lower() for c in rows[0].keys()}
    missing = {"index", "time", "volume"} - cols
    if missing:
        return Outcome.fail(STATUS_FORMAT_CORRUPT, [Diagnostic(
            "missing_column", f"{path}: 缺少列 {sorted(missing)}", path)])
    diagnostics: list[Diagnostic] = []
    injections: list[dict[str, Any]] = []
    times, vols = [], []
    for i, r in enumerate(rows):
        idx_tok = (r.get("index") or "").strip()
        t_tok = (r.get("time") or "").strip()
        v_tok = (r.get("volume") or "").strip()
        raw = {"index": idx_tok, "time": t_tok, "volume": v_tok}
        try:
            idx = int(float(idx_tok))
            tv = float(t_tok)
            vv = float(v_tok)
        except ValueError:
            diagnostics.append(Diagnostic(
                "value_unparsable",
                f"{path}.rows[{i}]={raw} 含不可解析值，该注射保留但不参与计算",
                f"{path}.rows[{i}]"))
            injections.append({"index": None, "raw": raw, "usable": False})
            continue
        injections.append({
            "index": idx, "time_raw": tv, "volume_raw": vv,
            "raw": raw, "usable": True})
        times.append(tv)
        vols.append(vv)
    def _norm_list(vals, unit, dim, pth):
        out, first = [], None
        for v in vals:
            nv, d = normalize(v, unit, dim, pth)
            first = first or d
            out.append(nv)
        return out, first
    t_s, dt = _norm_list(times, headers.get("time_unit"), "time",
                         f"{path}.time_unit")
    v_u, dv = _norm_list(vols, headers.get("volume_unit"), "volume",
                         f"{path}.volume_unit")
    for d in (dt, dv):
        if d:
            diagnostics.append(d)
    if t_s is None or v_u is None:
        return Outcome.fail(STATUS_SEMANTIC_UNSOLVABLE, diagnostics + [Diagnostic(
            "basis_unresolved", f"{path}: 时间或体积坐标基准无法建立", path)])
    usable = [inj for inj in injections if inj["usable"]]
    for inj, t_norm, v_norm in zip(usable, t_s, v_u):
        inj["time_s"] = t_norm
        inj["volume_uL"] = v_norm
    payload = {
        "kind": "schedule",
        "raw_headers": dict(headers),
        "basis": {"time": "s", "volume": "uL"},
        "injections": sorted(
            [i for i in injections if i["usable"]],
            key=lambda x: (x["time_s"], x["index"])),
        "unusable": [i for i in injections if not i["usable"]],
    }
    return Outcome.good(payload, diagnostics)


def parse_quantity_json(text: str, kind: str, dimension: str,
                        fields: list[str], path: str) -> Outcome:
    """concentrations / temperature / cell_volume 的 JSON 导入。

    无法转换的字段留在 raw_fields 并附诊断，不阻止其它字段规范化。
    但“业务必需字段”不可用会返回语义无解。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return Outcome.fail(STATUS_FORMAT_CORRUPT, [Diagnostic(
            "format_corrupt", f"{path}: JSON 解析失败 {exc}",
            f"{path}:{getattr(exc, 'lineno', '?')}")])
    if not isinstance(data, dict):
        return Outcome.fail(STATUS_FORMAT_CORRUPT, [Diagnostic(
            "format_corrupt", f"{path}: 顶层必须是对象", path)])
    diagnostics: list[Diagnostic] = []
    payload: dict[str, Any] = {"kind": kind, "raw_fields": {}, "fields": {},
                               "unconverted": [], "basis": {}}
    required_missing = False
    for key, spec in data.items():
        payload["raw_fields"][key] = spec
        if key not in fields:
            # 额外材料字段：无法识别/转换也不丢，留在 raw_fields + 诊断
            diagnostics.append(Diagnostic(
                "field_noncanonical",
                f"{path}.{key}={spec!r} 非规范量，原值保留在 raw_fields，不参与计算",
                f"{path}.{key}", "info"))
            payload["unconverted"].append(key)
            continue
        if not isinstance(spec, dict) or "value" not in spec or "unit" not in spec:
            diagnostics.append(Diagnostic(
                "shape_invalid",
                f"{path}.{key} 必须形如 {{value, unit}}，原值保留",
                f"{path}.{key}"))
            required_missing = True
            continue
        val, diag = normalize(spec["value"], spec["unit"], dimension, f"{path}.{key}")
        if diag:
            # 无法转换的业务字段留在原位（raw_fields），附诊断
            payload["unconverted"].append(key)
            if key in fields:
                required_missing = True
            diagnostics.append(diag)
            continue
        payload["fields"][key] = val
    if required_missing or any(f not in payload["fields"] for f in fields):
        return Outcome.fail(STATUS_SEMANTIC_UNSOLVABLE, diagnostics + [Diagnostic(
            "required_quantity_unresolved",
            f"{path}: 必需量 {fields} 未能全部规范化，拒绝用空值/默认值继续",
            path)])
    return Outcome.good(payload, diagnostics)


def parse_concentrations(text: str) -> Outcome:
    out = parse_quantity_json(
        text, "concentrations", "concentration",
        ["cell", "syringe"], "concentrations")
    if out.ok:
        out.value["cell_mM"] = out.value["fields"]["cell"]
        out.value["syringe_mM"] = out.value["fields"]["syringe"]
        out.value["basis"] = {"concentration": "mM"}
    return out


def parse_temperature(text: str) -> Outcome:
    out = parse_quantity_json(
        text, "temperature", "temperature", ["temperature"], "temperature")
    if out.ok:
        out.value["K"] = out.value["fields"]["temperature"]
        out.value["basis"] = {"temperature": "K"}
    return out


def parse_cell_volume(text: str) -> Outcome:
    out = parse_quantity_json(
        text, "cell_volume", "volume", ["cell_volume"], "cell_volume")
    if out.ok:
        out.value["uL"] = out.value["fields"]["cell_volume"]
        out.value["basis"] = {"volume": "uL"}
    return out


def parse_baseline(text: str) -> Outcome:
    return parse_power(text, kind="baseline", path="baseline")
