"""原始表示解析：功率/空白曲线 CSV、注射时序、分槽材料载荷。

解析层只做结构与词法检查，保留 raw 文本；规范化留给派生层。
结构损坏 -> format_corrupt；逐格诊断 -> diagnostics（不阻止存证）。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

from .constants import SLOT_INJECTIONS, SLOT_POWER
from .errors import Diagnostic, FormatCorruptError, Location


@dataclass
class ParsedCurve:
    rows: list[dict[str, Any]]
    columns: list[str]
    diagnostics: list[Diagnostic]


def parse_curve_csv(text: str, slot: str,
                    required: tuple[str, ...] = ("time", "power")) -> ParsedCurve:
    if text is None or not str(text).strip():
        raise FormatCorruptError(
            "曲线内容为空", Location(slot=slot))
    try:
        reader = csv.DictReader(io.StringIO(str(text)))
    except csv.Error as exc:  # pragma: no cover - DictReader 极少抛出
        raise FormatCorruptError(f"CSV 词法损坏: {exc}", Location(slot=slot))
    cols = reader.fieldnames or []
    missing = [c for c in required if c not in cols]
    if missing:
        raise FormatCorruptError(
            f"缺少必需列 {missing}；实际列 {cols}",
            Location(slot=slot), {"missing": missing})
    rows: list[dict[str, Any]] = []
    diags: list[Diagnostic] = []
    for i, raw_row in enumerate(reader):
        row = {c: (raw_row.get(c, "") or "").strip() for c in cols}
        if not any(row.values()):
            diags.append(Diagnostic("empty_row", "空行已保留但不参与采样",
                                    location=Location(slot=slot, row=i)))
            continue
        row["_row"] = i
        rows.append(row)
    if not rows:
        raise FormatCorruptError("没有任何数据行", Location(slot=slot))
    return ParsedCurve(rows=rows, columns=list(cols), diagnostics=diags)


def curve_payload(text: str, slot: str = SLOT_POWER,
                  time_unit: str = "s", power_unit: str = "ucal/s"
                  ) -> dict[str, Any]:
    """把原始 CSV 包装成材料载荷：保留 raw 文本，另存逐格原始表示。"""
    parsed = parse_curve_csv(text, slot)
    times = [r["time"] for r in parsed.rows]
    powers = [r["power"] for r in parsed.rows]
    gaps = []
    for r in parsed.rows:
        flag = (r.get("gap") or r.get("status") or "").lower()
        if flag in {"gap", "missing", "1", "true"}:
            gaps.append(int(r["_row"]))
    return {
        "format": "csv",
        "raw": text,
        "columns": parsed.columns,
        "time_unit": time_unit,
        "power_unit": power_unit,
        "samples": [{"row": r["_row"], "time_raw": r["time"],
                     "power_raw": r["power"],
                     "gap": int(r["_row"]) in gaps} for r in parsed.rows],
        "gap_rows": gaps,
        "diagnostics": [d.to_dict() for d in parsed.diagnostics],
        "time_base": "seconds_from_acquisition_start",
    }


def injection_payload(entries: list[dict[str, Any]],
                      volume_unit: str = "uL",
                      time_unit: str = "s") -> dict[str, Any]:
    """注射时序：体积、时间、序号全部保留原始表示。"""
    if not entries:
        raise FormatCorruptError("注射时序为空",
                                 Location(slot=SLOT_INJECTIONS))
    rows = []
    for i, e in enumerate(entries):
        if "volume" not in e or "time" not in e:
            raise FormatCorruptError(
                "每条注射必须含 volume 与 time",
                Location(slot=SLOT_INJECTIONS, row=i,
                         field="volume,time"))
        rows.append({
            "ordinal": int(e.get("ordinal", i + 1)),
            "volume_raw": e["volume"],
            "time_raw": e["time"],
            "label": e.get("label", f"I{int(e.get('ordinal', i + 1))}"),
        })
    rows.sort(key=lambda r: r["ordinal"])
    return {
        "format": "structured",
        "volume_unit": e.get("volume_unit", volume_unit),
        "time_unit": e.get("time_unit", time_unit),
        "time_base": "seconds_from_acquisition_start",
        "injections": rows,
    }
