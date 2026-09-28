"""原始材料解析：功率曲线、注射时序、空白曲线、条件量。

原则：保留原始表示；无法转换留在原位附诊断；不做静默默认值。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from .units import Diagnostic, TARGET, try_convert


@dataclass
class PowerSample:
    line: int
    raw_time: str
    raw_power: str
    time_s: float | None = None
    power_uW: float | None = None


@dataclass
class PowerSeries:
    raw_text: str
    declared_power_unit: str
    declared_time_unit: str
    time_basis: str  # "seconds_since_first_sample"
    samples: list[PowerSample] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass
class InjectionRow:
    line: int
    raw_index: str
    raw_time: str
    raw_volume: str
    time_s: float | None = None
    volume_uL: float | None = None


@dataclass
class Schedule:
    raw_text: str
    time_unit: str
    volume_unit: str
    injections: list[InjectionRow] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass
class Scalar:
    kind: str            # concentration | temperature | cellvolume
    raw_text: str
    raw_value: str
    raw_unit: str
    location: dict[str, Any]
    value: float | None  # 目标单位
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _read_rows(raw_text: str) -> tuple[list[dict[str, str]], list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    try:
        text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else raw_text
        reader = csv.DictReader(io.StringIO(text))
    except (UnicodeDecodeError, csv.Error) as exc:
        diagnostics.append(Diagnostic(
            "format_corrupt", "format_corrupt",
            f"CSV 无法解析: {exc}", {"offset": 0}, raw_text[:200]))
        return [], diagnostics
    if reader.fieldnames is None:
        diagnostics.append(Diagnostic(
            "missing_header", "format_corrupt",
            "CSV 缺少表头", {"line": 1}, raw_text[:200]))
        return [], diagnostics
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({(k or "").strip(): (v if v is not None else "") for k, v in row.items()})
    return rows, diagnostics


def parse_power(raw_text: str, power_unit: str, time_unit: str = "s") -> PowerSeries:
    ps = PowerSeries(raw_text=raw_text, declared_power_unit=power_unit,
                     declared_time_unit=time_unit,
                     time_basis="seconds_since_first_sample")
    rows, diag = _read_rows(raw_text)
    ps.diagnostics.extend(diag)
    required = {"time", "power"}
    for i, row in enumerate(rows, start=2):
        rt, rp = row.get("time", "").strip(), row.get("power", "").strip()
        sample = PowerSample(line=i, raw_time=rt, raw_power=rp)
        # 时间基准保持逐字；缺失/无法解析的点保留，规范化结果为 None
        sample.time_s = try_convert(
            "time", rt, time_unit, {"file": "power.csv", "line": i, "column": "time"},
            ps.diagnostics) if rt not in ("", None) else None
        if rt in ("", None):
            ps.diagnostics.append(Diagnostic(
                "missing_time", "format_corrupt", "采样缺少时间",
                {"file": "power.csv", "line": i, "column": "time"}, rt))
        sample.power_uW = try_convert(
            "power", rp, power_unit, {"file": "power.csv", "line": i, "column": "power"},
            ps.diagnostics) if rp not in ("", None) else None
        if rp in ("", None):
            ps.diagnostics.append(Diagnostic(
                "missing_power", "format_corrupt", "采样点功率缺失（数据缺口），原始位置保留",
                {"file": "power.csv", "line": i, "column": "power"}, rp))
        ps.samples.append(sample)
    return ps


def parse_schedule(raw_text: str, time_unit: str, volume_unit: str) -> Schedule:
    sch = Schedule(raw_text=raw_text, time_unit=time_unit, volume_unit=volume_unit)
    rows, diag = _read_rows(raw_text)
    sch.diagnostics.extend(diag)
    for row in rows:
        # DictReader 行号不直接暴露；用 injection 序号文本作定位锚
        line = reader_line(row, rows)
        inj = InjectionRow(line=line,
                           raw_index=row.get("injection", "").strip(),
                           raw_time=row.get("time", "").strip(),
                           raw_volume=row.get("volume", "").strip())
        loc_i = {"file": "injections.csv", "row": inj.raw_index, "column": "time"}
        inj.time_s = try_convert("time", inj.raw_time, time_unit, loc_i, sch.diagnostics)
        loc_v = {"file": "injections.csv", "row": inj.raw_index, "column": "volume"}
        inj.volume_uL = try_convert("volume", inj.raw_volume, volume_unit, loc_v, sch.diagnostics)
        sch.injections.append(inj)
    return sch


def reader_line(row: dict[str, str], rows: list[dict[str, str]]) -> int:
    for i, r in enumerate(rows, start=2):
        if r is row:
            return i
    return 0


def parse_scalar(kind: str, raw_text: str) -> Scalar:
    """条件量文件：两行 key,value 风格，如 cellvolume.csv:
    cellvolume,volume
    1.42,mL
    或 temperature.csv: temperature,degC / 25.0,degC
    """
    text = raw_text.decode() if isinstance(raw_text, bytes) else raw_text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    diagnostics: list[Diagnostic] = []
    if len(lines) < 2:
        diagnostics.append(Diagnostic(
            "scalar_short_file", "format_corrupt",
            f"{kind} 文件需要表头行和值行", {"file": f"{kind}.csv"}, raw_text[:120]))
        return Scalar(kind, raw_text, "", "", {}, None, diagnostics)
    value_cells = next(csv.reader([lines[1]]))
    raw_value = value_cells[0].strip() if value_cells else ""
    raw_unit = value_cells[1].strip() if len(value_cells) > 1 else ""
    quantity = {"concentration": "concentration", "temperature": "temperature",
                "cellvolume": "volume"}[kind]
    loc = {"file": f"{kind}.csv", "line": 2}
    sc = Scalar(kind, raw_text, raw_value, raw_unit, loc, None, diagnostics)
    sc.value = try_convert(quantity, raw_value, raw_unit or None, loc, diagnostics)
    return sc
