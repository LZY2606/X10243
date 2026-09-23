"""原始材料解析（导入层）。

只解析、不规范化：数值逐字保留，单位逐字保留，列内允许逐行单位。
三种失败严格区分：

* FormatCorruptError      结构损坏（无法定位行列时也要给出输入路径）
* BudgetExhaustedError    资源预算耗尽（点数 / 行数上限）
* SemanticError + Quantity.diagnostics  语义无解 / 个别不可转换值

无法转换的值留在原位置（canonical=None + 诊断），绝不为画图而丢。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from .units import (BudgetExhaustedError, Diagnostic, FormatCorruptError,
                    SemanticError, parse_quantity)

DEFAULT_MAX_POINTS = 500_000
DEFAULT_MAX_ROWS = 100_000
REQUIRED_COLUMNS = ("t", "power")


@dataclass
class ParsedColumn:
    name: str
    raw: list[str] = field(default_factory=list)
    unit: list[str | None] = field(default_factory=list)
    canonical: list[float | None] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw": self.raw,
            "unit": self.unit,
            "canonical": self.canonical,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass
class ParsedTable:
    header: list[str]
    columns: dict[str, ParsedColumn]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "header": self.header,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _split_header_cell(cell: str) -> tuple[str, str | None]:
    """``power[uJ/s]`` -> (``power``, ``uJ/s``)；无单位标注则为 None。"""
    text = cell.strip()
    if "[" in text and text.endswith("]"):
        name, unit = text[:-1].split("[", 1)
        return name.strip(), unit.strip()
    return text, None


def parse_csv(blob: str, source: str, required: tuple[str, ...] = (),
              max_rows: int = DEFAULT_MAX_ROWS) -> ParsedTable:
    if not isinstance(blob, str):
        raise FormatCorruptError("输入不是文本（期望 UTF-8 CSV）", source)
    try:
        reader = csv.reader(io.StringIO(blob))
        rows = list(reader)
    except csv.Error as exc:
        raise FormatCorruptError(f"CSV 结构无法解析: {exc}", source)
    if not rows:
        raise FormatCorruptError("空文件：缺少表头行", f"{source}#header")
    if len(rows) - 1 > max_rows:
        raise BudgetExhaustedError(f"数据行 {len(rows) - 1} 超过上限 {max_rows}",
                                   f"{source}#rows", limit=max_rows)
    header_cells = rows[0]
    if any(not cell.strip() for cell in header_cells):
        raise FormatCorruptError("表头存在空列名", f"{source}#header")
    header = []
    header_units: dict[str, str | None] = {}
    seen: set[str] = set()
    for cell in header_cells:
        name, unit = _split_header_cell(cell)
        if name in seen:
            raise FormatCorruptError(f"表头列重复: {name!r}", f"{source}#header")
        seen.add(name)
        header.append(name)
        header_units[name] = unit
    missing = [name for name in required if name not in seen]
    if missing:
        raise FormatCorruptError(f"缺少必需列: {', '.join(missing)}（实际: {', '.join(header)}）",
                                 f"{source}#header")
    cols = {name: ParsedColumn(name) for name in header}
    diagnostics: list[Diagnostic] = []
    for r, row in enumerate(rows[1:], start=1):
        if not row or all(not c.strip() for c in row):
            continue
        if len(row) != len(header):
            raise FormatCorruptError(
                f"列数 {len(row)} 与表头 {len(header)} 不一致", f"{source}#row{r}")
        for name, cell in zip(header, row):
            cols[name].raw.append(cell.strip())
            cols[name].unit.append(header_units[name])
    return ParsedTable(header=header, columns=cols, diagnostics=diagnostics)


def _convert_column(table: ParsedTable, name: str, kind: str, source: str) -> None:
    col = table.columns[name]
    for i, raw in enumerate(col.raw):
        q = parse_quantity(raw, kind, col.unit[i], f"{source}#row{i + 1}.{name}")
        col.canonical.append(q.canonical)
        col.diagnostics.extend(q.diagnostics)


def parse_power(blob: str, source: str, max_points: int = DEFAULT_MAX_POINTS) -> dict:
    """功率采样：``t[s],power[uJ/s]``（单位列标注仅给缺省单位，逐行可覆盖）。"""
    table = parse_csv(blob, source, REQUIRED_COLUMNS, max_rows=max_points)
    n = len(table.columns["t"].raw)
    if n == 0:
        raise SemanticError("SEMANTIC_EMPTY_SERIES", f"{source}#rows", "功率曲线没有任何采样点")
    if n > max_points:
        raise BudgetExhaustedError(f"采样点 {n} 超过上限 {max_points}", f"{source}#rows", max_points)
    _convert_column(table, "t", "time", source)
    _convert_column(table, "power", "power", source)
    payload = {
        "source": source,
        "coordinate_basis": {"time": "s", "power": "uJ/s", "t_zero": "第一条采样的时间原点"},
        "n": n,
        "table": table.to_dict(),
    }
    return payload


def parse_injections(blob: str, source: str, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """注射时序：序号、时间、体积（, 可选 stir_rate[rpm]，保留为不可规范化原始量）。"""
    table = parse_csv(blob, source, ("ordinal", "t", "volume"), max_rows=max_rows)
    for name, kind in (("ordinal", "dimensionless"), ("t", "time"), ("volume", "volume")):
        _convert_column(table, name, kind, source)
    ordinals = table.columns["ordinal"]
    seen: set[int] = set()
    for i, val in enumerate(ordinals.canonical):
        path = f"{source}#row{i + 1}.ordinal"
        if val is None:
            continue
        if val <= 0 or abs(val - round(val)) > 1e-9:
            table.diagnostics.append(
                Diagnostic("SEMANTIC_ORDINAL", path, f"注射序号必须为正整数，得到 {ordinals.raw[i]!r}",
                           ordinals.raw[i]))
            continue
        ordinal = int(round(val))
        if ordinal in seen:
            table.diagnostics.append(
                Diagnostic("SEMANTIC_ORDINAL_DUP", path, f"注射序号重复: {ordinal}", ordinals.raw[i]))
        seen.add(ordinal)
    if "stir_rate" in table.columns:
        col = table.columns["stir_rate"]
        for i, raw in enumerate(col.raw):
            q = parse_quantity(raw, "stir_rate", col.unit[i], f"{source}#row{i + 1}.stir_rate")
            col.canonical.append(q.canonical)
            col.diagnostics.extend(q.diagnostics)
    return {"source": source, "coordinate_basis": {"time": "s", "volume": "uL"},
            "table": table.to_dict()}


def parse_concentrations(blob: str, source: str) -> dict:
    """浓度与实验环境：单行标量（池/注射器浓度、温度、池体积），缺值给语义诊断。"""
    table = parse_csv(blob, source, ("field", "value"))
    fields = table.columns["field"].raw
    values = table.columns["value"]
    expected_units = {
        "cell_conc": "concentration",
        "syringe_conc": "concentration",
        "temperature": "temperature",
        "cell_volume": "volume",
    }
    rows = []
    present: dict[str, int] = {}
    for i, fname in enumerate(fields):
        unit = values.unit[i]
        kind = expected_units.get(fname.strip())
        if kind is None:
            diag = Diagnostic("SEMANTIC_UNKNOWN_FIELD", f"{source}#row{i + 1}.field",
                              f"未知环境字段 {fname.strip()!r}（保留原值，不参与计算）", fname)
            rows.append({"field": fname.strip(), "raw": values.raw[i], "unit": unit,
                         "kind": None, "canonical": None, "diagnostics": [diag.to_dict()]})
            continue
        q = parse_quantity(values.raw[i], kind, unit, f"{source}#row{i + 1}.value")
        present[fname.strip()] = i
        rows.append({"field": fname.strip(), "raw": q.raw, "unit": q.unit, "kind": kind,
                     "canonical": q.canonical,
                     "canonical_unit": q.canonical_unit,
                     "diagnostics": [d.to_dict() for d in q.diagnostics]})
    for fname, kind in expected_units.items():
        if fname not in present:
            table.diagnostics.append(
                Diagnostic("SEMANTIC_MISSING_FIELD", f"{source}#fields",
                           f"缺少环境字段 {fname}；不得用默认值伪装成功", None))
    return {"source": source, "coordinate_basis": {"concentration": "mM", "temperature": "K",
                                                   "volume": "uL"},
            "rows": rows, "diagnostics": [d.to_dict() for d in table.diagnostics]}


def parse_baseline(blob: str, source: str) -> dict:
    """基线曲线：时间-功率基线，与样品/空白功率曲线分开版本化。"""
    payload = parse_power(blob, source)
    payload["coordinate_basis"]["role"] = "baseline"
    return payload


def parse_blank(blob: str, source: str) -> dict:
    """空白对照功率曲线（稀释热实验），独立版本化。"""
    payload = parse_power(blob, source)
    payload["coordinate_basis"]["role"] = "blank"
    return payload
