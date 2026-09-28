"""不可变材料层：.itcmat 领域材料解析与分版。

材料类别（各自独立版本化）：
  power_series  功率采样（role=sample|blank）
  injections    注射体积与时间（role=sample|blank）
  concentrations 注射器/池浓度
  temperature   温度
  cell_volume   池体积
  baseline      基线曲线（与样品功率曲线分开）

导入只解析与校验：保留原始文本与每个原始 token；规范化发生在派生层。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import BudgetExhaustedError, Diagnostic, FormatCorruptError, SemanticError
from .units import Quantity, parse_quantity

MAGIC = "@ITC-MATERIAL/1"

POWER = "power_series"
INJECTIONS = "injections"
CONCENTRATIONS = "concentrations"
TEMPERATURE = "temperature"
CELL_VOLUME = "cell_volume"
BASELINE = "baseline"

COLUMN_KINDS = {POWER, INJECTIONS, BASELINE}
SCALAR_KINDS = {CONCENTRATIONS, TEMPERATURE, CELL_VOLUME}
ALL_KINDS = COLUMN_KINDS | SCALAR_KINDS

REQUIRED_COLUMNS = {
    POWER: ["time", "power"],
    BASELINE: ["time", "power"],
    INJECTIONS: ["ordinal", "time", "volume"],
}
SCALAR_KEYS = {
    CONCENTRATIONS: ["syringe", "cell"],
    TEMPERATURE: ["temperature"],
    CELL_VOLUME: ["cell_volume"],
}

_HEADER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]+)\])?$")


@dataclass
class ColumnData:
    name: str
    declared_unit: Optional[str]
    basis: Optional[str]
    quantities: List[Quantity]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "declared_unit": self.declared_unit,
            "basis": self.basis,
            "quantities": [q.as_dict() for q in self.quantities],
        }


@dataclass
class MaterialVersion:
    material_id: str
    kind: str
    role: str
    label: str
    source_name: str
    ordinal: int
    raw_text: str
    columns: Dict[str, ColumnData] = field(default_factory=dict)
    scalars: Dict[str, Quantity] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "kind": self.kind,
            "role": self.role,
            "label": self.label,
            "source_name": self.source_name,
            "ordinal": self.ordinal,
            "columns": {k: c.as_dict() for k, c in self.columns.items()},
            "scalars": {k: q.as_dict() for k, q in self.scalars.items()},
            "meta": self.meta,
            "diagnostics": self.diagnostics,
        }


def content_id(raw_text: str) -> str:
    return "mat_" + hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]


def parse_material(raw_text: str, source_name: str = "<inline>",
                   max_lines: int = 2_000_000, max_bytes: int = 64 * 1024 * 1024) -> MaterialVersion:
    """解析一份 .itcmat 文本。

    抛出 FormatCorruptError / BudgetExhaustedError / SemanticError，
    三者语义不同；数值/单位问题不抛异常，作为诊断留在对应位置。
    """
    if len(raw_text.encode("utf-8")) > max_bytes:
        raise BudgetExhaustedError(
            "材料超过字节预算", [Diagnostic("byte_budget", f"超过 {max_bytes} 字节", source_name)])
    lines = raw_text.splitlines()
    if len(lines) > max_lines:
        raise BudgetExhaustedError(
            "材料超过行数预算", [Diagnostic("line_budget", f"超过 {max_lines} 行", f"{source_name}")])

    diags: List[Diagnostic] = []
    idx = _skip_comments(lines, 0)
    if idx >= len(lines) or lines[idx].strip() != MAGIC:
        raise FormatCorruptError(
            "缺少材料魔数", [Diagnostic("bad_magic", f"首行必须是 {MAGIC}", f"{source_name}:L1")])

    meta: Dict[str, str] = {}
    section: Optional[str] = None
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    basis: Optional[str] = None

    i = idx + 1
    while i < len(lines):
        lineno = i + 1
        line = lines[i]
        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("["):
            if not stripped.endswith("]"):
                raise FormatCorruptError("区段名格式错误",
                                         [Diagnostic("bad_section", f"损坏的区段标记: {stripped!r}",
                                                     f"{source_name}:L{lineno}")])
            name = stripped[1:-1].strip()
            if name != "data":
                raise FormatCorruptError("未知区段",
                                         [Diagnostic("bad_section", f"未知区段 [{name}]",
                                                     f"{source_name}:L{lineno}")])
            if section is not None:
                raise FormatCorruptError("重复区段",
                                         [Diagnostic("dup_section", "只允许一个 [data] 区段",
                                                     f"{source_name}:L{lineno}")])
            section = name
            if i >= len(lines):
                raise FormatCorruptError("[data] 区段缺少表头",
                                         [Diagnostic("empty_data", "[data] 后无表头",
                                                     f"{source_name}:L{lineno}")])
            hdr_line = lines[i]
            i += 1
            header = [tok.strip() for tok in _split_row(hdr_line)]
            if any(not h for h in header):
                raise FormatCorruptError("空列名",
                                         [Diagnostic("bad_header", "表头存在空列名",
                                                     f"{source_name}:L{i}")])
            continue
        if section == "data":
            tokens = _split_row(stripped)
            if header is not None and len(tokens) != len(header):
                raise FormatCorruptError(
                    "数据行列数与表头不符",
                    [Diagnostic("row_arity", f"期望 {len(header)} 列, 实际 {len(tokens)} 列",
                                f"{source_name}:L{lineno}")])
            rows.append(tokens)
        else:
            if ":" not in stripped:
                raise FormatCorruptError("元数据行缺少 ':'",
                                         [Diagnostic("bad_meta", f"无法识别的行: {stripped!r}",
                                                     f"{source_name}:L{lineno}")])
            key, _, val = stripped.partition(":")
            key = key.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise FormatCorruptError("非法元数据键",
                                         [Diagnostic("bad_meta_key", key, f"{source_name}:L{lineno}")])
            if key in meta:
                raise FormatCorruptError("重复元数据键",
                                         [Diagnostic("dup_meta", key, f"{source_name}:L{lineno}")])
            meta[key] = val.strip()

    kind = meta.get("kind", "")
    if kind not in ALL_KINDS:
        raise SemanticError("未知或缺少材料类别",
                            [Diagnostic("unknown_kind", f"kind={kind!r} 不在 {sorted(ALL_KINDS)}",
                                        f"{source_name}:meta:kind")])
    role = meta.get("role", "sample" if kind != BASELINE else "baseline")
    if role not in {"sample", "blank", "baseline"}:
        raise SemanticError("未知材料角色",
                            [Diagnostic("unknown_role", f"role={role!r}", f"{source_name}:meta:role")])
    label = meta.get("label", source_name)
    basis = meta.get("basis", "experiment_t0" if kind in COLUMN_KINDS else None)

    mat = MaterialVersion(
        material_id=content_id(raw_text),
        kind=kind,
        role=role,
        label=label,
        source_name=source_name,
        ordinal=0,
        raw_text=raw_text,
        meta=meta,
    )

    if kind in COLUMN_KINDS:
        _build_columns(mat, kind, header, rows, basis, source_name, diags)
    else:
        _build_scalars(mat, kind, meta, diags, source_name)

    fatal = [d for d in diags if d.severity == "error"]
    mat.diagnostics = [d.as_dict() for d in diags]
    if fatal:
        raise SemanticError(f"材料 {kind} 语义无解", fatal)
    return mat


def _skip_comments(lines: List[str], start: int) -> int:
    i = start
    while i < len(lines):
        s = lines[i].strip()
        if s and not s.startswith("#"):
            return i
        i += 1
    return i


def _split_row(line: str) -> List[str]:
    # 逗号或空白分隔；不做引号（领域数值无逗号）
    return [tok for tok in re.split(r"[,\s]+", line.strip()) if tok]


def _build_columns(mat: MaterialVersion, kind: str, header: Optional[List[str]],
                   rows: List[List[str]], basis: Optional[str], source: str,
                   diags: List[Diagnostic]) -> None:
    if not header:
        diags.append(Diagnostic("missing_data", f"{kind} 材料缺少 [data] 列", f"{source}:[data]"))
        return
    specs: List[tuple] = []
    seen = set()
    for h in header:
        m = _HEADER_RE.match(h)
        if not m:
            diags.append(Diagnostic("bad_header", f"无法解析列头 {h!r}", f"{source}:[data]:{h}"))
            continue
        name, unit = m.group(1), m.group(2)
        if name in seen:
            diags.append(Diagnostic("dup_column", f"重复列 {name}", f"{source}:[data]:{name}"))
        seen.add(name)
        specs.append((name, unit))
    if any(d.code == "bad_header" for d in diags):
        return
    names = [s[0] for s in specs]
    for req in REQUIRED_COLUMNS[kind]:
        if req not in names:
            diags.append(Diagnostic("missing_column", f"{kind} 缺少必需列 {req!r}",
                                    f"{source}:[data]:{req}", ))
    by_index = {name: j for j, (name, _) in enumerate(specs)}
    for name, unit in specs:
        col_vals = [row[by_index[name]] for row in rows]
        col_basis = "experiment_t0" if name in {"time"} else None
        qs, bad = _parse_col(col_vals, unit, col_basis, f"{source}:[data]:{name}")
        for q in bad:
            for d in q.diagnostics:
                diags.append(Diagnostic(d["code"], d["message"], d["location"], "warning"))
        mat.columns[name] = ColumnData(name, unit, col_basis, qs)
    # ordinal 列必须为整数语义
    if kind == INJECTIONS and "ordinal" in mat.columns:
        for q in mat.columns["ordinal"].quantities:
            if q.ok and q.value is not None and abs(q.value - round(q.value)) > 1e-9:
                diags.append(Diagnostic("ordinal_not_integer", f"注射序号非整数: {q.raw!r}",
                                        f"{source}:[data]:ordinal", "error"))


def _parse_col(values: List[Any], unit: Optional[str], basis: Optional[str], location: str):
    from .units import parse_column
    return parse_column(values, unit, basis, location)


def _build_scalars(mat: MaterialVersion, kind: str, meta: Dict[str, str],
                   diags: List[Diagnostic], source: str) -> None:
    for key in SCALAR_KEYS[kind]:
        if key not in meta:
            diags.append(Diagnostic("missing_scalar", f"{kind} 缺少标量 {key!r}",
                                    f"{source}:meta:{key}"))
            continue
        q = parse_quantity(meta[key], None, None, f"{source}:meta:{key}")
        if not q.ok:
            for d in q.diagnostics:
                diags.append(Diagnostic(d["code"], d["message"], d["location"], "error"))
        mat.scalars[key] = q


def render_material(kind: str, role: str, label: str, columns: Optional[Dict[str, Any]] = None,
                    scalars: Optional[Dict[str, str]] = None, meta: Optional[Dict[str, str]] = None) -> str:
    """把结构化数据渲染回 .itcmat 文本（供案例包构建器使用）。"""
    out: List[str] = [MAGIC, f"kind: {kind}", f"role: {role}", f"label: {label}"]
    for k, v in (meta or {}).items():
        if k not in {"kind", "role", "label"}:
            out.append(f"{k}: {v}")
    if scalars:
        for k, v in scalars.items():
            out.append(f"{k}: {v}")
    if columns:
        names = list(columns.keys())
        units = columns.get("__units__", {})
        out.append("[data]")
        out.append(" ".join(f"{n}[{units[n]}]" if n in units else n for n in names if n != "__units__"))
        nrows = max(len(columns[n]) for n in names if n != "__units__")
        for r in range(nrows):
            out.append(" ".join(str(columns[n][r]) for n in names if n != "__units__"))
    return "\n".join(out) + "\n"
