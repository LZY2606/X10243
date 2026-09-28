"""ITC 案例文本格式解析器（领域专用，不是通用表格）。

格式（UTF-8 文本，段落用 [SECTION] 标识）：

    [ITC-CASE v1]
    source: main_run_2026-09-28.itc
    [power dt_s]
    # t_s  power_uW
    0.0  2.10
    ...
    [protocol]
    # index  time  duration  volume
    1  60.0s  8.0s  4.0uL
    [ligant] ... 等

每个材料段落单独解析，原始文本整段保留在 Material.raw；
数值列允许 `GAP` 表示缺口；无法识别的结构抛 FormatCorrupt，
单位无法转换记录诊断并保留原值（不在这里抛错，交给语义层）。
"""
from __future__ import annotations

import re

from .errors import FormatCorrupt, Diagnostic
from .units import parse_quantity
from .model import Material, Sample, Injection

_HEADER_RE = re.compile(r"^\[ITC-CASE\s+v(\d+)\]\s*$")
_SECTION_RE = re.compile(r"^\[([a-z_]+)(?:\s+([^\]]+))?\]\s*$")
_CURVE_KINDS = {"power", "baseline", "blank"}


def _line_diag(code, message, section, lineno, value=None,
               severity="error") -> Diagnostic:
    return Diagnostic(code, message, f"section[{section}] line {lineno}",
                      severity, value)


def parse_case(text: str, source: str) -> list[Material]:
    """把整份案例文本解析成多个 Material（尚未指定版本号）。"""
    lines = text.splitlines()
    if not lines:
        raise FormatCorrupt("空文档，找不到 [ITC-CASE vN] 头")
    head = lines[0].strip()
    m = _HEADER_RE.match(head)
    if not m:
        raise FormatCorrupt(
            "首行必须是 [ITC-CASE v1] 头",
            [Diagnostic("bad_header", "缺少或损坏的案例头", "line 1",
                        value=head)])
    version_fmt = int(m.group(1))
    if version_fmt != 1:
        raise FormatCorrupt(
            f"不支持的案例格式版本 v{version_fmt}",
            [Diagnostic("unsupported_format_version",
                        f"仅支持 v1，收到 v{version_fmt}", "line 1")])

    meta: dict[str, str] = {}
    sections: list[tuple[str, str, list[str], int]] = []
    current = None
    for i, raw_line in enumerate(lines[1:], start=2):
        stripped = raw_line.strip()
        sm = _SECTION_RE.match(stripped)
        if sm:
            current = (sm.group(1), (sm.group(2) or "").strip(), [], i)
            sections.append(current)
            continue
        if current is None:
            if stripped and not stripped.startswith("#"):
                if ":" in stripped:
                    k, _, v = stripped.partition(":")
                    meta[k.strip()] = v.strip()
                else:
                    raise FormatCorrupt(
                        "段落外出现无法识别的内容",
                        [_line_diag("orphan_content", "首段之前的游离文本",
                                    "(root)", i, stripped)])
        else:
            current[2].append(raw_line)

    if not sections:
        raise FormatCorrupt("文档没有任何材料段落")

    materials: list[Material] = []
    src = meta.get("source", source)
    for kind, arg, body, start in sections:
        if kind not in (
                "power", "protocol", "ligand", "analyte",
                "temperature", "cell", "baseline", "blank",
                "blank_protocol"):
            raise FormatCorrupt(
                f"未知材料段落 [{kind}]",
                [_line_diag("unknown_section", f"未知段落 {kind}", kind,
                            start)])
        materials.append(_parse_section(kind, arg, body, start, src, text))
    return materials


def _raw_slice(lines: list[str]) -> str:
    return "\n".join(lines)


def _data_rows(body: list[str], section: str, start: int,
               ncols: int | None = None):
    """返回 (行号, 原始行, 字段列表)，跳过注释/空行。"""
    out = []
    for off, raw_line in enumerate(body):
        lineno = start + 1 + off
        s = raw_line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if ncols is not None and len(parts) != ncols:
            raise FormatCorrupt(
                f"段落 [{section}] 第 {lineno} 行列数应为 {ncols}，"
                f"实际 {len(parts)}",
                [_line_diag("bad_column_count", "列数不一致", section,
                            lineno, s)])
        out.append((lineno, s, parts))
    return out


def _num(token: str, section: str, lineno: int) -> float:
    try:
        return float(token)
    except ValueError:
        raise FormatCorrupt(
            f"段落 [{section}] 第 {lineno} 行数值损坏: {token!r}",
            [_line_diag("bad_number", "无法解析的纯数值", section, lineno,
                        token)])


def _parse_section(kind, arg, body, start, source, full_text) -> Material:
    diagnostics: list[Diagnostic] = []
    data: dict = {}
    raw = _raw_slice(body)

    if kind in _CURVE_KINDS:
        dim_unit = None
        if arg:
            dim_unit = arg
        samples: list[Sample] = []
        dt = None
        # 形如 [power dt_s] 或 [power]
        rows = _data_rows(body, kind, start)
        if not rows:
            raise FormatCorrupt(
                f"段落 [{kind}] 没有任何采样行",
                [_line_diag("empty_curve", "曲线为空", kind, start)])
        first_tokens = len(rows[0][2])
        explicit_t = first_tokens >= 2
        t0 = 0.0
        for lineno, s, parts in rows:
            if len(parts) != (2 if explicit_t else 1):
                raise FormatCorrupt(
                    f"段落 [{kind}] 第 {lineno} 行列数不一致",
                    [_line_diag("bad_column_count",
                                "同一曲线必须统一为 t p 或 p", kind, lineno,
                                s)])
            if explicit_t:
                t = _num(parts[0], kind, lineno)
                ptoken = parts[1]
            else:
                t = t0
                t0 += (dt if dt is not None else 1.0)
                ptoken = parts[0]
            if ptoken.upper() in ("GAP", "NAN", "NA"):
                samples.append(Sample(t=t, p=None, gap=True))
            else:
                samples.append(Sample(t=t, p=_num(ptoken, kind, lineno)))
        # 功率/基线/空白的 y 轴原始单位
        y_unit = "uW" if dim_unit in (None, "", "dt_s") else dim_unit
        data = {
            "samples": [s.to_dict() for s in samples],
            "y_unit_raw": y_unit if kind == "power" else "uW",
            "time_basis": "s",
            "explicit_time": explicit_t,
            "role": arg,
        }

    elif kind in ("protocol", "blank_protocol"):
        injections = []
        rows = _data_rows(body, kind, start)
        if not rows:
            raise FormatCorrupt("[protocol] 没有任何注射行",
                                [_line_diag("empty_protocol", "协议为空",
                                            kind, start)])
        for lineno, s, parts in rows:
            if len(parts) != 4:
                raise FormatCorrupt(
                    f"[protocol] 第 {lineno} 行需要 index time duration volume",
                    [_line_diag("bad_column_count", "协议需要 4 列", kind,
                                lineno, s)])
            idx = int(_num(parts[0], kind, lineno))
            qt = parse_quantity(parts[1], "time",
                                f"section[protocol] line {lineno} col 2")
            qd = parse_quantity(parts[2], "time",
                                f"section[protocol] line {lineno} col 3")
            qv = parse_quantity(parts[3], "volume",
                                f"section[protocol] line {lineno} col 4")
            for q in (qt, qd, qv):
                diagnostics.extend(q.diagnostics)
            injections.append({
                "index": idx,
                "time_q": qt.to_dict(), "duration_q": qd.to_dict(),
                "volume_q": qv.to_dict(),
            })
        # 序号必须稳定唯一
        seen = set()
        for inj in injections:
            if inj["index"] in seen:
                diagnostics.append(_line_diag(
                    "duplicate_injection_index",
                    f"注射序号 {inj['index']} 重复，无法建立稳定身份",
                    "protocol", start, inj["index"]))
            seen.add(inj["index"])
        data = {"injections": injections}

    elif kind in ("ligand", "analyte"):
        rows = _data_rows(body, kind, start, 2)
        vals = []
        for lineno, s, parts in rows:
            idx = int(_num(parts[0], kind, lineno))
            q = parse_quantity(parts[1], "concentration",
                               f"section[{kind}] line {lineno} col 2")
            diagnostics.extend(q.diagnostics)
            vals.append({"index": idx, "concentration_q": q.to_dict()})
        data = {"values": vals, "species": arg or kind}

    elif kind == "temperature":
        rows = _data_rows(body, kind, start, 1)
        q = parse_quantity(rows[0][2][0], "temperature",
                           f"section[{kind}] line {rows[0][0]}")
        diagnostics.extend(q.diagnostics)
        data = {"temperature_q": q.to_dict()}

    elif kind == "cell":
        rows = _data_rows(body, kind, start, 1)
        q = parse_quantity(rows[0][2][0], "volume",
                           f"section[{kind}] line {rows[0][0]}")
        diagnostics.extend(q.diagnostics)
        data = {"cell_volume_q": q.to_dict()}

    return Material(kind=kind, version=0, source=source, raw=raw,
                    data=data, diagnostics=[d.to_dict() for d in diagnostics])
