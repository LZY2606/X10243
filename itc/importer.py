"""原始材料导入：功率曲线、注射时序、浓度、温度、池体积、基线、空白对照。

约定的纯文本格式（领域材料，不是通用表格）：

功率曲线 / 基线序列 / 空白（kind=power|baseline|blank）
    # time_unit: s
    # power_unit: µcal/s
    time,power
    0.0,5.01
空白可带分段：
    @@ injections
    index,time,volume

注射时序 (kind=injections)
    # volume_unit: µL
    index,time,volume

浓度 (kind=concentration)
    syringe = 0.5 mM
    cell = 0.02 mM

温度 (kind=temperature)     temperature = 25 °C
池体积 (kind=cell_volume)   cell_volume = 1.4 mL
基线标量 (kind=baseline)    baseline = 5.0 µcal/s

逐格保留原始字符串；无法转换的格子留在原位置并附 loc 诊断。
"""
from __future__ import annotations

import csv
import io
import re

from .errors import Diagnostic, FormatCorrupt
from .units import parse_measure, required

KINDS = {"power", "injections", "concentration", "temperature", "cell_volume", "baseline", "blank"}

_META = re.compile(r"^#\s*([a-z_]+)\s*[:=]\s*(.+?)\s*$", re.I)
_SECTION = re.compile(r"^@@\s*([a-z_]+)\s*$", re.I)
_KV = re.compile(r"^\s*([A-Za-z_][\w ]*?)\s*=\s*(.+?)\s*$")
TIME_HEADERS = {"time", "t", "时间", "time(s)", "time_s"}
POWER_HEADERS = {"power", "p", "功率", "power(ucal/s)", "dp", "power_ucal_s"}
INDEX_HEADERS = {"index", "i", "idx", "n", "序号", "inj", "injection"}
VOLUME_HEADERS = {"volume", "v", "vol", "体积", "dv", "volume_ul"}


def _split_meta(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    meta: dict[str, str] = {}
    rest: list[str] = []
    for line in lines:
        m = _META.match(line)
        if m:
            meta[m.group(1).strip().lower()] = m.group(2).strip()
        else:
            rest.append(line)
    return meta, rest


def _sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"_main": []}
    current = "_main"
    for line in lines:
        m = _SECTION.match(line.strip())
        if m:
            current = m.group(1).strip().lower()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return sections


def _read_rows(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    text = "\n".join(lines).strip("\n")
    if not text.strip():
        return [], []
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return [], []
    header = [c.strip().lower() for c in rows[0]]
    return header, rows[1:]


def _find_col(header: list[str], candidates: set[str]) -> int | None:
    for i, h in enumerate(header):
        token = h.strip().lower()
        if token in candidates:
            return i
    return None


def import_series(text: str, kind: str, loc_prefix: str = "") -> dict:
    """power / baseline(series) / blank 的曲线部分。"""
    raw_lines = text.splitlines()
    meta, body = _split_meta(raw_lines)
    time_unit = meta.get("time_unit", "s")
    power_unit = meta.get("power_unit", "µcal/s")
    header, rows = _read_rows(body)
    if not header:
        raise FormatCorrupt(f"{loc_prefix}缺少表头与数据行")
    ti = _find_col(header, TIME_HEADERS)
    pi = _find_col(header, POWER_HEADERS)
    if ti is None or pi is None:
        raise FormatCorrupt(
            f"{loc_prefix}表头必须包含时间列与功率列",
            [Diagnostic("missing_column", f"表头={header}", f"{loc_prefix}/header", header)],
        )
    samples: list[dict] = []
    diagnostics: list[Diagnostic] = []
    for n, row in enumerate(rows):
        loc = f"{loc_prefix}/row/{n}"
        raw_t = row[ti].strip() if ti < len(row) else ""
        raw_p = row[pi].strip() if pi < len(row) else ""
        tm = parse_measure(f"{raw_t} {time_unit}" if raw_t else "", "time", f"{loc}.time")
        pm = parse_measure(f"{raw_p} {power_unit}" if raw_p else "", "power", f"{loc}.power")
        rec = {
            "index": n,
            "raw_time": raw_t,
            "raw_power": raw_p,
            "time": tm.to_dict(),
            "power": pm.to_dict(),
            "diagnostics": [d.to_dict() for d in (tm.diagnostics + pm.diagnostics)],
        }
        samples.append(rec)
        diagnostics.extend(tm.diagnostics + pm.diagnostics)
    valid = [s for s in samples if s["time"]["canonical_value"] is not None
             and s["power"]["canonical_value"] is not None]
    if len(valid) < 2:
        raise FormatCorrupt(
            f"{loc_prefix}可解析采样点少于 2 个，无法构成曲线",
            [Diagnostic("insufficient_samples", f"有效采样 {len(valid)} 个", loc_prefix, len(valid))],
        )
    return {
        "meta": {"time_unit_declared": time_unit, "power_unit_declared": power_unit},
        "header": header,
        "samples": samples,
        "valid_sample_count": len(valid),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def import_injections(text: str, loc_prefix: str = "/injections") -> dict:
    meta, body = _split_meta(text.splitlines())
    vol_unit = meta.get("volume_unit", "µL")
    time_unit = meta.get("time_unit", "s")
    header, rows = _read_rows(body)
    if not header:
        raise FormatCorrupt(f"{loc_prefix}缺少注射时序表")
    ii = _find_col(header, INDEX_HEADERS)
    ti = _find_col(header, TIME_HEADERS)
    vi = _find_col(header, VOLUME_HEADERS)
    if ti is None or vi is None:
        raise FormatCorrupt(
            f"{loc_prefix}表头必须包含 time 与 volume 列",
            [Diagnostic("missing_column", f"表头={header}", f"{loc_prefix}/header", header)],
        )
    records: list[dict] = []
    diagnostics: list[Diagnostic] = []
    for n, row in enumerate(rows):
        loc = f"{loc_prefix}/row/{n}"
        raw_idx = row[ii].strip() if ii is not None and ii < len(row) else str(n + 1)
        raw_t = row[ti].strip() if ti < len(row) else ""
        raw_v = row[vi].strip() if vi < len(row) else ""
        try:
            idx = int(float(raw_idx))
        except (ValueError, TypeError):
            idx = n + 1
            diagnostics.append(Diagnostic("bad_index", f"注射序号无法解析: {raw_idx!r}", f"{loc}.index", raw_idx))
        tm = parse_measure(f"{raw_t} {time_unit}" if raw_t else "", "time", f"{loc}.time")
        vm = parse_measure(f"{raw_v} {vol_unit}" if raw_v else "", "volume", f"{loc}.volume")
        records.append({
            "index": idx,
            "raw_index": raw_idx,
            "raw_time": raw_t,
            "raw_volume": raw_v,
            "time": tm.to_dict(),
            "volume": vm.to_dict(),
            "diagnostics": [d.to_dict() for d in (tm.diagnostics + vm.diagnostics)],
        })
        diagnostics.extend(tm.diagnostics + vm.diagnostics)
    valid = [r for r in records if r["time"]["canonical_value"] is not None
             and r["volume"]["canonical_value"] is not None]
    if not valid:
        raise FormatCorrupt(
            f"{loc_prefix}没有可解析的注射记录",
            [Diagnostic("insufficient_injections", "0 条有效注射", loc_prefix, 0)],
        )
    return {
        "meta": {"volume_unit_declared": vol_unit, "time_unit_declared": time_unit},
        "header": header,
        "records": records,
        "valid_count": len(valid),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def _kv_lines(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _KV.match(s)
        if m:
            out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def import_concentration(text: str) -> dict:
    _, body = _split_meta(text.splitlines())
    kv = _kv_lines(body)
    syr_raw = kv.get("syringe") or kv.get("ligand") or kv.get("注射")
    cell_raw = kv.get("cell") or kv.get("macromolecule") or kv.get("池")
    if not syr_raw or not cell_raw:
        raise FormatCorrupt(
            "浓度材料必须包含 syringe 与 cell 两项",
            [Diagnostic("missing_key", f"解析到的键: {sorted(kv)}", "/concentration", sorted(kv))],
        )
    syr = required(syr_raw, "concentration", "/concentration/syringe")
    cell = required(cell_raw, "concentration", "/concentration/cell")
    return {
        "syringe": syr.to_dict(),
        "cell": cell.to_dict(),
        "diagnostics": [d.to_dict() for d in syr.diagnostics + cell.diagnostics],
    }


def _first_scalar(text: str, quantity: str, keys: tuple[str, ...], loc: str) -> Measure:  # type: ignore[name-defined]
    _, body = _split_meta(text.splitlines())
    kv = _kv_lines(body)
    for k in keys:
        if k in kv:
            return required(kv[k], quantity, loc)
    # 也允许整段就是 “25 °C”
    for line in body:
        s = line.strip()
        if s and not s.startswith("#"):
            return required(s, quantity, loc)
    raise FormatCorrupt(f"{loc} 缺少数值", [Diagnostic("missing_value", "未找到标量行", loc, None)])


def import_temperature(text: str):
    m = _first_scalar(text, "temperature", ("temperature", "temp", "t", "温度"), "/temperature")
    return {"temperature": m.to_dict(), "diagnostics": [d.to_dict() for d in m.diagnostics]}


def import_cell_volume(text: str):
    m = _first_scalar(text, "volume", ("cell_volume", "v0", "volume", "池体积"), "/cell_volume")
    return {"cell_volume": m.to_dict(), "diagnostics": [d.to_dict() for d in m.diagnostics]}


def import_baseline(text: str) -> dict:
    _, body0 = _split_meta(text.splitlines())
    body = [l for l in body0 if l.strip()]
    first = body[0].strip() if body else ""
    if "," in first:
        series = import_series(text, "baseline", "/baseline")
        series["mode"] = "series"
        return series
    kv = _kv_lines(body)
    raw = kv.get("baseline") or kv.get("b") or (first if first else "")
    if not raw:
        raise FormatCorrupt("/baseline 既非标量也非曲线",
                            [Diagnostic("missing_value", "未找到 baseline", "/baseline", None)])
    m = required(raw, "power", "/baseline")
    return {"mode": "scalar", "baseline": m.to_dict(), "diagnostics": [d.to_dict() for d in m.diagnostics]}


def import_blank(text: str) -> dict:
    meta, body = _split_meta(text.splitlines())
    sections = _sections(body)
    power_lines = sections.get("power", sections.get("_main", []))
    series = import_series("\n".join(power_lines), "blank", "/blank/power")
    out = {"mode": "blank", **series}
    inj_lines = sections.get("injections") or sections.get("inj")
    if inj_lines and any(l.strip() for l in inj_lines):
        inj_text = ("# volume_unit: " + meta.get("volume_unit", "µL") + "\n"
                    + "# time_unit: " + meta.get("time_unit", "s") + "\n"
                    + "\n".join(inj_lines))
        out["injections"] = import_injections(inj_text, "/blank/injections")
    else:
        out["injections"] = None
    return out


def import_material(kind: str, text: str) -> dict:
    if kind not in KINDS:
        raise FormatCorrupt(f"未知材料类型 {kind!r}",
                            [Diagnostic("unknown_kind", f"允许: {sorted(KINDS)}", "/kind", kind)])
    if kind == "power":
        return {"mode": "power", **import_series(text, "power", "/power")}
    if kind == "blank":
        return import_blank(text)
    if kind == "injections":
        return import_injections(text)
    if kind == "concentration":
        return import_concentration(text)
    if kind == "temperature":
        return import_temperature(text)
    if kind == "cell_volume":
        return import_cell_volume(text)
    if kind == "baseline":
        return import_baseline(text)
    raise FormatCorrupt(f"未实现的导入器 {kind}")
