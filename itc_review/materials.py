"""不可变材料：导入解析与版本载荷构造。

每种材料独立版本化：power / schedule / concentration / conditions / baseline / blank。
导入时保留原始表示（raw 文本、未解析单元格留在原位置）。
"""
from __future__ import annotations

import csv
import re
import io
import json
from dataclasses import dataclass, field

from .errors import Diagnostic, FormatCorrupted
from .ids import digest
from .units import parse_measure

MAX_POWER_ROWS = 200_000

TIME_KINDS = ("time", "power")


def _read_rows(text: str, path: str) -> tuple[list[str], list[dict[str, str]]]:
    if not text or not text.strip():
        raise FormatCorrupted(f"{path} 为空文件")
    try:
        reader = csv.DictReader(io.StringIO(text))
    except csv.Error as exc:
        raise FormatCorrupted(f"{path} CSV 无法解析", [Diagnostic("CSV_ERROR", path, str(exc))])
    if reader.fieldnames is None:
        raise FormatCorrupted(f"{path} 缺少表头", [Diagnostic("MISSING_HEADER", path, "无表头行")])
    rows = list(reader)
    if not rows:
        raise FormatCorrupted(f"{path} 无数据行", [Diagnostic("NO_ROWS", path, "只有表头")])
    return list(reader.fieldnames), rows


def _base(name: str) -> str:
    """去掉列名中的单位标注：Time (s) / power_uw / 注射体积(µL) -> 基础名。"""
    low = name.strip().lower()
    low = re.split(r"[（(\[【]", low, 1)[0].strip()
    low = re.sub(r"_(s|sec|min|h|uw|µw|mw|ul|µl|ml|l)$", "", low)
    return low


def _pick(fields_: list[str], candidates: tuple[str, ...]) -> str | None:
    norm = {_base(f): f for f in fields_}
    for cand in candidates:
        if cand.lower() in norm:
            return norm[cand.lower()]
    return None


_HEADER_UNITS = {
    "time": {"s": "s", "sec": "s", "second": "s", "seconds": "s",
             "min": "min", "minute": "min", "minutes": "min", "h": "h", "hour": "h"},
    "power": {"uw": "uW", "µw": "µW", "mw": None, "cal/s": "cal/s", "j/s": "J/s", "w": None},
    "volume": {"ul": "uL", "µl": "µL", "ml": "mL", "l": "L"},
}


def _unit_from_header(header: str | None, kind: str) -> str | None:
    """从列名识别单位基准，例如 'Time (s)'、'power_uw'、'注射体积(µL)'。"""
    if not header:
        return None
    low = header.strip().lower()
    table = _HEADER_UNITS.get(kind, {})
    # 括号
    import re as _re
    m = _re.findall(r"[（(\[【]\s*([^)）\]】]+?)\s*[)）\]】]", low)
    for token in m:
        token = token.strip()
        if token in table:
            return table[token]
    # 后缀 _s / _uw / _ul
    for token in sorted(table, key=len, reverse=True):
        if low.endswith("_" + token) or low.endswith(" " + token):
            return table[token]
    return None


def parse_power(text: str, source: str = "power.csv") -> dict:
    """功率采样：列 time / power（接受常见别名），保留时间基准与原始单元格。"""
    fields_, rows = _read_rows(text, source)
    if len(rows) > MAX_POWER_ROWS:
        from .errors import BudgetExhausted

        raise BudgetExhausted(
            f"功率采样行数 {len(rows)} 超过预算 {MAX_POWER_ROWS}",
            [Diagnostic("ROW_BUDGET", source, f"limit={MAX_POWER_ROWS}", len(rows))],
        )
    tc = _pick(fields_, ("time_s", "time", "t", "时间(s)", "时间"))
    pc = _pick(fields_, ("power_uw", "power", "p", "功率(uw)", "功率", "dp"))
    missing = []
    if tc is None:
        missing.append(Diagnostic("MISSING_COLUMN", source, "缺少时间列（time）"))
    if pc is None:
        missing.append(Diagnostic("MISSING_COLUMN", source, "缺少功率列（power）"))
    if missing:
        raise FormatCorrupted(f"{source} 列不完整", missing)

    samples: list[dict] = []
    diagnostics: list[Diagnostic] = []
    prev_t = None
    for i, row in enumerate(rows):
        path = f"{source}#row{i + 2}"
        tm = parse_measure(row.get(tc, ""), "time", _unit_from_header(tc, "time"), f"{path}.time")
        pm = parse_measure(row.get(pc, ""), "power", _unit_from_header(pc, "power"), f"{path}.power")
        nominal_gap_from_prev = False
        if tm.valid and prev_t is not None and tm.canonical < prev_t:
            diagnostics.append(
                Diagnostic(
                    "TIME_NOT_MONOTONIC",
                    f"{path}.time",
                    f"时间非单调递增: {tm.canonical} < {prev_t}",
                    tm.raw,
                )
            )
        if tm.valid:
            prev_t = tm.canonical
        samples.append(
            {
                "index": i,
                "time": tm.to_dict(),
                "power": pm.to_dict(),
                "time_raw": tm.raw,
                "power_raw": pm.raw,
            }
        )
        for d in tm.diagnostics + pm.diagnostics:
            diagnostics.append(d)

    valid_t = [s["time"]["canonical"] for s in samples if s["time"]["valid"]]
    payload = {
        "kind": "power",
        "source": source,
        "columns": {"time": tc, "power": pc},
        "time_unit": "s",
        "power_unit": "µW",
        "n_rows": len(samples),
        "n_valid_time": len(valid_t),
        "samples": samples,
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    payload["id"] = digest("mat_power", _summary(payload))
    return payload


def _summary(payload: dict) -> dict:
    """内容摘要：覆盖全部原始单元格与有效性，等价输入得到等价身份。"""
    if payload["kind"] == "schedule":
        rows = [[s["index"], s["time_raw"], s["volume_raw"],
                 s["time"]["valid"], s["volume"]["valid"]] for s in payload["injections"]]
        return {"kind": "schedule", "source": payload["source"],
                "columns": payload["columns"], "n_rows": payload["n_injections"], "samples": rows}
    return {
        "kind": payload["kind"],
        "source": payload["source"],
        "columns": payload["columns"],
        "n_rows": payload["n_rows"],
        "samples": [
            [s["index"], s["time_raw"], s["power_raw"], s["time"]["valid"], s["power"]["valid"]]
            for s in payload["samples"]
        ],
    }


def parse_schedule(text: str, source: str = "schedule.csv") -> dict:
    """注射时序：注射时间与注射体积，按源顺序编号 index=1..N。"""
    fields_, rows = _read_rows(text, source)
    tc = _pick(fields_, ("time_s", "injection_time_s", "time", "t", "注射时间"))
    vc = _pick(fields_, ("volume_ul", "volume", "v", "注射体积", "inj_vol_ul"))
    missing = []
    if tc is None:
        missing.append(Diagnostic("MISSING_COLUMN", source, "缺少注射时间列"))
    if vc is None:
        missing.append(Diagnostic("MISSING_COLUMN", source, "缺少注射体积列"))
    if missing:
        raise FormatCorrupted(f"{source} 列不完整", missing)

    injections: list[dict] = []
    diagnostics: list[Diagnostic] = []
    for i, row in enumerate(rows):
        path = f"{source}#row{i + 2}"
        tm = parse_measure(row.get(tc, ""), "time", _unit_from_header(tc, "time"), f"{path}.time")
        vm = parse_measure(row.get(vc, ""), "volume", _unit_from_header(vc, "volume"), f"{path}.volume")
        injections.append(
            {
                "index": i + 1,
                "time": tm.to_dict(),
                "volume": vm.to_dict(),
                "time_raw": tm.raw,
                "volume_raw": vm.raw,
            }
        )
        for d in tm.diagnostics + vm.diagnostics:
            diagnostics.append(d)
    payload = {
        "kind": "schedule",
        "source": source,
        "columns": {"time": tc, "volume": vc},
        "n_injections": len(injections),
        "injections": injections,
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    payload["id"] = digest("mat_schedule", _summary(payload))
    return payload


def parse_concentration(obj: dict, source: str = "concentration.json") -> dict:
    """浓度：注射器浓度与池内分析物浓度。raw 保留原始表示。"""
    diagnostics: list[Diagnostic] = []
    syr_unit = obj.get("syringe_unit")
    cell_unit = obj.get("cell_unit")
    syr = parse_measure(obj.get("syringe", ""), "concentration", syr_unit, f"{source}.syringe")
    cell = parse_measure(obj.get("cell", ""), "concentration", cell_unit, f"{source}.cell")
    for m in (syr, cell):
        diagnostics.extend(m.diagnostics)
    payload = {
        "kind": "concentration",
        "source": source,
        "syringe": syr.to_dict(),
        "cell": cell.to_dict(),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    payload["id"] = digest(
        "mat_conc",
        [payload["source"], payload["syringe"]["raw"], payload["cell"]["raw"]],
    )
    return payload


def parse_conditions(obj: dict, source: str = "conditions.json") -> dict:
    """温度、池体积等池条件。"""
    diagnostics: list[Diagnostic] = []
    temp = parse_measure(obj.get("temperature", ""), "temperature", obj.get("temperature_unit"), f"{source}.temperature")
    vcell = parse_measure(obj.get("cell_volume", ""), "volume", obj.get("cell_volume_unit"), f"{source}.cell_volume")
    diagnostics.extend(temp.diagnostics + vcell.diagnostics)
    payload = {
        "kind": "conditions",
        "source": source,
        "temperature": temp.to_dict(),
        "cell_volume": vcell.to_dict(),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    payload["id"] = digest(
        "mat_cond",
        [payload["source"], payload["temperature"]["raw"], payload["cell_volume"]["raw"]],
    )
    return payload


def parse_baseline(obj: dict, source: str = "baseline.json") -> dict:
    """独立版本化的基线曲线（time/power 对齐到功率采样的时间基准）。"""
    fields_, rows = _read_baseline_rows(obj, source)
    samples = []
    diagnostics: list[Diagnostic] = []
    for i, (traw, praw) in enumerate(rows):
        tm = parse_measure(traw, "time", path=f"{source}#p{i}")
        pm = parse_measure(praw, "power", path=f"{source}#p{i}")
        samples.append({"index": i, "time": tm.to_dict(), "power": pm.to_dict()})
        diagnostics.extend(tm.diagnostics + pm.diagnostics)
    payload = {
        "kind": "baseline",
        "source": source,
        "label": obj.get("label", source),
        "time_unit": "s",
        "power_unit": "µW",
        "n_rows": len(samples),
        "samples": samples,
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    payload["id"] = digest(
        "mat_baseline",
        [payload["source"], payload["label"], [[s["time"]["raw"], s["power"]["raw"]] for s in samples]],
    )
    return payload


def _read_baseline_rows(obj: dict, source: str) -> tuple[list[str], list[tuple]]:
    if "points" in obj:
        pts = obj["points"]
        if not isinstance(pts, list) or not pts:
            raise FormatCorrupted(f"{source}.points 为空")
        return ["time", "power"], [(p[0], p[1]) for p in pts]
    if "csv" in obj:
        fields_, rows = _read_rows(obj["csv"], source)
        tc = _pick(fields_, ("time", "t", "time_s"))
        pc = _pick(fields_, ("power", "p", "power_uw"))
        if tc is None or pc is None:
            raise FormatCorrupted(f"{source} 基线 CSV 需 time/power 列")
        return fields_, [(r[tc], r[pc]) for r in rows]
    raise FormatCorrupted(f"{source} 需要 points 或 csv", [Diagnostic("MISSING_FIELD", source, "points|csv")])


def parse_blank(text: str, source: str = "blank.csv") -> dict:
    """空白对照曲线：独立版本化。列 time,power,injection_index（对应规则关键键）。"""
    fields_, rows = _read_rows(text, source)
    tc = _pick(fields_, ("time_s", "time", "t"))
    pc = _pick(fields_, ("power_uw", "power", "p"))
    ic = _pick(fields_, ("injection_index", "inj_index", "index", "blank_index"))
    missing = []
    if tc is None:
        missing.append(Diagnostic("MISSING_COLUMN", source, "空白缺少时间列"))
    if pc is None:
        missing.append(Diagnostic("MISSING_COLUMN", source, "空白缺少功率列"))
    if missing:
        raise FormatCorrupted(f"{source} 列不完整", missing)
    samples = []
    diagnostics: list[Diagnostic] = []
    for i, row in enumerate(rows):
        path = f"{source}#row{i + 2}"
        tm = parse_measure(row.get(tc, ""), "time", _unit_from_header(tc, "time"), f"{path}.time")
        pm = parse_measure(row.get(pc, ""), "power", _unit_from_header(pc, "power"), f"{path}.power")
        idx_raw = row.get(ic, "") if ic else ""
        idx = None
        if str(idx_raw).strip() not in ("", "NA", "N/A"):
            try:
                idx = int(float(str(idx_raw).strip()))
            except ValueError:
                diagnostics.append(Diagnostic("UNPARSEABLE_VALUE", f"{path}.injection_index", "空白对应索引非整数", idx_raw))
        samples.append(
            {
                "index": i,
                "time": tm.to_dict(),
                "power": pm.to_dict(),
                "injection_index": idx,
                "injection_index_raw": idx_raw,
            }
        )
        diagnostics.extend(tm.diagnostics + pm.diagnostics)
    payload = {
        "kind": "blank",
        "source": source,
        "columns": {"time": tc, "power": pc, "injection_index": ic},
        "n_rows": len(samples),
        "samples": samples,
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    payload["id"] = digest(
        "mat_blank",
        [
            payload["source"],
            payload["columns"],
            [[s["time_raw"] if "time_raw" in s else s["time"]["raw"],
              s["power"]["raw"], s["injection_index_raw"]] for s in samples],
        ],
    )
    return payload
