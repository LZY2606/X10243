"""不可变材料层（各自独立版本化）。

六类材料：power / injections / concentration / conditions / baseline / blank。
- 导入时保留原始表示（raw_text / raw 字段与原始单位字符串）。
- 派生层才规范化（canonical 值与规范单位）。
- N/A 等标记是“数据缺口”（留在原位）；无法解析/未知单位是“格式损坏/无解”，
  以带位置的诊断返回，绝不丢值。
- 材料 ID = 内容哈希（sha256），跨导出/导入保持稳定身份。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from .models import (
    BudgetExhausted, Diagnostic, FormatCorrupted, ITCError, KIND_FILE,
)
from .units import canonical, convert_temperature, is_missing_token

# 资源预算（“资源预算耗尽”必须与格式损坏、语义无解是三种不同结果）
MAX_SAMPLES = 200_000
MAX_INJECTIONS = 1_000

_DIRECTIVE_RE = re.compile(r"^#\s*([a-z_]+)\s*[:=]\s*(.*?)\s*$")


def material_id(kind: str, raw_text: str) -> str:
    return f"{kind}_{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class Series:
    """带坐标基准的序列：raw 保留原始字符串，value 为规范化结果；缺口为 None。"""
    raw: list[str | None]
    value: list[float | None]
    unit: str                       # 原始单位（原样保留）
    canon_unit: str                 # 派生规范单位
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": self.value,
            "unit": self.unit,
            "canon_unit": self.canon_unit,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Series":
        return cls(
            raw=d["raw"], value=d["value"], unit=d["unit"],
            canon_unit=d["canon_unit"],
            diagnostics=[Diagnostic.from_dict(x) for x in d.get("diagnostics", [])],
        )


def _directives_and_rows(raw_text: str) -> tuple[dict[str, str], list[list[str]]]:
    directives: dict[str, str] = {}
    rows: list[list[str]] = []
    header: list[str] | None = None
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = _DIRECTIVE_RE.match(stripped)
            if m:
                directives[m.group(1)] = m.group(2)
            continue
        cells = [c.strip() for c in stripped.split(",")]
        if header is None:
            header = cells
            continue
        rows.append(cells)
    if header is None:
        raise FormatCorrupted(
            "CSV 缺少表头",
            [Diagnostic("missing_header", "CSV 缺少表头", "header")],
        )
    directives["__header__"] = json.dumps(header, ensure_ascii=False)
    directives["__rows__"] = json.dumps(rows, ensure_ascii=False)
    return directives, rows


def _parse_series_csv(raw_text: str, domain: str, *,
                      require_time: bool = True) -> dict:
    directives, rows = _directives_and_rows(raw_text)
    header = json.loads(directives.pop("__header__"))
    rows = json.loads(directives.pop("__rows__"))
    if len(rows) > MAX_SAMPLES:
        raise BudgetExhausted(
            f"样本数 {len(rows)} 超过预算 {MAX_SAMPLES}",
            [Diagnostic("budget_samples",
                        f"样本数 {len(rows)} 超过预算 {MAX_SAMPLES}", "rows")],
        )
    expected = {"time", "power"} if require_time else {"time", "value"}
    missing_cols = expected - {h.lower() for h in header}
    if missing_cols:
        raise FormatCorrupted(
            f"缺少列 {sorted(missing_cols)}（表头: {header}）",
            [Diagnostic("missing_column",
                        f"缺少列 {sorted(missing_cols)}", "header")],
        )
    col = {h.lower(): i for i, h in enumerate(header)}
    t_unit = directives.get("time_unit", "s")
    v_unit = directives.get("power_unit" if require_time else "value_unit", "")
    if not v_unit:
        raise FormatCorrupted(
            "缺少单位声明（# power_unit / # value_unit）",
            [Diagnostic("missing_unit", "序列必须声明单位", "header")],
        )
    t_raw: list[str | None] = []
    t_val: list[float | None] = []
    v_raw: list[str | None] = []
    v_val: list[float | None] = []
    diags: list[Diagnostic] = []
    prev_t = None
    for idx, cells in enumerate(rows):
        pos = f"row {idx + 2}"
        if len(cells) < len(header):
            raise FormatCorrupted(
                f"{pos}: 列数不足（{len(cells)} < {len(header)}）",
                [Diagnostic("ragged_row", "列数不足", pos)],
            )
        t_tok, v_tok = cells[col["time"]], cells[col["power" if require_time else "value"]]
        # 时间坐标必须存在且可换算
        try:
            t_s, _ = canonical("time", t_tok, t_unit, pos)
        except FormatCorrupted as exc:
            diags.extend(exc.diagnostics)
            raise FormatCorrupted(f"{pos}: 时间坐标无效", diags)
        if prev_t is not None and t_s <= prev_t:
            raise FormatCorrupted(
                f"{pos}: 时间坐标必须严格递增（{t_s} <= {prev_t}）",
                [Diagnostic("time_not_monotonic",
                            "时间坐标必须严格递增", pos)],
            )
        prev_t = t_s
        t_raw.append(t_tok)
        t_val.append(t_s)
        if is_missing_token(v_tok):
            # 合法缺口：值留在原位（raw 保留标记，value=None）
            v_raw.append(v_tok)
            v_val.append(None)
            diags.append(Diagnostic("sample_gap", "该采样点缺失（数据缺口）",
                                    pos, severity="info", value=v_tok))
        else:
            try:
                v_canon, _ = canonical(domain, v_tok, v_unit, pos)
                v_raw.append(v_tok)
                v_val.append(v_canon)
            except FormatCorrupted as exc:
                # 无法转换：原值留在原位，诊断挂在材料上
                v_raw.append(v_tok)
                v_val.append(None)
                for d in exc.diagnostics:
                    d.pos = pos
                    diags.append(d)
    time_series = Series(t_raw, t_val, t_unit, "s")
    value_series = Series(v_raw, v_val, v_unit,
                          {"power": "uW", "energy": "uJ"}[domain], diags)
    return {
        "directives": directives,
        "time": time_series.to_dict(),
        "value": value_series.to_dict(),
        "n_samples": len(rows),
    }


def _injection_schedule(rows, header, col, directives, pos_prefix="inj"):
    if len(rows) > MAX_INJECTIONS:
        raise BudgetExhausted(
            f"注射数 {len(rows)} 超过预算 {MAX_INJECTIONS}",
            [Diagnostic("budget_injections",
                        f"注射数 {len(rows)} 超过预算 {MAX_INJECTIONS}", "rows")],
        )
    t_unit = directives.get("time_unit", "s")
    v_unit = directives.get("volume_unit", "uL")
    schedule = []
    prev_t = None
    for i, cells in enumerate(rows):
        pos = f"{pos_prefix} row {i + 2}"
        if len(cells) < len(header):
            raise FormatCorrupted(f"{pos}: 列数不足",
                                  [Diagnostic("ragged_row", "列数不足", pos)])
        raw_index = cells[col["index"]]
        if is_missing_token(raw_index):
            raise FormatCorrupted(f"{pos}: 注射序号缺失",
                                  [Diagnostic("missing_index", "注射序号不可缺失", pos)])
        try:
            ordinal = int(float(raw_index))
        except ValueError:
            raise FormatCorrupted(
                f"{pos}: 注射序号非整数 {raw_index!r}",
                [Diagnostic("bad_index", "注射序号必须是整数", pos,
                            value=raw_index)],
            )
        raw_t, raw_v = cells[col["time"]], cells[col["volume"]]
        try:
            t_s, _ = canonical("time", raw_t, t_unit, pos)
            v_ul, _ = canonical("volume", raw_v, v_unit, pos)
        except FormatCorrupted as exc:
            raise FormatCorrupted(f"{pos}: 注射时序/体积无效",
                                  exc.diagnostics)
        if prev_t is not None and t_s <= prev_t:
            raise FormatCorrupted(f"{pos}: 注射时间必须严格递增",
                                  [Diagnostic("time_not_monotonic",
                                              "注射时间必须严格递增", pos)])
        prev_t = t_s
        if v_ul <= 0:
            raise ITCError(
                f"{pos}: 注射体积必须为正",
                [Diagnostic("non_positive_volume", "注射体积必须为正", pos,
                            value=raw_v)],
            )
        schedule.append({
            "index": ordinal,
            "raw_index": raw_index,
            "time_s": t_s,
            "raw_time": raw_t,
            "time_unit": t_unit,
            "volume_uL": v_ul,
            "raw_volume": raw_v,
            "volume_unit": v_unit,
        })
    if not schedule:
        raise ITCError("注射时序为空",
                       [Diagnostic("empty_schedule", "至少需要一次注射", "rows")])
    return schedule


def parse_injections(raw_text: str) -> dict:
    directives, rows = _directives_and_rows(raw_text)
    header = json.loads(directives.pop("__header__"))
    rows = json.loads(directives.pop("__rows__"))
    required = {"index", "time", "volume"}
    missing_cols = required - {h.lower() for h in header}
    if missing_cols:
        raise FormatCorrupted(
            f"缺少列 {sorted(missing_cols)}",
            [Diagnostic("missing_column", f"缺少列 {sorted(missing_cols)}", "header")],
        )
    col = {h.lower(): i for i in header}
    schedule = _injection_schedule(rows, header, col, directives)
    indices = [s["index"] for s in schedule]
    if len(set(indices)) != len(indices):
        raise ITCError("注射序号重复",
                       [Diagnostic("duplicate_index", "注射序号必须唯一", "rows")])
    return {
        "kind": "injections",
        "schedule": schedule,
        "n_injections": len(schedule),
        "directives": directives,
    }


def _parse_run_csv(raw_text: str, kind: str) -> dict:
    """power / baseline / blank 共用；blank 允许附空白自身注射时序列。"""
    directives, rows = _directives_and_rows(raw_text)
    header = json.loads(directives.pop("__header__"))
    rows = json.loads(directives.pop("__rows__"))
    low = [h.lower() for h in header]
    if len(rows) > MAX_SAMPLES:
        raise BudgetExhausted(
            f"样本数 {len(rows)} 超过预算 {MAX_SAMPLES}",
            [Diagnostic("budget_samples", "样本数超过预算", "rows")],
        )
    required = {"time", "power"}
    missing_cols = required - set(low)
    if missing_cols:
        raise FormatCorrupted(
            f"缺少列 {sorted(missing_cols)}",
            [Diagnostic("missing_column", f"缺少列 {sorted(missing_cols)}", "header")],
        )
    col = {h: i for i, h in enumerate(low)}
    t_unit = directives.get("time_unit", "s")
    p_unit = directives.get("power_unit", "")
    if not p_unit:
        raise FormatCorrupted(
            "缺少 # power_unit 单位声明",
            [Diagnostic("missing_unit", "功率必须声明单位", "header")],
        )
    t_raw, t_val, p_raw, p_val = [], [], [], []
    diags: list[Diagnostic] = []
    prev_t = None
    for idx, cells in enumerate(rows):
        pos = f"row {idx + 2}"
        if len(cells) < 2:
            raise FormatCorrupted(f"{pos}: 列数不足",
                                  [Diagnostic("ragged_row", "列数不足", pos)])
        t_tok, p_tok = cells[col["time"]], cells[col["power"]]
        try:
            t_s, _ = canonical("time", t_tok, t_unit, pos)
        except FormatCorrupted as exc:
            raise FormatCorrupted(f"{pos}: 时间坐标无效", exc.diagnostics)
        if prev_t is not None and t_s <= prev_t:
            raise FormatCorrupted(f"{pos}: 时间坐标必须严格递增",
                                  [Diagnostic("time_not_monotonic",
                                              "时间坐标必须严格递增", pos)])
        prev_t = t_s
        t_raw.append(t_tok)
        t_val.append(t_s)
        if is_missing_token(p_tok):
            p_raw.append(p_tok)
            p_val.append(None)
            diags.append(Diagnostic("sample_gap", "采样点缺失（数据缺口）",
                                    pos, severity="info", value=p_tok))
        else:
            try:
                p_canon, _ = canonical("power", p_tok, p_unit, pos)
                p_raw.append(p_tok)
                p_val.append(p_canon)
            except FormatCorrupted as exc:
                p_raw.append(p_tok)
                p_val.append(None)
                for d in exc.diagnostics:
                    d.pos = pos
                    diags.append(d)
    out = {
        "kind": kind,
        "directives": directives,
        "time": Series(t_raw, t_val, t_unit, "s").to_dict(),
        "power": Series(p_raw, p_val, p_unit, "uW", diags).to_dict(),
        "n_samples": len(rows),
    }
    # blank 可携带自身注射时序（空白滴定也是一次滴定）
    if {"inj_index", "inj_time", "inj_volume"} <= set(low):
        bcol = col
        brows = []
        for i, cells in enumerate(rows):
            if len(cells) > bcol["inj_index"] and not is_missing_token(cells[bcol["inj_index"]]):
                brows.append([
                    cells[bcol["inj_index"]],
                    cells[bcol["inj_time"]] if len(cells) > bcol["inj_time"] else None,
                    cells[bcol["inj_volume"]] if len(cells) > bcol["inj_volume"] else None,
                ])
        b_header = ["index", "time", "volume"]
        b_directives = {
            "time_unit": directives.get("inj_time_unit", t_unit),
            "volume_unit": directives.get("inj_volume_unit", "uL"),
        }
        out["blank_schedule"] = _injection_schedule(
            brows, b_header, {"index": 0, "time": 1, "volume": 2},
            b_directives, pos_prefix="blank inj")
    return out


def parse_power(raw_text: str) -> dict:
    return _parse_run_csv(raw_text, "power")


def parse_baseline(raw_text: str) -> dict:
    out = _parse_run_csv(raw_text, "baseline")
    if any(v is None for v in out["power"]["value"]):
        raise ITCError(
            "基线曲线不允许含缺口",
            [Diagnostic("baseline_gap", "基线存在数据缺口，无法建立对应", "power")],
        )
    return out


def parse_blank(raw_text: str) -> dict:
    out = _parse_run_csv(raw_text, "blank")
    if "blank_schedule" not in out:
        diags = [Diagnostic("blank_schedule_missing",
                            "空白材料缺少空白注射时序（inj_index/inj_time/inj_volume）",
                            "header", severity="warning")]
        out["power"]["diagnostics"] = (
            out["power"].get("diagnostics", []) + [d.to_dict() for d in diags])
    return out


def parse_concentration(raw_text: str) -> dict:
    """浓度材料（JSON）。

    形如：
      {"cell_macromolecule": {"value": "0.20", "unit": "mM"},
       "syringe_ligand":     {"value": "4.0",  "unit": "mM"},
       "per_injection": {"3": {"value": "3.95", "unit": "mM"}},
       "note": "…"}
    无法转换的字段留在原位（raw 保留 + 诊断），其余字段仍可用。
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FormatCorrupted(
            f"浓度 JSON 损坏: {exc.msg}",
            [Diagnostic("json_parse", f"JSON 损坏: {exc.msg}",
                        f"char {exc.pos}")],
        )
    if not isinstance(data, dict):
        raise FormatCorrupted("浓度材料必须是 JSON 对象",
                              [Diagnostic("json_shape", "顶层必须是对象", "$")])
    diagnostics: list[Diagnostic] = []

    def _conc(path: str, node, required: bool):
        if node is None:
            if required:
                raise ITCError(
                    f"缺少必填浓度字段 {path}",
                    [Diagnostic("missing_field", f"缺少必填字段 {path}", path)],
                )
            return None
        if not isinstance(node, dict) or "value" not in node or "unit" not in node:
            diagnostics.append(Diagnostic(
                "bad_concentration", "浓度项必须含 value 与 unit", path,
                value=json.dumps(node, ensure_ascii=False)))
            return {"raw": node, "value_mM": None, "unit": None,
                    "ok": False}
        raw_v, unit = str(node["value"]), str(node["unit"])
        if is_missing_token(raw_v):
            return {"raw_value": raw_v, "value_mM": None, "unit": unit,
                    "ok": False, "missing": True}
        try:
            v, _ = canonical("concentration", raw_v, unit, path)
            return {"raw_value": raw_v, "value_mM": v, "unit": unit, "ok": True}
        except FormatCorrupted as exc:
            for d in exc.diagnostics:
                d.pos = path
                diagnostics.append(d)
            return {"raw_value": raw_v, "value_mM": None, "unit": unit, "ok": False}

    cell = _conc("cell_macromolecule", data.get("cell_macromolecule"), True)
    syr = _conc("syringe_ligand", data.get("syringe_ligand"), True)
    per_inj = {}
    raw_per = data.get("per_injection", {})
    if raw_per and not isinstance(raw_per, dict):
        raise FormatCorrupted("per_injection 必须是对象",
                              [Diagnostic("json_shape", "per_injection 必须是对象",
                                          "per_injection")])
    for k, node in (raw_per or {}).items():
        per_inj[str(k)] = _conc(f"per_injection.{k}", node, False)
    if not cell or not cell.get("ok"):
        raise ITCError(
            "池内大分子浓度不可用，绑定模型无法定义",
            diagnostics + [Diagnostic("cell_concentration_unusable",
                                      "池内浓度无法使用", "cell_macromolecule")],
        )
    return {
        "kind": "concentration",
        "cell_macromolecule": cell,
        "syringe_ligand": syr,
        "per_injection": per_inj,
        "note": data.get("note", ""),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def parse_conditions(raw_text: str) -> dict:
    """实验条件（JSON）：温度、池体积；全部携带单位，原值保留。"""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FormatCorrupted(
            f"条件 JSON 损坏: {exc.msg}",
            [Diagnostic("json_parse", f"JSON 损坏: {exc.msg}", f"char {exc.pos}")],
        )
    if not isinstance(data, dict):
        raise FormatCorrupted("条件材料必须是 JSON 对象",
                              [Diagnostic("json_shape", "顶层必须是对象", "$")])
    diagnostics: list[Diagnostic] = []
    if "temperature" not in data or "cell_volume" not in data:
        raise ITCError(
            "条件必须包含 temperature 与 cell_volume",
            [Diagnostic("missing_field", "缺少 temperature/cell_volume", "$")],
        )

    tnode, vnode = data["temperature"], data["cell_volume"]
    for name, node in (("temperature", tnode), ("cell_volume", vnode)):
        if not isinstance(node, dict) or "value" not in node or "unit" not in node:
            raise FormatCorrupted(
                f"{name} 必须含 value 与 unit",
                [Diagnostic("bad_field", f"{name} 必须含 value 与 unit", name)],
            )
    raw_t, tunit = str(tnode["value"]), str(tnode["unit"])
    raw_v, vunit = str(vnode["value"]), str(vnode["unit"])
    try:
        temp_c, _ = convert_temperature(raw_t, tunit, "temperature")
    except FormatCorrupted as exc:
        diagnostics.extend(exc.diagnostics)
        temp_c = None
    try:
        vol_ul, _ = canonical("volume", raw_v, vunit, "cell_volume")
    except FormatCorrupted as exc:
        diagnostics.extend(exc.diagnostics)
        vol_ul = None
    if temp_c is None or vol_ul is None:
        raise ITCError("温度或池体积无法使用", diagnostics)
    if vol_ul <= 0:
        raise ITCError("池体积必须为正",
                       [Diagnostic("non_positive_volume", "池体积必须为正",
                                   "cell_volume", value=raw_v)])
    return {
        "kind": "conditions",
        "temperature": {"raw_value": raw_t, "value_degC": temp_c, "unit": tunit},
        "cell_volume": {"raw_value": raw_v, "value_uL": vol_ul, "unit": vunit},
        "reference_power": data.get("reference_power"),
        "note": data.get("note", ""),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


PARSERS = {
    "power": parse_power,
    "injections": parse_injections,
    "concentration": parse_concentration,
    "conditions": parse_conditions,
    "baseline": parse_baseline,
    "blank": parse_blank,
}


def import_material(kind: str, raw_text: str) -> dict:
    """统一入口：解析 + 内容身份。返回 {id, kind, raw_text, parsed}。"""
    if kind not in PARSERS:
        raise ITCError(f"未知材料类型 {kind!r}",
                       [Diagnostic("unknown_kind", f"未知材料类型 {kind!r}", kind)])
    parsed = PARSERS[kind](raw_text)
    mid = material_id(kind, raw_text)
    return {"id": mid, "kind": kind, "raw_text": raw_text, "parsed": parsed}
