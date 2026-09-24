"""原始材料导入：功率采样、注射时序、浓度、温度、池体积、基线、空白。

约定（需求“导入时保留原始表示，派生层才做规范化”）：
- ``raw_repr`` 字段保留文件中的原始字符串/数值；
- 可解析的量带 :class:`~itc.units.Quantity`；无法转换的值留在原位（``unconverted``）
  并附诊断，绝不为画图而丢掉；
- 数据缺口（空采样单元）是合法状态：保留为 gap，积分层决定积分窗是否跨越；
- 等价输入产生相同的内容寻址 id（见 :func:`material_id`）。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from enum import Enum

from .results import Diagnostic, Outcome
from .units import Quantity, parse_quantity

MAX_POWER_SAMPLES = 200_000
MAX_INJECTIONS = 10_000


class MaterialKind(str, Enum):
    POWER = "power"
    INJECTIONS = "injections"
    CONCENTRATION = "concentration"
    TEMPERATURE = "temperature"
    CELL_VOLUME = "cell_volume"
    BASELINE = "baseline"
    BLANK = "blank"

# 基线与空白曲线分开版本化：它们是独立种类
CURVE_KINDS = (MaterialKind.POWER, MaterialKind.BASELINE, MaterialKind.BLANK)
SCALAR_KINDS = (MaterialKind.TEMPERATURE, MaterialKind.CELL_VOLUME,
                MaterialKind.CONCENTRATION)
ALL_KINDS = tuple(MaterialKind)


@dataclass(frozen=True)
class Sample:
    """一个采样点；value=None 表示数据缺口，raw 保留原始单元文本。"""

    index: int
    t: Quantity | None = None
    value: Quantity | None = None
    raw_t: str = ""
    raw_v: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "t": self.t.canonical_value if self.t else None,
            "t_display": self.t.render() if self.t else self.raw_t,
            "value": self.value.canonical_value if self.value else None,
            "value_display": self.value.render() if self.value else self.raw_v,
            "gap": self.value is None,
        }


@dataclass(frozen=True)
class InjectionSpec:
    index: int                       # 1 基序号（滴定序）
    time: Quantity | None
    volume: Quantity | None
    raw_time: str = ""
    raw_volume: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "time": self.time.canonical_value if self.time else None,
            "time_display": self.time.render() if self.time else self.raw_time,
            "volume": self.volume.canonical_value if self.volume else None,
            "volume_display": self.volume.render() if self.volume else self.raw_volume,
        }


@dataclass(frozen=True)
class MaterialRecord:
    kind: MaterialKind
    material_id: str
    raw_repr: dict                  # 原始表示（逐单元字符串），永不规范化
    series: tuple[Sample, ...] = ()
    injections: tuple[InjectionSpec, ...] = ()
    scalar: dict = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()

    def summary(self) -> dict:
        if self.kind in CURVE_KINDS:
            gaps = sum(1 for s in self.series if s.value is None)
            return {"points": len(self.series), "gaps": gaps}
        if self.kind is MaterialKind.INJECTIONS:
            return {"injections": len(self.injections)}
        return dict(self.scalar)

    def as_dict(self) -> dict:
        out = {
            "kind": self.kind.value,
            "material_id": self.material_id,
            "summary": self.summary(),
            "raw_repr": self.raw_repr,
            "diagnostics": [vars(d) for d in self.diagnostics],
        }
        if self.kind in CURVE_KINDS:
            out["series"] = [s.as_dict() for s in self.series]
        elif self.kind is MaterialKind.INJECTIONS:
            out["injections"] = [i.as_dict() for i in self.injections]
        else:
            out["scalar"] = self.scalar
        return out


def material_id(kind: MaterialKind, raw_repr: dict) -> str:
    """内容寻址：相同原始表示（键排序）产生相同 id。"""
    body = json.dumps({"kind": kind.value, "raw_repr": raw_repr},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"mat_{kind.value}_{hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def import_material(kind: str, payload: object, *, source_name: str = "") -> Outcome[MaterialRecord]:
    """解析任意材料。返回 :class:`Outcome`，失败时给出带位置的诊断。"""
    try:
        k = MaterialKind(kind)
    except ValueError:
        return Outcome.format_error("UNKNOWN_KIND", f"未知材料种类 {kind!r}", source_name)
    src = source_name or f"{k.value}.input"
    if k in CURVE_KINDS:
        out = _parse_curve(k, payload, src)
    elif k is MaterialKind.INJECTIONS:
        out = _parse_injections(payload, src)
    else:
        out = _parse_scalar(k, payload, src)
    return out


def _decode(payload: object) -> tuple[list[list[str]] | None, dict | None, list[Diagnostic]]:
    """把 CSV 文本 / dict / list[dict] 统一成表格或字典；损坏返回 None。"""
    diags: list[Diagnostic] = []
    if isinstance(payload, str):
        try:
            text = payload
            reader = csv.reader(io.StringIO(text))
            rows = [list(r) for r in reader]
        except csv.Error as exc:
            diags.append(Diagnostic("CSV_PARSE", f"CSV 无法解析: {exc}", payload[:40] if payload else ""))
            return None, None, diags
        while rows and all(c.strip() == "" for c in rows[-1]):
            rows.pop()
        return rows, None, diags
    if isinstance(payload, dict):
        return None, payload, diags
    if isinstance(payload, list):
        # list[dict] -> 带表头的行表
        if payload and isinstance(payload[0], dict):
            keys: list[str] = []
            for row in payload:
                for key in row:
                    if key not in keys:
                        keys.append(str(key))
            rows = [keys]
            for row in payload:
                rows.append(["" if row.get(kk) is None else str(row.get(kk, "")) for kk in keys])
            return rows, None, diags
        diags.append(Diagnostic("BAD_SHAPE", "列表材料必须是对象数组", ""))
        return None, None, diags
    diags.append(Diagnostic("BAD_TYPE", f"不支持的材料载荷类型 {type(payload).__name__}", ""))
    return None, None, diags


def _norm_header(name: str) -> str:
    return name.strip().lower().lstrip("#").strip()


def _pick(headers: list[str], *candidates: str) -> int:
    norm = [_norm_header(h) for h in headers]
    for cand in candidates:
        if cand in norm:
            return norm.index(cand)
    return -1


# ---------------------------------------------------------------------------
# 曲线（power / baseline / blank）
# ---------------------------------------------------------------------------

def _parse_curve(kind: MaterialKind, payload: object, src: str) -> Outcome[MaterialRecord]:
    rows, obj, diags = _decode(payload)
    raws: list[dict] = []
    samples: list[Sample] = []
    t_family = "time"
    v_family = "power"
    if rows is not None:
        if len(rows) < 2:
            return Outcome.format_error("EMPTY_CURVE", f"{kind.value} 曲线为空或仅有表头", src)
        headers = rows[0]
        ci_t = _pick(headers, "time_s", "time", "t", "seconds")
        ci_v = _pick(headers, "power_uw", "power", "p", "heat_rate", "dq_dt", "baseline", "blank")
        if ci_t < 0 or ci_v < 0:
            return Outcome.format_error(
                "MISSING_COLUMN",
                f"{kind.value} 需要时间与功率两列，实际表头={headers}", src)
        if len(rows) - 1 > MAX_POWER_SAMPLES:
            return Outcome.resource("TOO_MANY_SAMPLES",
                                    f"功率采样数超过预算 {MAX_POWER_SAMPLES}", src)
        for r_idx, row in enumerate(rows[1:], start=2):
            raw_t = row[ci_t].strip() if ci_t < len(row) else ""
            raw_v = row[ci_v].strip() if ci_v < len(row) else ""
            raws.append({"row": r_idx, "t": raw_t, "value": raw_v})
            tq = vq = None
            if not raw_t:
                diags.append(Diagnostic("MISSING_TIME", "采样时间缺失，该点不可用",
                                        f"{src}:row {r_idx}", raw_v))
                continue
            try:
                tq = parse_quantity(raw_t, t_family, default_unit="s")
            except ValueError as exc:
                diags.append(Diagnostic("BAD_TIME", f"时间无法解析: {exc}",
                                        f"{src}:row {r_idx} / field t", raw_t))
                continue
            if raw_v == "":
                diags.append(Diagnostic("DATA_GAP", "功率值为空：记为数据缺口",
                                        f"{src}:row {r_idx} / field value", severity="warning"))
            else:
                try:
                    vq = parse_quantity(raw_v, v_family, default_unit="µW")
                except ValueError as exc:
                    diags.append(Diagnostic("BAD_VALUE", f"功率无法转换: {exc}",
                                            f"{src}:row {r_idx} / field value", raw_v))
            samples.append(Sample(len(samples), tq, vq, raw_t, raw_v))
    else:
        return Outcome.format_error("BAD_CURVE", f"{kind.value} 曲线需要 CSV 或对象数组", src)
    valid = [s for s in samples if s.t is not None]
    if len(valid) < 2:
        return Outcome.fail("unresolvable",
                            [Diagnostic("TOO_FEW_POINTS",
                                        f"{kind.value} 可解析采样点不足 2 个，无法建立时间基准", src)] + diags)
    raw_repr = {"format": "csv", "source": src, "rows": raws}
    rec = MaterialRecord(kind, material_id(kind, raw_repr), raw_repr,
                         tuple(samples), (), (), tuple(diags))
    return Outcome.success(rec, [d for d in diags if d.severity == "warning"])


# ---------------------------------------------------------------------------
# 注射时序
# ---------------------------------------------------------------------------

def _parse_injections(payload: object, src: str) -> Outcome[MaterialRecord]:
    rows, obj, diags = _decode(payload)
    raws: list[dict] = []
    specs: list[InjectionSpec] = []
    if rows is not None:
        if len(rows) < 2:
            return Outcome.format_error("EMPTY_INJECTIONS", "注射时序为空或仅有表头", src)
        headers = rows[0]
        ci_t = _pick(headers, "time_s", "time", "t", "inject_time")
        ci_v = _pick(headers, "volume_ul", "volume", "v", "inj_volume", "dv")
        if ci_t < 0 or ci_v < 0:
            return Outcome.format_error("MISSING_COLUMN",
                                        f"注射时序需要时间与体积两列，实际表头={headers}", src)
        if len(rows) - 1 > MAX_INJECTIONS:
            return Outcome.resource("TOO_MANY_INJECTIONS",
                                    f"注射次数超过预算 {MAX_INJECTIONS}", src)
        for r_idx, row in enumerate(rows[1:], start=2):
            raw_t = row[ci_t].strip() if ci_t < len(row) else ""
            raw_v = row[ci_v].strip() if ci_v < len(row) else ""
            raws.append({"row": r_idx, "time": raw_t, "volume": raw_v})
            tq = vq = None
            if not raw_t:
                diags.append(Diagnostic("MISSING_INJ_TIME", "注射时间缺失",
                                        f"{src}:row {r_idx} / field time", raw_v))
            else:
                try:
                    tq = parse_quantity(raw_t, "time", default_unit="s")
                except ValueError as exc:
                    diags.append(Diagnostic("BAD_INJ_TIME", f"注射时间无法解析: {exc}",
                                            f"{src}:row {r_idx} / field time", raw_t))
            if not raw_v:
                diags.append(Diagnostic("MISSING_INJ_VOL", "注射体积缺失",
                                        f"{src}:row {r_idx} / field volume", raw_t))
            else:
                try:
                    vq = parse_quantity(raw_v, "volume", default_unit="µL")
                except ValueError as exc:
                    diags.append(Diagnostic("BAD_INJ_VOL", f"注射体积无法转换: {exc}",
                                            f"{src}:row {r_idx} / field volume", raw_v))
            specs.append(InjectionSpec(len(specs) + 1, tq, vq, raw_t, raw_v))
    else:
        # dict 形式：{"injections": [{"time":"..","volume":".."}]}
        items = obj.get("injections") if obj else None
        if not isinstance(items, list):
            return Outcome.format_error("BAD_INJECTIONS", "注射时序需要 CSV 或 injections 数组", src)
        if len(items) > MAX_INJECTIONS:
            return Outcome.resource("TOO_MANY_INJECTIONS",
                                    f"注射次数超过预算 {MAX_INJECTIONS}", src)
        for i, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                diags.append(Diagnostic("BAD_INJECTION", "注射条目必须是对象",
                                        f"{src}:injections[{i - 1}]"))
                raws.append({"index": i, "time": str(item), "volume": ""})
                specs.append(InjectionSpec(i, None, None, str(item), ""))
                continue
            raw_t = "" if item.get("time") is None else str(item.get("time"))
            raw_v = "" if item.get("volume") is None else str(item.get("volume"))
            raws.append({"index": i, "time": raw_t, "volume": raw_v})
            tq = vq = None
            try:
                tq = parse_quantity(raw_t, "time", default_unit="s")
            except ValueError as exc:
                diags.append(Diagnostic("BAD_INJ_TIME", f"注射时间无法解析: {exc}",
                                        f"{src}:injections[{i - 1}].time", raw_t))
            try:
                vq = parse_quantity(raw_v, "volume", default_unit="µL")
            except ValueError as exc:
                diags.append(Diagnostic("BAD_INJ_VOL", f"注射体积无法转换: {exc}",
                                        f"{src}:injections[{i - 1}].volume", raw_v))
            specs.append(InjectionSpec(i, tq, vq, raw_t, raw_v))
    if errors := [d for d in diags if d.severity == "error"]:
        return Outcome.fail("format_error", errors)
    if not specs:
        return Outcome.unresolvable("NO_INJECTIONS", "没有任何注射条目", src)
    raw_repr = {"format": "csv" if rows is not None else "json", "source": src, "entries": raws}
    rec = MaterialRecord(MaterialKind.INJECTIONS, material_id(MaterialKind.INJECTIONS, raw_repr),
                         raw_repr, (), tuple(specs), (), ())
    return Outcome.success(rec)


# ---------------------------------------------------------------------------
# 标量（浓度 / 温度 / 池体积）
# ---------------------------------------------------------------------------

_SCALAR_SPEC = {
    MaterialKind.CONCENTRATION: (
        "concentration", "concentration", "syringe_concentration", "ligand_concentration", "c"),
    MaterialKind.TEMPERATURE: ("temperature", "temperature_c", "temp", "t"),
    MaterialKind.CELL_VOLUME: ("cell_volume", "cellvolume", "v0", "volume_cell", "cell"),
}
_SCALAR_FAMILY = {
    MaterialKind.CONCENTRATION: "concentration",
    MaterialKind.TEMPERATURE: "temperature",
    MaterialKind.CELL_VOLUME: "volume",
}
_SCALAR_DEFAULT_UNIT = {
    MaterialKind.CONCENTRATION: "mM",
    MaterialKind.TEMPERATURE: "°C",
    MaterialKind.CELL_VOLUME: "µL",
}


def _parse_keyvalue_text(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        out[k.strip()] = v.strip()
    return out


def _parse_scalar(kind: MaterialKind, payload: object, src: str) -> Outcome[MaterialRecord]:
    if isinstance(payload, str) and not payload.lstrip().startswith("{"):
        data = _parse_keyvalue_text(payload)
    elif isinstance(payload, dict):
        data = payload
    else:
        return Outcome.format_error("BAD_SCALAR", f"{kind.value} 需要键值对象", src)
    keys = _SCALAR_SPEC[kind]
    chosen = next((k for k in keys if k in data), None)
    if chosen is None:
        return Outcome.format_error(
            "MISSING_FIELD",
            f"{kind.value} 缺少字段，接受的键名: {keys}", src)
    raw_val = data[chosen]
    raw_text = "" if raw_val is None else str(raw_val)
    if raw_text.strip() == "":
        return Outcome.unresolvable("EMPTY_SCALAR",
                                    f"{kind.value} 值为空，无法规范化",
                                    f"{src}:{chosen}")
    raw_repr = {"format": "json", "source": src, "field": chosen, "value": raw_text}
    try:
        q = parse_quantity(raw_text, _SCALAR_FAMILY[kind],
                           default_unit=_SCALAR_DEFAULT_UNIT[kind])
    except ValueError as exc:
        # 无法转换的值留在原位置并附诊断
        rec = MaterialRecord(kind, material_id(kind, raw_repr), raw_repr, (), (),
                             {"field": chosen, "raw": raw_text, "converted": False},
                             (Diagnostic("UNCONVERTIBLE_VALUE",
                                         f"{kind.value} 无法转换，保留原始表示: {exc}",
                                         f"{src}:{chosen}", raw_text),))
        return Outcome.success(rec)
    scalar = {
        "field": chosen,
        "raw": raw_text,
        "converted": True,
        "value": q.canonical_value,
        "canonical_unit": q.canonical_unit,
        "display": q.render(),
    }
    rec = MaterialRecord(kind, material_id(kind, raw_repr), raw_repr, (), (), scalar)
    return Outcome.success(rec)
