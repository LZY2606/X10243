"""不可变材料层。

六类材料分开版本化：
power_series / injection_schedule / concentrations / conditions / baseline / blank_run。
每类材料保留原始表示(raw)，规范化结果放入 canonical；无法转换的值在原位置
保留为缺失并附 token 诊断。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import Diagnostic
from .identity import material_id
from .units import Token, parse_quantity, require_number

POWER = "power_series"
SCHEDULE = "injection_schedule"
CONCENTRATIONS = "concentrations"
CONDITIONS = "conditions"
BASELINE = "baseline"
BLANK = "blank_run"

MATERIAL_KINDS = (POWER, SCHEDULE, CONCENTRATIONS, CONDITIONS, BASELINE, BLANK)


@dataclass(frozen=True)
class Material:
    kind: str
    source_ref: str
    revision: int
    name: str
    raw: Any
    canonical: Any
    tokens: list[dict[str, Any]] = field(default_factory=list)
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source_ref": self.source_ref,
            "revision": self.revision,
            "name": self.name,
            "raw": self.raw,
            "canonical": self.canonical,
            "tokens": self.tokens,
        }


def make_material(
    kind: str,
    source_ref: str,
    revision: int,
    name: str,
    raw: Any,
    canonical: Any,
    tokens: list[Token] | list[dict[str, Any]] | None = None,
) -> Material:
    token_dicts = [t.to_dict() if isinstance(t, Token) else t for t in (tokens or [])]
    mid = material_id(kind, source_ref, revision, canonical)
    return Material(kind, source_ref, revision, name, raw, canonical, token_dicts, mid)


# ---------------------------------------------------------------- 解析

def _parse_power_block(
    block: dict[str, Any],
    loc_prefix: str,
) -> tuple[dict[str, Any], list[Token]]:
    tokens: list[Token] = []
    t_unit = block.get("time_unit", "s")
    p_unit = block.get("power_unit", "ucal/s")
    rows = block.get("samples")
    if not isinstance(rows, list):
        raise TypeError("samples_not_list")
    samples: list[list[Any]] = []
    missing: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        raw_t, raw_p = row[0], row[1]
        tval = parse_quantity(raw_t, t_unit, "s", f"{loc_prefix}.samples[{i}].t")
        pval = parse_quantity(raw_p, p_unit, "cal/s", f"{loc_prefix}.samples[{i}].p")
        if isinstance(tval, Token):
            tokens.append(tval)
            raise ValueError(f"time_axis_unparseable_at_{i}")
        p_norm: float | None
        if isinstance(pval, Token):
            tokens.append(pval)
            p_norm = None
            missing.append({"index": i, "raw": pval.raw, "reason": pval.reason})
        else:
            p_norm = pval
        samples.append([tval, p_norm])
    return (
        {
            "time_unit": "s",
            "power_unit": "cal/s",
            "samples": samples,
            "missing": missing,
        },
        tokens,
    )


def _parse_schedule_block(
    block: dict[str, Any],
    loc_prefix: str,
) -> tuple[dict[str, Any], list[Token]]:
    tokens: list[Token] = []
    t_unit = block.get("time_unit", "s")
    v_unit = block.get("volume_unit", "uL")
    injections: list[dict[str, Any]] = []
    for i, inj in enumerate(block.get("injections", [])):
        tval = parse_quantity(inj["time"], t_unit, "s", f"{loc_prefix}.injections[{i}].time")
        vval = parse_quantity(inj["volume"], v_unit, "L", f"{loc_prefix}.injections[{i}].volume")
        if isinstance(tval, Token):
            tokens.append(tval)
            raise ValueError(f"injection_time_unparseable_at_{i}")
        if isinstance(vval, Token):
            tokens.append(vval)
        injections.append(
            {
                "index": inj.get("index", i + 1),
                "t": tval,
                "volume": None if isinstance(vval, Token) else vval,
                "raw_time": inj["time"],
                "raw_volume": inj["volume"],
            }
        )
    return (
        {"time_unit": "s", "volume_unit": "L", "injections": injections},
        tokens,
    )


def parse_bundle(bundle: dict[str, Any], source_ref: str = "alpha", revision: int = 1) -> dict[str, Material]:
    """把原始 ITC 包解析为 6 类独立版本化材料。

    保留原始表示；语义问题以 Token 记录，结构损坏抛 TypeError/KeyError，
    由调用层包装为 FormatCorrupt。
    """
    mats: dict[str, Material] = {}

    power_raw = bundle["power_series"]
    power_canon, power_tokens = _parse_power_block(power_raw, f"{source_ref}.power_series")
    mats[POWER] = make_material(POWER, source_ref, revision, "样品池功率曲线", power_raw, power_canon, power_tokens)

    sched_raw = bundle["injection_schedule"]
    sched_canon, sched_tokens = _parse_schedule_block(sched_raw, f"{source_ref}.injection_schedule")
    mats[SCHEDULE] = make_material(SCHEDULE, source_ref, revision, "注射时序与体积", sched_raw, sched_canon, sched_tokens)

    conc = bundle["concentrations"]
    c_unit = conc.get("conc_unit", "uM")
    diag: list[Diagnostic] = []
    cell = parse_quantity(conc["cell"], c_unit, "M", f"{source_ref}.concentrations.cell")
    syr = parse_quantity(conc["syringe"], c_unit, "M", f"{source_ref}.concentrations.syringe")
    conc_tokens = [v for v in (cell, syr) if isinstance(v, Token)]
    mats[CONCENTRATIONS] = make_material(
        CONCENTRATIONS,
        source_ref,
        revision,
        "浓度",
        conc,
        {
            "conc_unit": "M",
            "cell": None if isinstance(cell, Token) else cell,
            "syringe": None if isinstance(syr, Token) else syr,
            "raw_cell": conc["cell"],
            "raw_syringe": conc["syringe"],
        },
        conc_tokens,
    )

    cond = bundle["conditions"]
    tval = parse_quantity(cond["temperature"], cond.get("temperature_unit", "K"), "K",
                          f"{source_ref}.conditions.temperature")
    vval = parse_quantity(cond["cell_volume"], cond.get("cell_volume_unit", "uL"), "L",
                          f"{source_ref}.conditions.cell_volume")
    cond_tokens = [v for v in (tval, vval) if isinstance(v, Token)]
    mats[CONDITIONS] = make_material(
        CONDITIONS,
        source_ref,
        revision,
        "温度与池体积",
        cond,
        {
            "temperature": None if isinstance(tval, Token) else tval,
            "cell_volume": None if isinstance(vval, Token) else vval,
            "temperature_unit": "K",
            "cell_volume_unit": "L",
            "raw_temperature": cond["temperature"],
            "raw_cell_volume": cond["cell_volume"],
        },
        cond_tokens,
    )

    base_raw = bundle["baseline"]
    base_canon, base_tokens = _parse_baseline(base_raw, f"{source_ref}.baseline")
    mats[BASELINE] = make_material(BASELINE, source_ref, revision, "基线", base_raw, base_canon, base_tokens)

    blank_raw = bundle.get("blank_run")
    if blank_raw is not None:
        bp_canon, bp_tokens = _parse_power_block(blank_raw["power_series"], f"{source_ref}.blank_run.power_series")
        bs_canon, bs_tokens = _parse_schedule_block(blank_raw["injection_schedule"],
                                                    f"{source_ref}.blank_run.injection_schedule")
        bb_canon, bb_tokens = _parse_baseline(
            blank_raw.get("baseline", {"time_unit": "s", "power_unit": "ucal/s",
                                       "anchors": [{"t": 0.0, "p": 0.0}, {"t": 1.0, "p": 0.0}]}),
            f"{source_ref}.blank_run.baseline")
        mats[BLANK] = make_material(
            BLANK,
            source_ref,
            revision,
            "空白对照曲线",
            blank_raw,
            {"power_series": bp_canon, "injection_schedule": bs_canon, "baseline": bb_canon},
            bp_tokens + bs_tokens + bb_tokens,
        )

    return mats


def _parse_baseline(block: dict[str, Any], loc_prefix: str) -> tuple[dict[str, Any], list[Token]]:
    tokens: list[Token] = []
    t_unit = block.get("time_unit", "s")
    p_unit = block.get("power_unit", "ucal/s")
    anchors = []
    for i, a in enumerate(block.get("anchors", [])):
        tval = parse_quantity(a["t"], t_unit, "s", f"{loc_prefix}.anchors[{i}].t")
        pval = parse_quantity(a["p"], p_unit, "cal/s", f"{loc_prefix}.anchors[{i}].p")
        if isinstance(tval, Token):
            tokens.append(tval)
            raise ValueError("baseline_time_unparseable")
        if isinstance(pval, Token):
            tokens.append(pval)
        anchors.append({"t": tval, "p": None if isinstance(pval, Token) else pval, "raw_p": a["p"]})
    return (
        {
            "time_unit": "s",
            "power_unit": "cal/s",
            "method": block.get("method", "piecewise_linear"),
            "anchors": anchors,
        },
        tokens,
    )


def material_health(materials: dict[str, Material]) -> list[Diagnostic]:
    """跨材料语义检查：必需数值缺失则带位置诊断，绝不伪装成功。"""
    diag: list[Diagnostic] = []
    conc = materials[CONCENTRATIONS].canonical
    if conc["cell"] is None:
        require_number(Token(conc["raw_cell"], None, "unconvertible", "concentrations.cell"), "池浓度", diag)
    if conc["syringe"] is None:
        require_number(Token(conc["raw_syringe"], None, "unconvertible", "concentrations.syringe"), "注射器浓度", diag)
    cond = materials[CONDITIONS].canonical
    if cond["temperature"] is None:
        require_number(Token(cond["raw_temperature"], None, "unconvertible", "conditions.temperature"), "温度", diag)
    if cond["cell_volume"] is None:
        require_number(Token(cond["raw_cell_volume"], None, "unconvertible", "conditions.cell_volume"), "池体积", diag)
    for inj in materials[SCHEDULE].canonical["injections"]:
        if inj["volume"] is None:
            require_number(Token(inj["raw_volume"], None, "unconvertible", "schedule.volume"), "注射体积", diag)
    for a in materials[BASELINE].canonical["anchors"]:
        if a["p"] is None:
            require_number(Token(a["raw_p"], None, "unconvertible", "baseline.anchor"), "基线锚点", diag)
    return diag
