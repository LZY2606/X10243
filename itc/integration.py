"""注射积分引擎（纯 NumPy，服务端/存储层可直接测试）。

硬规则（业务判断，不藏在浏览器）：
1. 注射后峰可重叠：窗口按“下一次注射时刻”截尾，并显式标 overlap。
2. 积分窗不跨越数据缺口：窗口碰到缺口即截断，被吞掉区间时标 window_truncated_gap；
   覆盖不足 min_coverage 时该注积分 invalid，绝不插值。
3. 首注舍弃：首注保留原始积分，fitting 状态 discarded_first。
4. 稀释热（空白）扣除必须说明对应规则：match_ordinal / match_none；
   空白缺对应时 raw 仍给出，blank_corrected=None 并附诊断。
5. 积分端点是可修订决定（window_mode 与手动端点），每次修订产生新事件、
   新版本结果；历史结果不被覆盖。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ResourceExhausted, SemanticImpossible
from .models import MaterialVersion, Point, gaps_from_points, parse_points
from .units import parse_measurement, require

# 参数版本：规则变化时显式升版，结果记录所用版本。
RULE_VERSION = "integration-1.0.0"
DEFAULT_LEAD_S = 2.0
DEFAULT_TAIL_S = 120.0
MIN_COVERAGE = 0.60


@dataclass
class InjectionWindow:
    index: int
    start_s: float
    end_s: float
    next_injection_s: float | None
    overlap: bool
    gap_truncated: bool
    coverage: float
    valid: bool
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class InjectionIntegral:
    index: int
    time_raw: str
    time_s: float
    volume_raw: str
    volume_l: float
    window: dict
    raw_heat_j: float | None
    blank_heat_j: float | None
    blank_rule: str
    blank_corrected_j: float | None
    status: str               # ok | discarded_first | invalid | ...
    diagnostics: list[str]


def _sample_series(power_mv: MaterialVersion) -> tuple[list[Point], str, str]:
    c = power_mv.content
    pts = parse_points(c["points"])
    tunit = c.get("t_unit", "s")
    punit_hint = c.get("p_unit")
    return pts, tunit, (punit_hint or "")


def _numeric_arrays(pts: list[Point], t_unit: str, where: str):
    """缺口样本留在序列中（p=None），不插值、不删除。"""
    t_meas = [parse_measurement(p.t_raw, "time", location=f"{where}.t[{i}]")
              for i, p in enumerate(pts)]
    t = np.array([require(m, "采样时间", f"{where}.t[{i}]")
                  for i, m in enumerate(t_meas)], dtype=float)
    pvals = np.full(len(pts), np.nan)
    punit = None
    for i, p in enumerate(pts):
        if p.missing or p.p_raw is None:
            continue
        m = parse_measurement(p.p_raw, "power", location=f"{where}.p[{i}]")
        if m.canonical is None:
            raise SemanticImpossible(m.diagnostic or "功率单位不可换算",
                                     locations=[f"{where}.p[{i}]"])
        punit = m.canonical_unit
        pvals[i] = m.canonical
    return t, pvals, punit


def _baseline_value(ts, pv, baseline_mv: MaterialVersion | None,
                    window: tuple[float, float], where: str) -> float:
    """基线：显式 pre/post 区间的均值（缺口与 NaN 不参与）。"""
    if baseline_mv and baseline_mv.content.get("method") == "constant":
        m = parse_measurement(baseline_mv.content["value"], "power",
                              location=f"{where}.baseline")
        return require(m, "基线功率", f"{where}.baseline")
    lo, hi = window
    sel = (ts >= lo) & (ts <= hi) & np.isfinite(pv)
    if sel.sum() < 3:
        raise SemanticImpossible(
            "基线区间内有效功率点不足（至少 3 点）",
            locations=[f"{where}.baseline_window"])
    return float(np.mean(pv[sel]))


def build_windows(inj_times, inj_volumes, *, lead_s=DEFAULT_LEAD_S,
                  tail_s=DEFAULT_TAIL_S, window_mode="next_injection",
                  manual_end=None):
    """构造每注窗口。

    window_mode:
      - next_injection: end = min(t_i+tail, t_{i+1})，重叠时 overlap=True
      - fixed_tail:     end = t_i+tail（允许窗口区间重叠，仅报告 overlap）
      - manual:         end = manual_end[index]
    """
    windows: list[InjectionWindow] = []
    n = len(inj_times)
    for i, ti in enumerate(inj_times):
        nxt = inj_times[i + 1] if i + 1 < n else None
        start = ti - lead_s
        if window_mode == "fixed_tail":
            end = ti + tail_s
        elif window_mode == "manual":
            end = float(manual_end[i])
        else:
            end = ti + tail_s if nxt is None else min(ti + tail_s, nxt)
        overlap = bool(nxt is not None and ti + tail_s > nxt)
        windows.append(InjectionWindow(
            index=i, start_s=start, end_s=end, next_injection_s=nxt,
            overlap=overlap, gap_truncated=False, coverage=1.0, valid=True))
    return windows


def _integrate_one(ts, pv, w: InjectionWindow, gaps, baseline):
    """对单注做梯形积分；缺口从积分域中切除并检查覆盖率。

    任何一边落在缺口的样本区间都不参与积分（窗不跨越缺口、不插值）。
    """
    sel = (ts >= w.start_s) & (ts < w.end_s)
    idxs = np.where(sel)[0]
    wanted_span = w.end_s - w.start_s
    if idxs.size < 2:
        w.valid = False
        w.coverage = 0.0
        w.diagnostics.append("积分窗内样本点不足 2 个")
        return None
    missing = ~np.isfinite(pv)
    total = 0.0
    covered = 0.0
    for a, b in zip(idxs[:-1], idxs[1:]):
        if missing[a] or missing[b]:
            continue
        total += 0.5 * (pv[a] + pv[b] - 2 * baseline) * (ts[b] - ts[a])
        covered += ts[b] - ts[a]
    coverage = covered / wanted_span if wanted_span > 0 else 0.0
    w.coverage = float(coverage)
    gap_inside = any(not (ts[b - 1] < w.start_s or ts[a] > w.end_s)
                     for a, b in gaps)
    if gap_inside:
        w.gap_truncated = True
        w.diagnostics.append(
            f"积分窗与数据缺口相交（覆盖 {coverage:.0%}），缺口未插值")
    if coverage < MIN_COVERAGE:
        w.valid = False
        w.diagnostics.append(
            f"有效覆盖 {coverage:.0%} < {MIN_COVERAGE:.0%}，积分判定无效")
    return float(total) if w.valid else None




def _read_injections(inj_mv: MaterialVersion):
    rows = inj_mv.content["rows"]
    times, volumes, traws, vraws = [], [], [], []
    for i, row in enumerate(rows):
        tm = parse_measurement(row["time"], "time",
                               location=f"injections[{i}].time")
        vm = parse_measurement(row["volume"], "volume",
                               location=f"injections[{i}].volume")
        times.append(require(tm, "注射时间", f"injections[{i}].time"))
        volumes.append(require(vm, "注射体积", f"injections[{i}].volume"))
        traws.append(tm.raw)
        vraws.append(vm.raw)
    return times, volumes, traws, vraws


def _run_heats(power_mv: MaterialVersion, inj_mv: MaterialVersion,
               baseline_mv, *, window_mode, tail_s, lead_s, manual_end,
               where):
    pts, tunit, _ = _sample_series(power_mv)
    ts, pv, _ = _numeric_arrays(pts, tunit, where)
    gaps = gaps_from_points(pts, location=where)
    times, volumes, traws, vraws = _read_injections(inj_mv)
    t_ref = power_mv.content.get("time_reference")
    i_ref = inj_mv.content.get("time_reference")
    if t_ref != i_ref:
        raise SemanticImpossible(
            f"功率曲线时间基准 {t_ref!r} 与注射时序基准 {i_ref!r} 不一致",
            locations=[f"{where}.time_reference"])
    windows = build_windows(times, volumes, lead_s=lead_s, tail_s=tail_s,
                            window_mode=window_mode, manual_end=manual_end)
    # 基线：默认使用首注前 60s 区间
    base_window = (max(ts[0], times[0] - 60.0), max(ts[0], times[0] - lead_s))
    baseline = _baseline_value(ts, pv, baseline_mv, base_window, where)
    results = []
    for w in windows:
        heat = _integrate_one(ts, pv, w, gaps, baseline)
        results.append((w, heat))
    return results, times, volumes, traws, vraws


def integrate_run(materials: dict[tuple[str, str], MaterialVersion], *,
                  role: str = "sample",
                  blank_rule: str = "match_ordinal",
                  window_mode: str = "next_injection",
                  tail_s: float = DEFAULT_TAIL_S,
                  lead_s: float = DEFAULT_LEAD_S,
                  manual_end: dict[int, float] | None = None,
                  budgets: dict | None = None) -> dict:
    """对一次运行（sample 或 blank）积分并按规则扣空白。

    返回结构化状态：每注窗口、覆盖、缺口、空白对应与诊断。
    """
    budgets = budgets or {}
    max_points = budgets.get("max_power_points", 2_000_000)
    power = materials[("power_sample", role)]
    inj = materials[("injections", role)]
    baseline = materials.get(("baseline", role))
    pts = parse_points(power.content["points"])
    if len(pts) > max_points:
        raise ResourceExhausted(
            f"功率采样点 {len(pts)} 超过预算 {max_points}",
            locations=[f"power_sample:{role}.points"])
    manual = None
    if window_mode == "manual":
        n = len(inj.content["rows"])
        try:
            manual = [float(manual_end[str(i)]) if str(i) in (manual_end or {})
                      else float(manual_end[i]) for i in range(n)]
        except (KeyError, TypeError):
            raise SemanticImpossible(
                "manual 模式要求为每注提供端点时间",
                locations=["decision.manual_end"])
    res, times, volumes, traws, vraws = _run_heats(
        power, inj, baseline, window_mode=window_mode, tail_s=tail_s,
        lead_s=lead_s, manual_end=manual, where=f"run:{role}")

    # 空白运行
    blank_heats: dict[int, float | None] = {}
    blank_diag: dict[int, str] = {}
    if role == SAMPLE_KIND and ("power_sample", BLANK_KIND) in materials:
        bm = materials
        if blank_rule == "match_none":
            for i in range(len(res)):
                blank_diag[i] = "规则 match_none：显式不扣空白，保留总热"
        else:
            try:
                bres, _, _, _, _ = _run_heats(
                    bm[("power_sample", BLANK_KIND)],
                    bm[("injections", BLANK_KIND)],
                    bm.get(("baseline", BLANK_KIND)),
                    window_mode=window_mode, tail_s=tail_s, lead_s=lead_s,
                    manual_end=manual, where="run:blank")
                bheats = {i: h for i, (_, h) in enumerate(bres)}
            except KeyError:
                bheats = {}
                bres = []
            if blank_rule not in ("match_ordinal", "match_none"):
                raise SemanticImpossible(
                    f"未知空白对应规则 {blank_rule!r}",
                    locations=["decision.blank_rule"])
            for i in range(len(res)):
                if i in bheats and bheats[i] is not None:
                    blank_heats[i] = bheats[i]
                else:
                    blank_heats[i] = None
                    blank_diag[i] = ("空白对应缺失：第 %d 注在空白运行中"
                                     "没有可用积分（match_ordinal）" % (i + 1))

    injections: list[InjectionIntegral] = []
    for i, (w, heat) in enumerate(res):
        diags = list(w.diagnostics)
        status = "ok"
        if i == 0:
            status = "discarded_first"
            diags.append("首注舍弃：首注常含计量与预混误差，不参与拟合")
        if not w.valid:
            status = "invalid"
        bh = blank_heats.get(i)
        corrected = None
        if role == SAMPLE_KIND and ("power_sample", BLANK_KIND) in materials:
            if blank_rule == "match_none":
                corrected = heat
            elif bh is not None and heat is not None:
                corrected = heat - bh
            else:
                diags.append(blank_diag.get(i, "空白对应缺失"))
        else:
            corrected = heat
        injections.append(InjectionIntegral(
            index=i, time_raw=traws[i], time_s=times[i],
            volume_raw=vraws[i], volume_l=volumes[i],
            window={
                "start_s": w.start_s, "end_s": w.end_s,
                "next_injection_s": w.next_injection_s,
                "overlap": w.overlap, "gap_truncated": w.gap_truncated,
                "coverage": w.coverage,
                "time_reference": power.content.get("time_reference"),
            },
            raw_heat_j=heat, blank_heat_j=bh, blank_rule=blank_rule,
            blank_corrected_j=corrected, status=status, diagnostics=diags))
    digest_src = {
        "power": power.version_id, "injections": inj.version_id,
        "baseline": baseline.version_id if baseline else None,
        "blank_power": materials.get(("power_sample", BLANK_KIND),
                                     MaterialVersion).__class__ and
                       materials.get(("power_sample", BLANK_KIND)) and
                       materials[("power_sample", BLANK_KIND)].version_id,
        "rule": RULE_VERSION, "mode": window_mode, "tail_s": tail_s,
        "lead_s": lead_s, "manual_end": manual, "blank_rule": blank_rule,
    }
    return {
        "rule_version": RULE_VERSION,
        "role": role,
        "inputs": {k: v for k, v in digest_src.items()},
        "input_digest": hashlib.sha256(
            __import__("json").dumps(digest_src, sort_keys=True,
                                     default=str).encode()).hexdigest(),
        "power_version_id": power.version_id,
        "injection_version_id": inj.version_id,
        "baseline_version_id": baseline.version_id if baseline else None,
        "time_reference": power.content.get("time_reference"),
        "n_points": len(pts),
        "injections": [vars(x) for x in injections],
    }
