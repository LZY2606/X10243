"""逐注射积分引擎。

领域规则（也是机器可读事件的来源）：
  * peaks_may_overlap          注射后峰可重叠：窗口允许与相邻注射相接/重叠，
                               重叠只标记，不丢数据。
  * window_must_not_cross_gap  积分窗不跨越数据缺口：遇到缺口在缺口起点裁窗，
                               记录 window_before/window_after。
  * endpoints_snap_to_samples  积分端点必须落在真实采样时刻。
  * first_injection_discarded  首注体积小于阈值时舍弃，不进入拟合，但保留积分。
  * dilution 规则必须显式声明：none / per_injection_blank；空白对应缺失时
                               heat_net 留空并诊断，绝不拿零值伪装。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .errors import Budget, Diagnostic, Outcome, SemanticUnsolvable

PARAM_VERSION = "integ-params/v1"


@dataclass
class IntegrationParams:
    max_window_s: float = 240.0
    gap_factor: float = 3.0
    overlap_decay_s: float = 60.0
    first_injection_min_uL: float = 1.0
    dilution_rule: str = "per_injection_blank"  # none | per_injection_blank
    param_version: str = PARAM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_gaps(t: np.ndarray, gap_factor: float
                ) -> list[tuple[float, float]]:
    if len(t) < 3:
        return []
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        return []
    return [(float(t[i]), float(t[i + 1]))
            for i, d in enumerate(dt) if d > gap_factor * med]


def _baseline_at(bt: np.ndarray, bp: np.ndarray, t: np.ndarray
                 ) -> tuple[np.ndarray, list[str]]:
    b = np.interp(t, bt, bp)
    outside: list[str] = []
    if len(bt) and (t[0] < bt[0] - 1e-9 or t[-1] > bt[-1] + 1e-9):
        outside = ["baseline_range_exceeded"]
    return b, outside


def integrate(
    power: dict[str, Any],
    schedule: dict[str, Any],
    baseline: dict[str, Any],
    blank: dict[str, Any] | None,
    blank_ref: str | None,
    params: IntegrationParams,
    budget: Budget,
) -> Outcome:
    diagnostics: list[Diagnostic] = []
    t = np.asarray(power["t"], dtype=float)
    p = np.asarray(power["p"], dtype=float)
    bt = np.asarray(baseline["t"], dtype=float)
    bp = np.asarray(baseline["p"], dtype=float)
    injections = schedule["injections"]
    gaps = detect_gaps(t, params.gap_factor)

    rule_events: list[dict[str, Any]] = [{
        "rule": "endpoints_snap_to_samples",
        "basis": {"time": "s", "power": "uJ/s", "heat": "uJ"},
        "description": "积分端点一律吸附到真实采样时刻，记录端点来源",
    }]

    rows: list[dict[str, Any]] = []
    overlap_seen: set[tuple[str, str]] = set()

    for i, inj in enumerate(injections):
        budget.spend(1, f"injection[{i}]")
        iid = f"inj:{inj['index']}"
        t0_req = float(inj["time_s"])
        next_t = (float(injections[i + 1]["time_s"])
                  if i + 1 < len(injections) else None)
        t1_req = min(t0_req + params.max_window_s,
                     next_t if next_t is not None else t0_req + params.max_window_s)
        t1_req = min(t1_req, float(t[-1]))

        flags: list[str] = []
        inj_diag: list[Diagnostic] = []

        # 重叠峰：下一注射早于衰减窗口 → 允许，标记
        if next_t is not None and next_t - t0_req < params.overlap_decay_s:
            jid = f"inj:{injections[i + 1]['index']}"
            pair = tuple(sorted((iid, jid)))
            flags.append("overlap")
            if pair not in overlap_seen:
                overlap_seen.add(pair)  # type: ignore[arg-type]
                rule_events.append({
                    "rule": "peaks_may_overlap",
                    "injections": list(pair),
                    "spacing_s": next_t - t0_req,
                    "decay_window_s": params.overlap_decay_s,
                    "decision": "windows_kept_flagged",
                })

        # 端点吸附到采样点
        inside = np.where((t >= t0_req) & (t <= t1_req))[0]
        if len(inside) == 0:
            rows.append({
                "injection_id": iid, "index": inj["index"],
                "time_s": t0_req, "volume_uL": float(inj["volume_uL"]),
                "window_requested_s": [t0_req, t1_req],
                "window_s": None, "n_samples": 0,
                "heat_raw_uJ": None, "baseline_level_uJ_s": None,
                "blank_heat_uJ": None, "heat_net_uJ": None,
                "endpoint_source": None,
                "flags": flags + ["no_data"], "status": "no_data",
                "usable_for_fit": False,
                "diagnostics": [Diagnostic(
                    "no_samples_in_window",
                    f"{iid}: 请求窗口 [{t0_req:.3g},{t1_req:.3g}]s 内无采样点",
                    f"injections[{inj['index']}].window").to_dict()],
            })
            continue

        i0, i1 = int(inside[0]), int(inside[-1])
        t0, t1 = float(t[i0]), float(t[i1])
        window_before = [t0, t1]

        # 积分窗不跨越数据缺口：在第一个缺口处裁窗
        clipped = False
        for gs, ge in gaps:
            if t0 < gs < t1:
                left = np.where((t >= t0) & (t < gs))[0]
                rule_events.append({
                    "rule": "window_must_not_cross_gap",
                    "injection": iid,
                    "gap_s": [gs, ge],
                    "window_before_s": window_before,
                    "window_after_s": [t0, float(t[left[-1]])] if len(left) else None,
                    "decision": "clip_at_gap_start",
                })
                if len(left) == 0:
                    inj_diag.append(Diagnostic(
                        "no_samples_before_gap",
                        f"{iid}: 缺口 {gs:.3g}s 之前无样本，无法积分",
                        f"injections[{inj['index']}].window"))
                    flags.append("gap_blocked")
                else:
                    i1 = int(left[-1])
                    t1 = float(t[i1])
                    flags.append("gap_clipped")
                    clipped = True
                break

        if "gap_blocked" in flags:
            rows.append({
                "injection_id": iid, "index": inj["index"],
                "time_s": t0_req, "volume_uL": float(inj["volume_uL"]),
                "window_requested_s": [t0_req, t1_req],
                "window_s": None, "n_samples": 0,
                "heat_raw_uJ": None, "baseline_level_uJ_s": None,
                "blank_heat_uJ": None, "heat_net_uJ": None,
                "endpoint_source": None,
                "flags": flags, "status": "gap_blocked",
                "usable_for_fit": False,
                "diagnostics": [d.to_dict() for d in inj_diag],
            })
            continue

        tw = t[i0:i1 + 1]
        pw = p[i0:i1 + 1]
        bvals, outside = _baseline_at(bt, bp, tw)
        heat_raw = float(np.trapezoid(pw - bvals, tw))
        base_level = float(np.mean(bvals))

        # 首注舍弃（保留积分值，仅不参与拟合）
        discarded = (i == 0 and float(inj["volume_uL"])
                     < params.first_injection_min_uL)
        if discarded:
            flags.append("discarded")
            rule_events.append({
                "rule": "first_injection_discarded",
                "injection": iid,
                "volume_uL": float(inj["volume_uL"]),
                "threshold_uL": params.first_injection_min_uL,
                "decision": "excluded_from_fit_heat_kept",
            })

        # 稀释热扣除：必须说明对应规则
        blank_heat = None
        heat_net = None
        dilution = {"rule": params.dilution_rule, "blank": blank_ref}
        if params.dilution_rule == "none":
            heat_net = heat_raw
            dilution["decision"] = "no_subtraction"
        elif params.dilution_rule == "per_injection_blank":
            bh, bdiag = _blank_heat(blank, t0, t1, iid)
            if bdiag:
                flags.append("blank_missing")
                inj_diag.extend(bdiag)
                dilution["decision"] = "not_subtracted_blank_correspondence_missing"
            else:
                blank_heat = bh
                heat_net = heat_raw - bh
                dilution["decision"] = "subtracted_per_injection"
        else:
            raise SemanticUnsolvable(
                f"未知稀释热规则 {params.dilution_rule!r}")

        status = "discarded" if discarded else "ok"
        usable_for_fit = (
            not discarded
            and heat_net is not None
            and not clipped
            and not outside
        )
        rows.append({
            "injection_id": iid, "index": inj["index"],
            "time_s": t0_req, "volume_uL": float(inj["volume_uL"]),
            "window_requested_s": [t0_req, t1_req],
            "window_s": [t0, t1],
            "n_samples": int(i1 - i0 + 1),
            "heat_raw_uJ": heat_raw,
            "baseline_level_uJ_s": base_level,
            "blank_heat_uJ": blank_heat,
            "heat_net_uJ": heat_net,
            "endpoint_source": "sample",
            "endpoint_samples_s": [t0, t1],
            "gap_clipped": clipped,
            "flags": flags, "status": status,
            "usable_for_fit": usable_for_fit,
            "outside_baseline": outside,
            "dilution": dilution,
            "diagnostics": [d.to_dict() for d in inj_diag],
        })

    # 统一稀释规则事件（缺失对应清单）
    missing = [r["injection_id"] for r in rows if "blank_missing" in r["flags"]]
    rule_events.append({
        "rule": "dilution_correspondence",
        "rule_name": params.dilution_rule,
        "blank": blank_ref,
        "missing_injections": missing,
        "decision": ("keep_unsubtracted_with_diagnostic" if missing
                     else "all_subtracted"),
    })

    n_ok = sum(1 for r in rows if r["heat_net_uJ"] is not None
               and "discarded" not in r["flags"])
    diagnostics += [
        Diagnostic(d["code"], d["message"], d.get("path", ""),
                   d.get("severity", "error"))
        for r in rows for d in r["diagnostics"]]

    payload = {
        "kind": "integration",
        "param_version": params.param_version,
        "params": params.to_dict(),
        "units": {"time": "s", "power": "uJ/s", "heat": "uJ", "volume": "uL"},
        "gaps_s": gaps,
        "rule_events": rule_events,
        "injections": rows,
        "n_usable_for_fit": n_ok,
    }

    if not rows:
        return Outcome.fail("semantic_unsolvable", [Diagnostic(
            "no_injections", "时序中没有可用注射", "injections")])
    if n_ok == 0:
        return Outcome.fail("semantic_unsolvable", diagnostics + [Diagnostic(
            "no_integrable_injections",
            "没有任何注射同时满足：未舍弃、空白对应齐全、窗口有效",
            "integration")])
    return Outcome.good(payload, diagnostics)


def _blank_heat(blank: dict[str, Any] | None, t0: float, t1: float,
                iid: str
                ) -> tuple[float | None, list[Diagnostic]]:
    # 空白曲线本身是零基准差功率（baseline 单独版本化，不在这里复用主曲线基线）
    if blank is None:
        return None, [Diagnostic(
            "blank_missing",
            f"{iid}: 无空白对照材料，稀释热未扣除，heat_net 留空",
            f"integration.{iid}.blank")]
    bt2 = np.asarray(blank["t"], dtype=float)
    bp2 = np.asarray(blank["p"], dtype=float)
    mask = (bt2 >= t0) & (bt2 <= t1)
    if not np.any(mask) or bt2[0] > t0 or bt2[-1] < t1:
        return None, [Diagnostic(
            "blank_correspondence_missing",
            f"{iid}: 空白曲线不覆盖窗口 [{t0:.3g},{t1:.3g}]s，"
            "稀释热对应缺失，heat_net 留空",
            f"integration.{iid}.blank")]
    return float(np.trapezoid(bp2[mask], bt2[mask])), []
