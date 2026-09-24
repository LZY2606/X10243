"""积分窗：注射后峰允许重叠，但窗不跨越数据缺口。

规则（window-v2）：
1. 每针默认窗 [t_i, t_{i+1}]，末针到最后一个有效采样时刻；
2. 相邻峰重叠时，两窗在间隔中点分区（partition），互不重复计数，标 overlap；
3. 人工端点可缩短窗；分区仍作用于缩短后的窗；
4. 窗端点必须落在有效采样时刻（容差 1e-6 s），否则 semantic_unsolvable，
   诊断带注射序号与时间坐标；
5. 窗内部包含任何缺口区间（含单点缺口）即 blocked:gap。
"""
from __future__ import annotations

from typing import Any

from .constants import STATUS_BLOCKED_GAP, STATUS_OK, WINDOW_RULE_VERSION
from .errors import Diagnostic, Location, SemanticUnsolvableError

EPS = 1e-6


def _sample_index(times: list[float | None], valid: list[bool],
                  value: float, label: str, ordinal: int
                  ) -> tuple[int | None, Diagnostic | None]:
    for i, tv in enumerate(times):
        if valid[i] and tv is not None and abs(tv - value) <= EPS:
            return i, None
    return None, Diagnostic(
        "endpoint_not_on_sample",
        f"注射 {ordinal} 的积分{label} {value}s 不落在有效采样时刻",
        location=Location(slot="power", injection=ordinal, at=value))


def compute_windows(injections: list[dict[str, Any]],
                    power: dict[str, Any],
                    endpoint_overrides: dict[int, dict[str, float]] | None = None
                    ) -> list[dict[str, Any]]:
    endpoint_overrides = endpoint_overrides or {}
    times = power["t"]
    valid_t = power["valid_t"]
    valid_p = power["valid_p"]
    valid = [a and b for a, b in zip(valid_t, valid_p)]
    valid_times = [t for t, ok in zip(times, valid) if ok and t is not None]
    t_end_default = valid_times[-1] if valid_times else None

    placed = [inj for inj in injections if inj.get("t") is not None]
    placed.sort(key=lambda x: x["t"])
    result: list[dict[str, Any]] = []
    n = len(placed)

    for k, inj in enumerate(placed):
        ordinal = inj["ordinal"]
        t0 = inj["t"]
        next_t = placed[k + 1]["t"] if k + 1 < n else None
        default_end = next_t if next_t is not None else t_end_default
        ov = endpoint_overrides.get(ordinal, {})
        start = float(ov.get("start", t0))
        end_req = ov.get("end", default_end)
        if end_req is None:
            raise SemanticUnsolvableError(
                f"注射 {ordinal} 无法确定积分窗结束时刻",
                Location(slot="power", injection=ordinal))
        end = float(end_req)

        overlap = False
        partition: float | None = None
        if next_t is not None:
            partition = (t0 + next_t) / 2.0
            if start < t0 - EPS:
                raise SemanticUnsolvableError(
                    f"注射 {ordinal} 积分窗起点早于注射时刻",
                    Location(slot="power", injection=ordinal, at=start))
            if end > partition + EPS or next_t - t0 < EPS:
                end = min(end, partition)
            overlap = True
        if start > end + EPS:
            return_blocked = {
                "ordinal": ordinal, "label": inj["label"],
                "start": start, "end": end, "start_index": None,
                "end_index": None, "status": STATUS_BLOCKED_GAP,
                "reason": "window_reversed", "overlap": overlap,
                "partition": partition,
                "rule_version": WINDOW_RULE_VERSION,
                "injection_t": t0,
            }
            result.append(return_blocked)
            continue

        i0, d0 = _sample_index(times, valid, start, "起点", ordinal)
        i1, d1 = _sample_index(times, valid, end, "终点", ordinal)
        diagnostics = [d for d in (d0, d1) if d is not None]
        if diagnostics:
            raise SemanticUnsolvableError(
                f"注射 {ordinal} 积分端点无效",
                diagnostics[0].location,
                {"diagnostics": [d.to_dict() for d in diagnostics]})

        gap_hit = _gap_inside(power, start, end)
        status = STATUS_BLOCKED_GAP if gap_hit else STATUS_OK
        result.append({
            "ordinal": ordinal,
            "label": inj["label"],
            "start": start,
            "end": end,
            "start_index": i0,
            "end_index": i1,
            "status": status,
            "reason": "crosses_data_gap" if gap_hit else None,
            "overlap": overlap,
            "partition": partition,
            "rule_version": WINDOW_RULE_VERSION,
            "injection_t": t0,
            "gap_interval": gap_hit,
        })
    result.sort(key=lambda w: w["ordinal"])
    return result


def _gap_inside(power: dict[str, Any], start: float,
                end: float) -> tuple[float, float] | None:
    for a, b in power.get("gap_time_intervals", []):
        if a is None or b is None:
            continue
        if a > b:
            a, b = b, a
        if start + EPS <= a and b <= end - EPS:
            return (a, b)
        if start < a < end or start < b < end:
            return (a, b)
    return None
