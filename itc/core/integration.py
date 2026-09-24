"""逐注射积分：power - baseline 在窗内梯形积分 -> 原始热量 Q（µcal）。

integration-trapezoid-v1：
- 只积分 blocked 以外、端点之间的有效样本；
- 基线支持曲线（线性插值）、常数、多项式；
- 原始热量是上游结果：浓度修订不会改写它。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (STATUS_BLOCKED_BASELINE, STATUS_BLOCKED_GAP,
                        STATUS_DISCARDED, STATUS_OK,
                        INTEGRATION_RULE_VERSION)
from .errors import SemanticUnsolvableError
from .materials import baseline_view


def _baseline_at(t: np.ndarray, power_material: dict[str, Any],
                 baseline_material: dict[str, Any] | None) -> np.ndarray:
    if baseline_material is None:
        raise SemanticUnsolvableError(
            "缺少基线材料，无法积分", {"slot": "baseline"})
    view = baseline_view(baseline_material)
    if view.get("model") == "constant":
        from .units import parse_quantity
        q = parse_quantity(view["value_raw"], "power", view.get("unit",
                                                                "ucal/s"))
        if q.value is None:
            raise SemanticUnsolvableError("基线常数无法规范化",
                                          {"raw": view["value_raw"]})
        return np.full_like(t, q.value, dtype=float)
    if view.get("model") == "polynomial":
        coeff = [float(c) for c in view["coeff"]]
        return np.polyval(coeff, t)
    bt = np.array([x for x, ok in zip(view["t"], view["valid_t"]) if ok],
                  dtype=float)
    bp = np.array([x for x, ok in zip(view["p"], view["valid_p"]) if ok],
                  dtype=float)
    order = np.argsort(bt)
    return np.interp(t, bt[order], bp[order])


def integrate_all(power_material: dict[str, Any],
                  baseline_material: dict[str, Any] | None,
                  windows: list[dict[str, Any]],
                  power_view: dict[str, Any],
                  discarded: set[int]
                  ) -> list[dict[str, Any]]:
    t_all = np.array([x if x is not None else np.nan
                      for x in power_view["t"]], dtype=float)
    p_all = np.array([x if x is not None else np.nan
                      for x in power_view["p"]], dtype=float)
    try:
        base_all = _baseline_at(t_all, power_material, baseline_material)
        have_baseline = True
    except SemanticUnsolvableError:
        base_all = np.zeros_like(t_all)
        have_baseline = False

    results = []
    for w in windows:
        ordinal = w["ordinal"]
        record = {
            "ordinal": ordinal,
            "label": w["label"],
            "window": {"start": w["start"], "end": w["end"]},
            "unit": "ucal",
            "rule_version": INTEGRATION_RULE_VERSION,
        }
        if w["status"] == STATUS_BLOCKED_GAP:
            record.update(status=STATUS_BLOCKED_GAP, value=None,
                          reason=w.get("reason"),
                          gap_interval=w.get("gap_interval"))
            results.append(record)
            continue
        if not have_baseline:
            record.update(status=STATUS_BLOCKED_BASELINE, value=None)
            results.append(record)
            continue
        i0, i1 = w["start_index"], w["end_index"]
        seg_t = t_all[i0:i1 + 1]
        net = p_all[i0:i1 + 1] - base_all[i0:i1 + 1]
        q = float(np.trapezoid(net, seg_t))
        record.update(value=q, status=STATUS_OK,
                      n_samples=int(i1 - i0 + 1))
        if ordinal in discarded:
            record["status"] = STATUS_DISCARDED
            record["participates"] = False
        results.append(record)
    results.sort(key=lambda r: r["ordinal"])
    return results
