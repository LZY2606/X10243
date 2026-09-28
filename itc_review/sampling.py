"""功率采样的有效段与数据缺口检测（纯函数，可直接被测试调用）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Segments:
    times: np.ndarray        # 仅有效（time、power 均可解析）采样点
    powers: np.ndarray
    median_dt: float
    gap_threshold: float
    gap_edges: list[tuple[float, float]]  # (last_before_gap, first_after_gap)
    segments: list[tuple[int, int]]       # 有效点在 times 中的下标区间 [a,b]
    invalid_points: list[int]             # 原始采样行 index（无法解析者留在原位置）


def build_segments(power_payload: dict, gap_factor: float) -> Segments:
    samples = power_payload["samples"]
    valid = []
    invalid = []
    for s in samples:
        if s["time"]["valid"] and s["power"]["valid"]:
            valid.append((s["time"]["canonical"], s["power"]["canonical"]))
        else:
            invalid.append(s["index"])
    times = np.array([v[0] for v in valid], dtype=float)
    powers = np.array([v[1] for v in valid], dtype=float)

    gap_edges: list[tuple[float, float]] = []
    segments: list[tuple[int, int]] = []
    if len(times) >= 2:
        dts = np.diff(times)
        median_dt = float(np.median(dts))
        threshold = gap_factor * median_dt
        start = 0
        for i, dt in enumerate(dts):
            if dt > threshold:
                gap_edges.append((float(times[i]), float(times[i + 1])))
                segments.append((start, i))  # 段尾为缺口左侧点
                start = i + 1
        segments.append((start, len(times) - 1))
    else:
        median_dt = 0.0
        threshold = 0.0
        if len(times) == 1:
            segments.append((0, 0))
    return Segments(times, powers, median_dt, threshold, gap_edges, segments, invalid)


def segment_for_time(seg: Segments, t: float) -> tuple[int, int] | None:
    """返回包含 t 的有效段（t 位于段端点闭区间内）；不跨缺口。"""
    idx = np.searchsorted(seg.times, t)
    # 候选段：包含插入位置的段
    for a, b in seg.segments:
        if a <= idx <= b:
            return a, b
    # t 恰好落在右端点
    if len(seg.times) and seg.times[-1] == t:
        return seg.segments[-1]
    return None
