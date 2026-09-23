import math
from .errors import ResourceBudgetExceededError, SemanticNoSolutionError, diagnostic
from .units import normalize_power, normalize_time

DEFAULT_BUDGET = {"series_points": 50000, "injections": 200}


def check_budget(materials, budget=None):
    budget = budget or DEFAULT_BUDGET
    points = 0
    injections = 0
    for material in materials.values():
        if material.get("type") == "power_series":
            points += len(material.get("points", []))
        if material.get("type") == "injection_schedule":
            injections += len(material.get("injections", []))
    diagnostics = []
    if points > budget["series_points"]:
        diagnostics.append(diagnostic("budget.series_points", "功率采样点超过资源预算", points, f"<= {budget['series_points']}"))
    if injections > budget["injections"]:
        diagnostics.append(diagnostic("budget.injections", "注射数量超过资源预算", injections, f"<= {budget['injections']}"))
    if diagnostics:
        raise ResourceBudgetExceededError("资源预算耗尽；输入未被接收", diagnostics)


def prepare_series(material, path="materials"):
    if material.get("type") != "power_series":
        raise SemanticNoSolutionError("材料不是功率采样", [diagnostic(path, "type 必须为 power_series", material.get("type"))])
    time_unit = material.get("time_unit")
    power_unit = material.get("power_unit")
    times = []
    powers = []
    raw_points = []
    missing = []
    last_t = None
    for index, point in enumerate(material.get("points", [])):
        p = f"{path}.points[{index}]"
        if not isinstance(point, dict) or "t" not in point:
            raise SemanticNoSolutionError("功率采样点损坏", [diagnostic(p, "需要 {t, p?}", point)])
        tq = normalize_time(point["t"], time_unit, f"{p}.t")
        t = tq["normalized_value"]
        if last_t is not None and t <= last_t:
            raise SemanticNoSolutionError("功率采样时间必须严格递增", [diagnostic(f"{p}.t", "需要大于前一采样时间", t, f"(> {last_t} s)")])
        raw_points.append({"t": point["t"], "p": point.get("p"), "unit": {"t": time_unit, "p": power_unit}})
        if point.get("p") is None:
            powers.append(math.nan)
            missing.append(t)
        else:
            pq = normalize_power(point["p"], power_unit, f"{p}.p")
            powers.append(pq["normalized_value"])
        times.append(t)
        last_t = t
    gaps = []
    for gap in material.get("gaps", []):
        if not isinstance(gap, dict) or "start" not in gap or "end" not in gap:
            raise SemanticNoSolutionError("缺口声明需要 start/end", [diagnostic(f"{path}.gaps", "需要 {start,end,reason}", gap)])
        sq = normalize_time(gap["start"], time_unit, f"{path}.gaps.start")
        eq = normalize_time(gap["end"], time_unit, f"{path}.gaps.end")
        if eq["normalized_value"] <= sq["normalized_value"]:
            raise SemanticNoSolutionError("缺口结束时间必须晚于开始时间", [diagnostic(f"{path}.gaps", "end > start", gap)])
        gaps.append({"start": sq["normalized_value"], "end": eq["normalized_value"], "reason": gap.get("reason", "declared_gap"), "raw": gap})
    if missing:
        start = missing[0]
        end = missing[-1]
        if not any(abs(g["start"] - start) < 1e-9 and abs(g["end"] - end) < 1e-9 for g in gaps):
            gaps.append({"start": start, "end": end, "reason": "null_sample", "raw": {"start": start, "end": end}})
    return {"times": times, "powers": powers, "raw_points": raw_points, "gaps": gaps, "time_unit": "s", "power_unit": "µW"}
