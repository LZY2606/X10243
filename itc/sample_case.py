from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


INJECTION_TIMES = [50, 170, 290, 410, 530, 650]
INJECTION_VOLUMES = [2.0, 4.0, 4.0, 4.0, 4.0, 4.0]
SAMPLE_TIMES = list(range(0, 841, 2))


def _gaussian(t: float, center: float, amplitude: float, sigma: float) -> float:
    return amplitude * math.exp(-0.5 * ((t - center) / sigma) ** 2)


def _power_points(flat: bool = False, missing_gap: bool = False) -> list[dict[str, Any]]:
    points = []
    amplitudes = [12.0, 10.0, 8.0, 6.0, 4.5, 3.0]
    for t in SAMPLE_TIMES:
        value = 0.05 + 0.0007 * t
        if not flat:
            for center, amp, volume in zip(INJECTION_TIMES, amplitudes, INJECTION_VOLUMES):
                value += _gaussian(t, center + 8, amp * (0.7 + volume / 10), 24)
        if missing_gap and 300 <= t <= 360:
            points.append({"time": t, "value": None, "quality": "gap_placeholder"})
        else:
            points.append({"time": t, "value": round(value, 6), "quality": "instrument_sample"})
    return points


def _baseline_points() -> list[dict[str, Any]]:
    return [{"time": t, "value": round(0.05 + 0.0007 * t, 6), "quality": "baseline_sample"} for t in SAMPLE_TIMES]


def _blank_points() -> list[dict[str, Any]]:
    amplitudes = [1.1, 1.0, 0.9, 0.8, 0.75, 0.7]
    points = []
    for t in SAMPLE_TIMES:
        value = 0.05 + 0.0007 * t
        for center, amp, volume in zip(INJECTION_TIMES, amplitudes, INJECTION_VOLUMES):
            value += _gaussian(t, center + 8, amp * (0.7 + volume / 10), 24)
        points.append({"time": t, "value": round(value, 6), "quality": "blank_sample"})
    return points


def _schedule(times: list[int] | None = None, volumes: list[float] | None = None) -> dict[str, Any]:
    times = times or INJECTION_TIMES
    volumes = volumes or INJECTION_VOLUMES
    return {
        "provenance": {"instrument": "VP-ITC synthetic archive", "operator": "lab-review"},
        "injections": [
            {"index": index + 1, "time": {"value": time_value, "unit": "s"}, "volume": {"value": volume, "unit": "µL"}, "syringe": "ligand"}
            for index, (time_value, volume) in enumerate(zip(times, volumes))
        ],
    }


def _curve(points: list[dict[str, Any]], gaps: list[dict[str, Any]] | None = None, identity: str = "sample") -> dict[str, Any]:
    return {
        "provenance": {"source_run": identity, "instrument": "calorimeter"},
        "time": {"unit": "s", "zero": "injection_run_start"},
        "power": {"unit": "µW", "reference": "electrical_power_to_cell"},
        "points": points,
        "gaps": gaps or [],
    }


def initial_materials() -> dict[str, dict[str, Any]]:
    return {
        "power_curve": _curve(_power_points(), identity="sample-power"),
        "injection_schedule": _schedule(),
        "concentration": {
            "provenance": {"source": "lab_notebook/2026-09-28", "material": "ligand stock"},
            "measurement": {"value": "1.20", "unit": "mM"},
            "analyte_role": "syringe_ligand",
        },
        "temperature": {
            "provenance": {"source": "instrument_header", "stability": "equilibrated"},
            "measurement": {"value": 298.15, "unit": "K"},
        },
        "cell_volume": {
            "provenance": {"source": "instrument_header"},
            "measurement": {"value": 1.4, "unit": "mL"},
        },
        "baseline_curve": {
            **_curve(_baseline_points(), identity="pre-equilibration-baseline"),
            "curve_role": "baseline",
        },
        "blank_curve": {
            **_curve(_blank_points(), identity="blank-buffer-dilution"),
            "curve_role": "blank",
            "blank_identity": "buffer-into-buffer",
            "injections": _schedule()["injections"],
        },
    }


def build_case() -> dict[str, Any]:
    return {
        "case_id": "itc-titration-review-offline-v1",
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "domain": "biophysical_isothermal_titration_calorimetry",
        "offline": True,
        "narrative": [
            "导入原始功率曲线、注射时序和空白对照。",
            "核对分开版本化的功率、注射、浓度、温度、池体积、基线和空白。",
            "审议首注舍弃、重叠峰、空白对应、缺口、端点、浓度修订和等优模型。",
            "在联动图中比较一位点、两位点与稀释假设，并留下接受或否定事件。",
        ],
        "coordinate_reference": {
            "time": "experiment elapsed seconds",
            "power": "electrical power in µW",
            "heat": "trapezoidal integrated energy in µJ",
            "temperature_canonical": "°C",
        },
        "initial_rules": {
            "version": "rules:itc-review/v1",
            "first_injection": "discard",
            "window_rule": "close_at_next_injection",
            "post_window_s": 170,
            "blank_correspondence": "same_injection_index_and_matching_volume",
            "blank_volume_tolerance_ratio": 0.02,
            "cell_macromolecule_concentration_mM": 0.05,
        },
        "materials": initial_materials(),
        "operations": [
            {"id": "op-import", "type": "import_initial_materials", "branch": "main", "message": "原始材料分开入库并版本化"},
            {"id": "op-analyze-initial", "type": "analyze", "branch": "main", "after": "op-import", "message": "计算注射积分并比较绑定模型"},
            {"id": "op-concentration-revision", "type": "revise_material", "branch": "main", "after": "op-analyze-initial", "kind": "concentration", "direct_source": "lab_notebook/2026-09-28/correction-1", "material": {"provenance": {"source": "lab_notebook/2026-09-28/correction-1"}, "measurement": {"value": "1.08", "unit": "mM"}, "analyte_role": "syringe_ligand"}, "message": "浓度修订；原积分不变，绑定候选过期"},
            {"id": "op-analyze-revised", "type": "analyze", "branch": "main", "after": "op-concentration-revision", "reuse_integration_from": "op-analyze-initial"},
            {"id": "op-local-endpoint-change", "type": "revise_material", "branch": "main", "after": "op-analyze-revised", "kind": "injection_schedule", "direct_source": "injector-log/local-clock-correction", "material": _schedule(times=[50, 170, 290, 411, 530, 650]), "affected_scope": "injection_windows:3,4", "message": "仅第 4 次注射局部时钟修订，检验积分端点"},
            {"id": "op-analyze-local", "type": "analyze", "branch": "main", "after": "op-local-endpoint-change"},
            {"id": "op-decision-accept-one-site", "type": "decision", "branch": "main", "after": "op-analyze-local", "subject": {"kind": "model", "name": "one_site"}, "action": "accept", "rationale": "审阅者接受一位点候选"},
            {"id": "op-decision-reject-one-site-dilution", "type": "decision", "branch": "main", "after": "op-decision-accept-one-site", "subject": {"kind": "model", "name": "one_site"}, "action": "reject", "conflicts_with": "op-decision-accept-one-site", "rationale": "复核认为稀释贡献未被充分解释，否定先前决定"},
            {"id": "br-first-keep", "type": "create_branch", "branch": "first_injection_kept", "from_branch": "main", "after": "op-analyze-initial", "changed_direct_source": "rules:itc-review/v1#first_injection", "rules_patch": {"first_injection": "keep"}},
            {"id": "br-first-keep-analyze", "type": "analyze", "branch": "first_injection_kept", "after": "br-first-keep"},
            {"id": "br-blank-missing", "type": "create_branch", "branch": "blank_correspondence_missing", "from_branch": "main", "after": "op-analyze-initial", "changed_direct_source": "material:blank_curve", "remove_material": "blank_curve"},
            {"id": "br-blank-missing-analyze", "type": "analyze", "branch": "blank_correspondence_missing", "after": "br-blank-missing"},
            {"id": "br-flat", "type": "create_branch", "branch": "multiple_equivalent_models", "from_branch": "main", "after": "op-analyze-initial", "changed_direct_source": "material:power_curve", "revise_material": {"kind": "power_curve", "direct_source": "synthetic-flat-power", "material": _curve(_power_points(flat=True), identity="flat-no-binding")}},
            {"id": "br-flat-analyze", "type": "analyze", "branch": "multiple_equivalent_models", "after": "br-flat"},
            {"id": "br-gap", "type": "create_branch", "branch": "data_gap_window", "from_branch": "main", "after": "op-analyze-initial", "changed_direct_source": "material:power_curve", "revise_material": {"kind": "power_curve", "direct_source": "acquisition/gap-marker", "material": _curve(_power_points(missing_gap=True), gaps=[{"start": 300, "end": 360, "unit": "s", "reason": "detector dropout"}], identity="gapped-power")}},
            {"id": "br-gap-analyze", "type": "analyze", "branch": "data_gap_window", "after": "br-gap"},
            {"id": "br-endpoint", "type": "create_branch", "branch": "integration_endpoints", "from_branch": "main", "after": "op-analyze-initial", "changed_direct_source": "material:injection_schedule", "revise_material": {"kind": "injection_schedule", "direct_source": "injector-log/odd-sample-grid", "material": _schedule(times=[51, 171, 291, 411, 531, 651])}},
            {"id": "br-endpoint-analyze", "type": "analyze", "branch": "integration_endpoints", "after": "br-endpoint"},
            {"id": "br-semantic", "type": "create_branch", "branch": "semantic_no_solution", "from_branch": "main", "after": "op-analyze-initial", "changed_direct_source": "material:concentration", "revise_material": {"kind": "concentration", "direct_source": "lab_notebook/unreadable", "material": {"provenance": {"source": "lab_notebook/unreadable"}, "measurement": {"value": "N/A", "unit": "mM"}}}},
            {"id": "br-semantic-analyze", "type": "analyze", "branch": "semantic_no_solution", "after": "br-semantic", "expect_failure": "SEMANTIC_IMPOSSIBLE"},
        ],
        "expected_evidence": {
            "overlapping_injection": "injection entries expose overlaps_next_peak and window end at next injection",
            "first_injection": "main discards injection 1; rule branch keeps it",
            "blank_missing": "blank_correspondence.status is missing and corrected integral remains uncorrected",
            "concentration_revision": "new binding result depends on concentration v2 while integration references the original integral",
            "equivalent_models": "flat branch returns multiple equivalent legal model candidates",
            "data_gap": "windows intersecting a gap are invalid and no heat is interpolated across",
            "integration_endpoints": "odd-time windows expose endpoint interpolation diagnostics",
        },
    }


def write_case_package(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = build_case()
    case_path = path / "case.json"
    case_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "case_id": payload["case_id"],
        "title": payload["title"],
        "files": ["case.json", "README.md"],
        "domain_material": True,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "README.md").write_text(
        "# ITC 离线案例包\n\n本目录是滴定热谱领域案例，不是通用表格。`case.json` 含原始表示、坐标基准、版本化材料、分支和可重放事件指令。\n",
        encoding="utf-8",
    )
    return case_path


if __name__ == "__main__":
    write_case_package(Path(__file__).resolve().parents[1] / "cases" / "itc_titration_review")
