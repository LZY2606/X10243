import sys
sys.path.insert(0, ".")

import math

import pytest

from itc.integrate import SemanticUnsolvable, correct_blank, integrate


def make_power(peaks, gap=None, t_end=600.0, tau=20.0):
    ts, ps = [], []
    t = 0.0
    while t <= t_end:
        if gap and gap[0] <= t < gap[1]:
            t += 1.0
            continue
        p = 0.5
        for t0, q in peaks:
            if t >= t0:
                p += (q / tau) * math.exp(-(t - t0) / tau)
        ts.append(t)
        ps.append(p)
        t += 1.0
    segments = [{"t": [], "p": []}]
    for t, p in zip(ts, ps):
        if segments[-1]["t"] and t - segments[-1]["t"][-1] > 1.5:
            segments.append({"t": [], "p": []})
        segments[-1]["t"].append(t)
        segments[-1]["p"].append(p)
    return {"time_unit": "s", "power_unit": "ucal/s",
            "time_base": "seconds_since_run_start", "segments": segments}


def injections(entries):
    return {"time_unit": "s", "volume_unit": "uL",
            "entries": [{"index": i, "time": t, "volume": v}
                        for i, (t, v) in enumerate(entries)]}


BASELINE = {"kind": "poly", "coeffs": [0.0, 0.5]}


def test_overlapping_peaks_integrate_each_window():
    power = make_power([(100.0, -30.0), (140.0, -25.0)], tau=35.0)
    inj = injections([(100.0, 10.0), (140.0, 10.0)])
    out = integrate(power, inj, BASELINE, {"window_seconds": 60})
    assert [r["status"] for r in out["injections"]] == ["ok", "ok"]
    assert out["injections"][1]["peak_overlap"] is True
    assert out["injections"][0]["heat_ucal"] < 0


def test_first_injection_discarded_not_deleted():
    power = make_power([(100.0, -2.0), (200.0, -30.0)])
    inj = injections([(100.0, 2.0), (200.0, 10.0)])
    out = integrate(power, inj, BASELINE,
                    {"window_seconds": 60, "discard_first": True})
    first, second = out["injections"]
    assert first["excluded"] is True
    assert first["exclude_reason"] == "first_injection_discarded"
    assert first["heat_ucal"] is not None
    assert second["excluded"] is False


def test_window_crossing_gap_is_diagnostic_not_fabricated():
    power = make_power([(930.0, -30.0)], gap=(1000.0, 1050.0), t_end=1200.0)
    inj = injections([(930.0, 10.0)])
    out = integrate(power, inj, BASELINE, {"window_seconds": 90})
    rec = out["injections"][0]
    assert rec["status"] == "invalid"
    assert rec["heat_ucal"] is None
    assert rec["diagnostics"][0]["code"] == "integration_window_crosses_gap"
    assert rec["diagnostics"][0]["location"]["injection_index"] == 0


def test_window_ending_exactly_at_gap_start_is_allowed():
    power = make_power([(900.0, -30.0)], gap=(1000.0, 1050.0), t_end=1200.0)
    inj = injections([(900.0, 10.0)])
    out = integrate(power, inj, BASELINE, {"window_seconds": 99})
    assert out["injections"][0]["status"] == "ok"


def test_window_starting_inside_gap_is_invalid():
    power = make_power([(1020.0, -30.0)], gap=(1000.0, 1050.0), t_end=1200.0)
    inj = injections([(1020.0, 10.0)])
    out = integrate(power, inj, BASELINE, {"window_seconds": 60})
    assert out["injections"][0]["status"] == "invalid"


def test_blank_correspondence_missing_keeps_raw_value():
    payload = {"injections": [
        {"index": 0, "time": 100.0, "volume": 10.0, "window": [100, 160],
         "status": "ok", "heat_ucal": -12.0, "excluded": False, "diagnostics": []},
        {"index": 1, "time": 200.0, "volume": 10.0, "window": [200, 260],
         "status": "ok", "heat_ucal": -11.0, "excluded": False, "diagnostics": []}]}
    blank = {"entries": [{"injection_index": 0, "heat": -2.0, "heat_unit": "ucal"}]}
    out = correct_blank(payload, blank, {"rule": "by_index"})
    first, second = out["injections"]
    assert first["blank_applied"] is True
    assert first["corrected_ucal"] == pytest.approx(-10.0)
    assert second["blank_applied"] is False
    assert second["corrected_ucal"] == pytest.approx(-11.0)
    assert second["diagnostics"][0]["code"] == "blank_correspondence_missing"
    assert second["diagnostics"][0]["location"]["injection_index"] == 1


def test_blank_rule_must_be_stated():
    payload = {"injections": [
        {"index": 0, "time": 100.0, "volume": 10.0, "window": [100, 160],
         "status": "ok", "heat_ucal": -12.0, "excluded": False, "diagnostics": []}]}
    out = correct_blank(payload, {"entries": []}, {})
    assert out["status"] == "diagnostic"
    assert out["diagnostics"][0]["code"] == "blank_rule_unspecified"


def test_semantically_unsolvable_integration():
    power = make_power([(100.0, -30.0)])
    inj = injections([(100.0, 10.0)])
    with pytest.raises(SemanticUnsolvable):
        integrate(power, inj, BASELINE, {"window_seconds": 0})
