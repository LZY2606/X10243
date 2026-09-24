import numpy as np

from itc.integrate import find_gaps, integrate


def _trace(gap=None, t_end=400.0):
    t = np.arange(0.0, t_end, 1.0)
    p = 0.5 + 0.0 * t
    for ti, h in [(60.0, -20.0), (150.0, -25.0), (195.0, -25.0), (300.0, -20.0)]:
        m = t >= ti
        p[m] += (h / 10.0) * np.exp(-(t[m] - ti) / 10.0)
    keep = np.ones_like(t, dtype=bool)
    if gap:
        keep = (t < gap[0]) | (t >= gap[1])
    return {"samples": [{"t_s": float(a), "power_uW": float(b)} for a, b in zip(t[keep], p[keep])]}


def _injs(times):
    return [{"id": f"I{i+1:02d}", "time_s": t, "volume_uL": 5.0} for i, t in enumerate(times)]


def _base():
    return {"samples": [{"t_s": float(t), "power_uW": 0.5} for t in range(0, 400, 5)]}


def test_overlapping_peaks_allowed_and_recorded():
    # I2 at 150 and I3 at 195: post=45 window of I2 reaches the next rise.
    status, payload = integrate(_trace(), _injs([60.0, 150.0, 195.0, 300.0]), _base())
    assert status == "ok"
    i2 = payload["injections"][1]
    assert i2["status"] == "ok"
    assert any(d["code"] == "overlapping-peaks" for d in i2["diagnostics"])
    assert i2["window"][1] == 195.0 - 0.5  # clamped to next injection start


def test_window_never_crosses_data_gap():
    status, payload = integrate(_trace(gap=(210.0, 230.0)),
                                _injs([60.0, 150.0, 195.0, 300.0]), _base())
    assert status == "ok"
    i3 = payload["injections"][2]  # window [193, 239.5] crosses gap at 210
    assert i3["status"] == "truncated_gap"
    assert i3["window"][1] == 210.0
    assert any(d["code"] == "window-truncated-at-gap" for d in i3["diagnostics"])
    for e in payload["injections"]:
        if e["window"]:
            for g in payload["gaps"]:
                assert not (e["window"][0] < g[0] < e["window"][1])


def test_injection_inside_gap_is_invalid():
    status, payload = integrate(_trace(gap=(290.0, 310.0)),
                                _injs([60.0, 150.0, 195.0, 300.0]), _base())
    i4 = payload["injections"][3]
    assert i4["status"] == "invalid"
    assert any(d["code"] == "gap-at-injection" for d in i4["diagnostics"])


def test_first_injection_discarded_but_kept():
    status, payload = integrate(_trace(), _injs([60.0, 150.0, 195.0, 300.0]), _base())
    i1 = payload["injections"][0]
    assert i1["status"] == "discarded"
    assert i1["heat_uJ"] is not None  # kept on record
    status2, payload2 = integrate(_trace(), _injs([60.0, 150.0, 195.0, 300.0]), _base(),
                                  config={"discard_first": False})
    assert payload2["injections"][0]["status"] == "ok"


def test_gap_detection():
    times = [0.0, 1.0, 2.0, 3.0, 40.0, 41.0]
    assert find_gaps(times, 5.0) == [[3.0, 40.0]]


def test_budget_exhausted_is_distinct():
    status, payload = integrate(_trace(), _injs([60.0, 150.0, 195.0, 300.0]), _base(), budget=2)
    assert status == "budget_exhausted"
    assert payload is None


def test_equivalent_inputs_stable_order_and_digest():
    a = integrate(_trace(), _injs([60.0, 150.0, 195.0, 300.0]), _base())
    b = integrate(_trace(), _injs([60.0, 150.0, 195.0, 300.0]), _base())
    assert a == b
