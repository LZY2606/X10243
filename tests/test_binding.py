import numpy as np
import pytest

from itc.binding import bound_conc, fit_models, predict_heats


def _one_site_data(K=10.0, dH=-30.0, Mt=0.1, V0=1400.0, n=8):
    X = np.linspace(0.002, 0.03, n)
    heats = predict_heats("one_site", (K, dH), Mt, X, V0)
    return heats, X


def test_two_site_degenerates_to_one_site():
    heats, X = _one_site_data()
    pred = predict_heats("two_site", (10.0, 10.0, -30.0), 0.1, X, 1400.0)
    assert np.allclose(pred, heats)


def test_equipotent_candidates_all_kept():
    heats, X = _one_site_data()
    status, payload = fit_models(heats.tolist(), 0.1, X.tolist(), 1400.0)
    assert status == "ok"
    assert payload["candidates"] == ["one_site", "two_site"]
    assert payload["unique_decision"] is False
    assert len(payload["models"]) == 3  # dilution hypothesis kept too
    dil = [m for m in payload["models"] if m["model"] == "dilution_const"][0]
    assert dil["is_candidate"] is False


def test_budget_exhausted_distinct_from_success():
    heats, X = _one_site_data()
    status, payload = fit_models(heats.tolist(), 0.1, X.tolist(), 1400.0, budget=5)
    assert status == "budget_exhausted"
    assert payload is None


def test_bound_conc_limits():
    assert bound_conc(10.0, 0.1, 0.0) == pytest.approx(0.0)
    assert bound_conc(10.0, 0.1, 100.0) == pytest.approx(0.1, rel=1e-3)
