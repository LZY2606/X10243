import sys
sys.path.insert(0, ".")

import math

import pytest

from itc.models import (BudgetExhausted, SemanticUnsolvable, fit_models,
                        predict)


def cond():
    return {"V0": 1400e-6, "M0": 0.06e-3, "Xs": 0.8e-3,
            "dv": [2e-6] + [10e-6] * 17}


def noisy_heats(noise=0.0):
    true = predict("one_site", [0.95, math.log10(3e6), -7000.0], cond())
    return [q + noise * math.sin(i * 2.7) for i, q in enumerate(true)]


def included():
    return [False] + [True] * 17


def test_one_site_recovers_parameters():
    fit = fit_models(noisy_heats(), included(), cond(), families=("one_site",))
    best = fit["candidates"][0]
    assert best["family"] == "one_site"
    assert best["params"]["n"] == pytest.approx(0.95, abs=0.05)
    assert best["params"]["log10_K"] == pytest.approx(math.log10(3e6), abs=0.3)
    assert best["params"]["dH_cal_per_mol"] == pytest.approx(-7000, abs=400)


def test_equipotent_candidates_all_kept_in_stable_order():
    kwargs = dict(families=("one_site", "two_site", "dilution"), budget=20000)
    fit1 = fit_models(noisy_heats(0.3), included(), cond(), **kwargs)
    fit2 = fit_models(noisy_heats(0.3), included(), cond(), **kwargs)
    eq1 = [c for c in fit1["candidates"] if c["equipotent"]]
    assert len(eq1) >= 2
    ids1 = [c["id"] for c in fit1["candidates"]]
    ids2 = [c["id"] for c in fit2["candidates"]]
    assert ids1 == ids2
    ssrs = [c["ssr"] for c in fit1["candidates"]]
    assert ssrs == sorted(ssrs)


def test_budget_exhaustion_is_distinct():
    with pytest.raises(BudgetExhausted) as exc:
        fit_models(noisy_heats(), included(), cond(), budget=5)
    assert exc.value.status == "budget_exhausted"


def test_semantic_unsolvable_is_distinct():
    bad = cond()
    bad["M0"] = 0.0
    with pytest.raises(SemanticUnsolvable) as exc:
        fit_models(noisy_heats(), included(), bad)
    assert exc.value.status == "semantic_unsolvable"
    with pytest.raises(SemanticUnsolvable):
        fit_models(noisy_heats(), [False] * 18, cond())


def test_dilution_family_fits_constant_data():
    heats = [-2.5] * 18
    fit = fit_models(heats, included(), cond(), families=("dilution",))
    assert fit["candidates"][0]["params"]["q_constant_ucal"] == pytest.approx(-2.5, abs=1e-3)
