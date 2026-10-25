"""Validator gate: permutation, BH-FDR, effect floor, held-out-state and held-out-year replication,
verdict rule."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from theorylab.executor import fixed_effects
from theorylab.validator import (GateInput, UnitPermuter, benjamini_hochberg, decide_verdict,
                                 difference_in_means, group_folds, held_out_group_replication,
                                 held_out_year_replication, passes_effect_floor, permutation_test)


def _county_panel(effect=-1.0, counties=60, per=10, years=(2020, 2021, 2022), seed=0,
                  effect_by_year=None, states=10, effect_by_state=None):
    """Organizations in counties, county c in state c % states; the top quarter of counties is
    exposed in every year. effect_by_state overrides the effect state by state."""
    rng = np.random.default_rng(seed)
    exposed_county = np.zeros(counties)
    exposed_county[: counties // 4] = 1.0
    rows = []
    for t in years:
        eff_t = effect if effect_by_year is None else effect_by_year.get(t, 0.0)
        for c in range(counties):
            state = f"S{c % states:02d}"
            eff = eff_t if effect_by_state is None else effect_by_state[state]
            for o in range(per):
                rows.append({"county_fips": f"{c:05d}", "tax_year": t, "ein": f"{c:04d}{o:05d}",
                             "state": state,
                             "high_need": exposed_county[c],
                             "y": eff * exposed_county[c] + rng.normal()})
    return pd.DataFrame(rows)


def test_bh_matches_hand_computation_and_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005]
    q, reject = benjamini_hochberg(p, alpha=0.03)
    assert q.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert reject.tolist() == [True, False, False, True]
    ref = multipletests(p, alpha=0.03, method="fdr_bh")
    assert q.tolist() == pytest.approx(ref[1].tolist())


def test_bh_leaves_missing_p_values_out_of_the_family():
    q, reject = benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(q[1]) and not reject[1]
    assert q[0] == pytest.approx(0.02)       # family of 2, not 3
    assert q[2] == pytest.approx(0.04)


def test_unit_permuter_keeps_one_value_per_county_year_and_the_exposed_count():
    df = _county_panel()
    perm = UnitPermuter(df, "high_need", unit="county_fips", within="tax_year")
    assert perm.n_units == 60 * 3
    draw = perm.draw(np.random.default_rng(1))
    check = df.assign(d=draw)
    assert check.groupby(["tax_year", "county_fips"])["d"].nunique().max() == 1
    per_year = check.groupby(["tax_year", "county_fips"])["d"].first().groupby(level=0).sum()
    assert per_year.tolist() == [15.0, 15.0, 15.0]


def _persistent_panel(counties=80, per=5, years=(2018, 2019, 2020, 2021, 2022), seed=0):
    """No exposure effect, but a county effect and an exposure that both persist over years;
    a few counties switch exposure in 2020."""
    rng = np.random.default_rng(seed)
    county_effect = rng.normal(0, 1.0, counties)
    rows = []
    for t in years:
        for c in range(counties):
            exposed = c < counties // 4 if not (t >= 2020 and c in (0, 1, 30, 31)) else c >= 30
            for o in range(per):
                rows.append({"county_fips": f"{c:05d}", "tax_year": t, "ein": f"{c:04d}{o:05d}",
                             "high_need": float(exposed),
                             "y": county_effect[c] + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_keep_paths_draws_whole_county_histories_and_keeps_yearly_counts():
    df = _persistent_panel()
    perm = UnitPermuter(df, "high_need", unit="county_fips", within="tax_year", keep_paths=True)
    assert perm.n_units == 80
    paths = lambda frame, col: (frame.groupby(["county_fips", "tax_year"])[col].first()  # noqa: E731
                                .unstack().apply(tuple, axis=1))
    real = paths(df, "high_need")
    draw = df.assign(d=perm.draw(np.random.default_rng(5)))
    drawn = paths(draw, "d")
    assert sorted(drawn.tolist()) == sorted(real.tolist())            # same histories, reassigned
    assert (drawn != real).any()
    yearly = lambda frame, col: frame.groupby(["tax_year", "county_fips"])[col].first().groupby(level=0).sum()  # noqa: E731
    assert yearly(draw, "d").tolist() == yearly(df, "high_need").tolist()


def test_keep_paths_only_swaps_counties_present_in_the_same_years():
    df = _persistent_panel()
    df = df[~((df["county_fips"] == "00079") & (df["tax_year"] == 2022))]   # one county misses a year
    perm = UnitPermuter(df, "high_need", unit="county_fips", within="tax_year", keep_paths=True)
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = df.assign(d=perm.draw(rng))
        lone = d[d["county_fips"] == "00079"]
        assert (lone["d"] == lone["high_need"]).all()       # alone in its stratum, never moves


def test_year_by_year_shuffle_understates_the_null_when_exposure_persists():
    df = _persistent_panel()
    y = df["y"].to_numpy()
    stat = lambda xv: difference_in_means(y, xv)  # noqa: E731
    yearly = permutation_test(stat, UnitPermuter(df, "high_need", unit="county_fips",
                                                 within="tax_year"), 199, seed=1)
    paths = permutation_test(stat, UnitPermuter(df, "high_need", unit="county_fips",
                                                within="tax_year", keep_paths=True), 199, seed=1)
    assert paths["null_sd"] > 1.8 * yearly["null_sd"]


def test_unit_permuter_rejects_exposure_that_varies_inside_a_unit():
    df = _county_panel()
    df.loc[0, "high_need"] = 1.0 - df.loc[0, "high_need"]
    with pytest.raises(ValueError):
        UnitPermuter(df, "high_need", unit="county_fips", within="tax_year")


def test_permutation_test_finds_a_real_gap():
    df = _county_panel(effect=-1.0)
    y = df["y"].to_numpy()
    perm = UnitPermuter(df, "high_need", unit="county_fips", within="tax_year")
    res = permutation_test(lambda x: difference_in_means(y, x), perm, n_permutations=199, seed=3)
    assert res["observed"] == pytest.approx(-1.0, abs=0.2)
    assert res["p_value"] == pytest.approx(1 / 200)


def test_permutation_p_is_one_when_there_is_no_gap_at_all():
    # Each exposed county is an exact copy of an unexposed one, so the observed gap is 0.
    rows = []
    for c in range(20):
        vals = np.arange(5, dtype=float) + (c // 2)
        for o, v in enumerate(vals):
            rows.append({"county_fips": f"{c:05d}", "tax_year": 2022, "high_need": float(c % 2), "y": v})
    df = pd.DataFrame(rows)
    y = df["y"].to_numpy()
    perm = UnitPermuter(df, "high_need", unit="county_fips", within="tax_year")
    res = permutation_test(lambda x: difference_in_means(y, x), perm, n_permutations=99, seed=0)
    assert res["observed"] == pytest.approx(0.0)
    assert res["p_value"] == 1.0


def test_effect_floor():
    assert passes_effect_floor(-0.25, 0.1)
    assert not passes_effect_floor(0.05, 0.1)
    assert not passes_effect_floor(float("nan"), 0.1)


def _fit(frame):
    return fixed_effects(frame, "y", "high_need", absorb=["tax_year"], cluster="county_fips")


def test_replication_holds_for_a_stable_effect():
    df = _county_panel(effect=-1.0)
    rep = held_out_year_replication(df, _fit, 2022, expected_sign=-1)
    assert rep["replicated"] is True
    assert rep["train_years"] == [2020, 2021]


def test_replication_fails_when_the_effect_flips_in_the_held_out_year():
    df = _county_panel(effect_by_year={2020: -1.0, 2021: -1.0, 2022: 1.0})
    rep = held_out_year_replication(df, _fit, 2022, expected_sign=-1)
    assert rep["train"]["effect"] < -0.5
    assert rep["test"]["effect"] > 0.5
    assert rep["same_sign"] is False
    assert rep["replicated"] is False


def test_group_folds_are_deterministic_for_a_seed_and_cover_every_state_once():
    states = pd.Series([f"S{i:02d}" for i in range(10)] * 3)
    folds = group_folds(states, 5, seed=7)
    assert folds == group_folds(states, 5, seed=7)
    assert folds != group_folds(states, 5, seed=8)
    assert sorted(s for f in folds for s in f) == sorted(states.unique())
    assert [len(f) for f in folds] == [2, 2, 2, 2, 2]
    with pytest.raises(ValueError):
        group_folds(states, 11, seed=7)


def test_state_replication_holds_for_a_stable_effect():
    rep = held_out_group_replication(_county_panel(effect=-1.0), _fit, n_folds=5, seed=3,
                                     expected_sign=-1)
    assert rep["replicated"] is True
    assert rep["folds_with_expected_sign"] == 5
    assert rep["pooled_effect"] == pytest.approx(-1.0, abs=0.25)
    assert rep["pooled_p_value"] < 0.05
    for fold in rep["folds"]:                 # no held-out state is in its training set
        assert not set(fold["held_out"]) & set(
            s for f in rep["folds"] if f is not fold for s in f["held_out"])


def test_state_replication_fails_when_a_few_states_carry_the_gap():
    # 3 of 10 states have a large gap; the other 7 have a small one the other way. The
    # pooled sample still shows a gap, but at least two folds hold only the 7 states.
    effect_by_state = {f"S{i:02d}": (-1.5 if i < 3 else 0.3) for i in range(10)}
    df = _county_panel(effect_by_state=effect_by_state)
    assert _fit(df).effect < 0
    rep = held_out_group_replication(df, _fit, n_folds=5, seed=3, expected_sign=-1)
    assert rep["folds_with_expected_sign"] <= 3
    assert rep["same_sign_in_every_fold"] is False
    assert rep["replicated"] is False


def test_state_replication_fails_when_there_is_no_effect():
    rep = held_out_group_replication(_county_panel(effect=0.0), _fit, n_folds=5, seed=3,
                                     expected_sign=-1)
    assert rep["replicated"] is False


@pytest.mark.parametrize("h1,h2,replicated,expected", [
    ((-0.5, -0.2, 0.001), (-0.4, -0.15, 0.001), True, "supported"),
    ((-0.5, -0.2, 0.001), (-0.01, -0.004, 0.6), True, "weak"),      # explained by the controls
    ((-0.5, -0.2, 0.001), (-0.4, -0.15, 0.001), False, "weak"),     # does not replicate
    ((-0.5, -0.2, 0.001), (0.4, 0.15, 0.001), True, "weak"),        # H2 flips sign
    ((-0.5, -0.2, 0.30), (-0.4, -0.15, 0.001), True, "unsupported"),  # H1 not significant
    ((-0.1, -0.04, 0.001), (-0.4, -0.15, 0.001), True, "unsupported"),  # H1 below the floor
    ((0.5, 0.2, 0.001), (0.4, 0.15, 0.001), True, "reversed"),
])
def test_verdict_rule(h1, h2, replicated, expected):
    out = decide_verdict(GateInput(*h1), GateInput(*h2), expected_sign=-1, alpha=0.05, floor=0.1,
                         replicated=replicated)
    assert out["verdict"] == expected
    assert out["reasons"]
