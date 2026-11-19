"""The heavy-tailed outcome rule (CP5): diagnosis, winsorizing, and a planted effect in a heavy-tailed ratio."""

import copy

import numpy as np
import pandas as pd
import pytest

from theorylab.outcomes import apply_outcome_rule, tail_ratio, validate_rule
from theorylab.spec import GATE_POLICY, lock_gate, validate_spec
from theorylab.sweep import cell_spec
from theorylab.theory_run import available_columns, run_spec, select_population

RULE = GATE_POLICY["outcome_rule"]


def test_the_rule_is_part_of_the_locked_gate():
    assert RULE == {"heavy_tail_ratio": 3.0, "winsorize": [0.01, 0.99]}
    spec = cell_spec("operating_margin", "poverty_rate", -1)
    assert spec["validator"]["outcome_rule"] == RULE
    loose = copy.deepcopy(spec)
    loose["validator"]["outcome_rule"] = {"heavy_tail_ratio": 1e9, "winsorize": [0, 1]}
    locked, overrides = lock_gate(loose)
    assert locked["validator"]["outcome_rule"] == RULE
    assert [o["field"] for o in overrides] == ["validator.outcome_rule"]


def test_rule_validation():
    assert validate_rule(RULE) == []
    assert validate_rule({"heavy_tail_ratio": 0.5, "winsorize": [0.01, 0.99]})
    assert validate_rule({"heavy_tail_ratio": 3, "winsorize": [0.6, 0.4]})
    spec = cell_spec("operating_margin", "poverty_rate", -1)
    spec["validator"]["outcome_rule"] = {"heavy_tail_ratio": 3, "winsorize": [2, 3]}
    assert any("winsorize" in e for e in validate_spec(spec, available_columns()))


def test_well_behaved_outcome_is_untouched_and_heavy_tailed_one_is_clipped():
    rng = np.random.default_rng(0)
    normal = pd.DataFrame({"y": rng.normal(size=5000)})
    out, log = apply_outcome_rule(normal, "y", RULE)
    assert not log["applied"] and out["y"].equals(normal["y"])
    assert 0.8 < log["sd_over_robust_sd"] < 1.2

    heavy = normal.copy()
    heavy.loc[:24, "y"] = 1e4  # 0.5% of rows, like a ratio with a tiny denominator
    assert tail_ratio(heavy["y"].to_numpy()) > 100
    out, log = apply_outcome_rule(heavy, "y", RULE)
    assert log["applied"] and log["rows_clipped"] >= 50
    assert out["y"].max() == pytest.approx(log["upper"]) and out["y"].max() < 10
    assert log["sd_over_robust_sd_after"] < RULE["heavy_tail_ratio"]
    assert heavy["y"].max() == 1e4  # the input is not mutated


def test_no_rule_means_no_change():
    df = pd.DataFrame({"y": [1.0, 2.0, 1e6]})
    out, log = apply_outcome_rule(df, "y", None)
    assert out is df and not log["applied"]


def _panel(effect: float, counties=60, per=20, seed=3):
    """Continuous county exposure; the outcome has `effect` robust SDs per exposure SD (sign -1)
    plus a few huge values, like a ratio with a tiny denominator."""
    rng = np.random.default_rng(seed)
    x_county = rng.normal(size=counties)
    rows = []
    for t in (2019, 2020, 2021, 2022):
        for c in range(counties):
            for o in range(per):
                y = rng.normal() - effect * x_county[c]
                if rng.random() < 0.005:
                    y += rng.choice([-1, 1]) * 1e4
                rows.append({"ein": f"{c:04d}{o:05d}", "tax_year": t, "county_fips": f"{c + 1000:05d}",
                             "state": f"S{c % 10}", "category": ["Health", "Education"][o % 2],
                             "return_type": "990", "has_county": "True", "has_need": "True",
                             "revenue_total": float(np.exp(13 + rng.normal())),
                             "operating_margin": y, "poverty_rate": float(x_county[c])})
    return pd.DataFrame(rows)


def _spec(with_rule: bool):
    spec = cell_spec("operating_margin", "poverty_rate", -1, n_permutations=99)
    spec["validator"]["n_permutations"] = 99
    if not with_rule:
        del spec["validator"]["outcome_rule"]
    return spec


def test_planted_effect_in_a_heavy_tailed_ratio_is_missed_without_the_rule_and_found_with_it():
    panel = _panel(effect=0.5)
    without = run_spec(panel, _spec(False), include_robustness=False)
    assert without["outcome_sd"] > 100  # a few rows set the SD
    assert without["verdict"] != "supported"  # 0.5 / SD is far below the 0.1 SD floor

    with_rule = run_spec(panel, _spec(True), include_robustness=False)
    assert with_rule["population"]["outcome_rule"]["applied"]
    assert with_rule["outcome_sd"] < 5
    assert with_rule["verdict"] == "supported"
    assert with_rule["H1"]["effect"] < -0.3


def test_null_heavy_tailed_ratio_stays_unsupported_under_the_rule():
    report = run_spec(_panel(effect=0.0, seed=4), _spec(True), include_robustness=False)
    assert report["verdict"] == "unsupported"


def test_population_log_records_the_rule_for_a_normal_outcome():
    spec = _spec(True)
    df = _panel(effect=0.0)
    df["operating_margin"] = np.random.default_rng(1).normal(size=len(df))
    df["log_revenue"] = np.log(df["revenue_total"])
    pop, log = select_population(df, spec)
    assert log["outcome_rule"]["applied"] is False
    assert len(pop) == len(df)
