"""Per-year H1/H2 and the gap trend on synthetic panels with a planted answer."""

import copy

import numpy as np
import pandas as pd
import pytest

from theorylab.catalog import REPO_ROOT
from theorylab.spec import load_spec
from theorylab.theory_run import available_columns
from theorylab.trends import gap_trend, meta_trend, run_by_year

YEARS = (2018, 2019, 2020, 2021, 2022)


@pytest.fixture(scope="module")
def spec():
    s = copy.deepcopy(load_spec(REPO_ROOT / "specs" / "ratings_fairness.json", available_columns()))
    s["validator"]["n_permutations"] = 99
    return s


def _panel(gap_by_year, counties=100, per=10, seed=0):
    """Rating = 7 + 0.5 log revenue + gap_by_year[year] in high-need counties + noise."""
    rng = np.random.default_rng(seed)
    high = np.zeros(counties, dtype=bool)
    high[: counties // 4] = True
    county_effect = rng.normal(0, 0.3, counties)
    rows = []
    for t in YEARS:
        for c in range(counties):
            for o in range(per):
                log_rev = 13 + rng.normal()
                rows.append({
                    "ein": f"{c:04d}{o:05d}", "tax_year": t, "county_fips": f"{c + 1000:05d}",
                    "category": ["Health", "Education"][o % 2], "return_type": "990",
                    "has_county": "True", "has_need": "True",
                    "high_need": "True" if high[c] else "False",
                    "revenue_total": float(np.exp(log_rev)),
                    "standard_rating": 7 + 0.5 * (log_rev - 13) + gap_by_year[t] * high[c]
                    + county_effect[c] + rng.normal(0, 1.0),
                })
    return pd.DataFrame(rows)


def test_widening_gap_has_a_negative_trend_and_each_year_recovers_its_gap(spec):
    gaps = {2018: -0.2, 2019: -0.4, 2020: -0.6, 2021: -0.8, 2022: -1.0}
    report = run_by_year(_panel(gaps), spec)
    for year, gap in gaps.items():
        assert report["by_year"][str(year)]["H2"]["effect"] == pytest.approx(gap, abs=0.2)
    t = report["trend"]
    assert t["gap_change_per_year"] == pytest.approx(-0.2, abs=0.06)
    assert t["p_value"] < 0.01
    assert t["gap_at_center_year"] == pytest.approx(-0.6, abs=0.15)
    assert report["trend_meta_regression"]["slope_per_year"] == pytest.approx(-0.2, abs=0.06)
    assert "2022" in report["years_H2_holds"]


def test_constant_gap_has_no_trend(spec):
    report = run_by_year(_panel({t: -0.5 for t in YEARS}, seed=1), spec)
    t = report["trend"]
    assert abs(t["gap_change_per_year"]) < 0.06
    assert t["p_value"] > 0.05
    assert report["years_H1_holds"] == [str(y) for y in YEARS]


def test_no_gap_holds_in_no_year(spec):
    report = run_by_year(_panel({t: 0.0 for t in YEARS}, seed=2), spec)
    assert report["years_H1_holds"] == []
    assert report["years_H2_holds"] == []
    assert all(r["H1"]["q_value"] > 0.05 or abs(r["H1"]["standardized_effect"]) < 0.1
               for r in report["by_year"].values())


def test_per_year_h2_does_not_absorb_tax_year_and_uses_county_permutation(spec):
    report = run_by_year(_panel({t: -0.5 for t in YEARS}, seed=3), spec)
    r = report["by_year"]["2020"]
    assert r["H2"]["absorb"] == ["category"]
    assert r["units"] == 100 and r["exposed_units"] == 25
    assert report["settings"]["permutation_unit"] == "county_fips"


def test_meta_trend_needs_three_years():
    two = {"2021": {"H2": {"effect": -0.1, "std_error": 0.1}},
           "2022": {"H2": {"effect": -0.2, "std_error": 0.1}}}
    assert "error" in meta_trend(two)
