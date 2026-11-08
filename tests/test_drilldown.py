"""Drill-down finds a planted stratum and no other (CP4 V5)."""

import copy

import numpy as np
import pandas as pd
import pytest

from theorylab.catalog import REPO_ROOT
from theorylab.drilldown import drill_down
from theorylab.spec import load_spec
from theorylab.theory_run import available_columns


@pytest.fixture(scope="module")
def spec():
    s = copy.deepcopy(load_spec(REPO_ROOT / "specs" / "ratings_fairness.json", available_columns()))
    s["validator"]["n_permutations"] = 99   # min p 0.01; 4 strata -> min q 0.04
    return s


def _panel(planted_state="S0", penalty=-1.5, states=4, counties=40, per=4, seed=0):
    """4 states x 40 counties; a quarter of each state's counties high-need; the penalty
    applies only in high-need counties of planted_state (None: everywhere)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(states):
        for c in range(counties):
            high = c < counties // 4
            u = rng.normal(0, 0.3)
            for t in (2018, 2019):
                for o in range(per):
                    shift = penalty * high * (planted_state is None or f"S{s}" == planted_state)
                    rows.append({"ein": f"{s}{c:03d}{o}", "tax_year": t,
                                 "county_fips": f"{s}{c:04d}", "state": f"S{s}",
                                 "category": ["Health", "Education"][o % 2], "return_type": "990",
                                 "has_county": "True", "has_need": "True",
                                 "high_need": "True" if high else "False",
                                 "revenue_total": float(np.exp(13 + rng.normal())),
                                 "standard_rating": 7.0 + shift + u + rng.normal(0, 1.0)})
    return pd.DataFrame(rows)


def test_drill_down_by_state_finds_the_planted_state_only(spec):
    out = drill_down(_panel("S0"), spec, {"verdict": "weak"}, "state")
    assert out["family_size"] == 4
    assert out["survivors"] == ["S0"]
    s0 = next(s for s in out["strata"] if s["level"] == "S0")
    assert s0["same_sign"] and s0["q_value"] < 0.05


def test_drill_down_on_a_national_effect_gives_the_expected_sign_everywhere(spec):
    out = drill_down(_panel(None), spec, {"verdict": "weak"}, "state")
    assert all(s["same_sign"] for s in out["strata"])


def test_drill_down_skips_small_strata_and_non_survivors(spec):
    out = drill_down(_panel("S0"), spec, {"verdict": "weak"}, "state", min_units=41)
    assert out["family_size"] == 0 and len(out["skipped"]) == 4
    assert "skipped" in drill_down(_panel("S0"), spec, {"verdict": "unsupported"}, "state")
