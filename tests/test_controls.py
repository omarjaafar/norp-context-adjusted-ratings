"""Negative and positive controls through the gate, on a small synthetic panel with no real gap."""

import copy

import numpy as np
import pandas as pd
import pytest

from theorylab.catalog import REPO_ROOT
from theorylab.controls import negative_control, population, positive_control, run_controls
from theorylab.spec import load_spec
from theorylab.theory_run import available_columns


@pytest.fixture(scope="module")
def spec():
    s = copy.deepcopy(load_spec(REPO_ROOT / "specs" / "ratings_fairness.json", available_columns()))
    s["validator"]["n_permutations"] = 99
    s["robustness"] = []
    return s


def _panel(counties=80, per=30, seed=0):
    """High-need = top quarter of counties, spread over 10 states; the rating depends on size only.

    30 organizations per county keep the noise in the H1 gap (about 0.025 SD) well below the
    0.1 SD floor. With 10 per county it is about 0.04 SD, and a shuffle crosses the floor by
    chance often enough to make the test depend on the seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in (2019, 2020, 2021, 2022):
        for c in range(counties):
            for o in range(per):
                log_rev = 13 + rng.normal()
                rows.append({
                    "ein": f"{c:04d}{o:05d}", "tax_year": t, "county_fips": f"{c + 1000:05d}",
                    "state": f"S{c % 10}", "category": ["Health", "Education"][o % 2],
                    "return_type": "990", "has_county": "True", "has_need": "True",
                    "high_need": "True" if c < counties // 4 else "False",
                    "revenue_total": float(np.exp(log_rev)),
                    "standard_rating": 7.0 + 0.6 * (log_rev - 13) + rng.normal(),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pop(spec):
    return population(_panel(), spec)[0]


def test_negative_control_shuffles_whole_counties_and_finds_nothing(spec, pop):
    out = negative_control(pop, spec, seeds=[1, 2, 3])
    assert out["n_runs"] == 3
    assert out["false_positives"] == 0
    assert out["verdict_counts"]["unsupported"] == 3
    assert [r["seed"] for r in out["runs"]] == [1, 2, 3]


def test_positive_control_above_the_floor_is_supported(spec, pop):
    out = positive_control(pop, spec, 0.5, "supported")
    assert out["verdict"] == "supported" and out["passed"]
    assert out["H1_standardized"] == pytest.approx(-0.5, abs=0.1)
    assert out["planted_shift"] < 0                       # expected_sign is -1


def test_positive_control_below_the_floor_is_unsupported(spec, pop):
    out = positive_control(pop, spec, 0.03, "unsupported")
    assert out["verdict"] == "unsupported" and out["passed"]


def test_run_controls_does_not_change_the_panel(spec):
    panel = _panel()
    before = panel.copy()
    report = run_controls(panel, spec, negative_seeds=[1], positive=((0.5, "supported"),))
    pd.testing.assert_frame_equal(panel, before)
    assert report["all_passed"] is True
