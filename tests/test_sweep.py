"""The CP4 ratio x need sweep: grid, signs, locked gate, nulls reported."""

import numpy as np
import pandas as pd

from theorylab.ratios import RATING_ANCHORS
from theorylab.spec import GATE_POLICY
from theorylab.sweep import RATING_SIGNS, cell_spec, main_grid, run_sweep, ta_grid


def test_signs_follow_the_rating_anchors():
    assert set(RATING_SIGNS) == set(RATING_ANCHORS)
    assert RATING_SIGNS == {"program_expense_ratio": -1, "admin_expense_ratio": 1,
                            "fundraising_efficiency": 1, "liabilities_to_assets": 1}


def test_main_grid_is_the_ta_suggestion_plus_chr_columns():
    assert len(main_grid(["need_index"])) == 8
    grid = main_grid(["need_index", "chr_uninsured", "chr_some_college"])
    assert len(grid) == 16 and {n for _, n, _ in grid} >= {"chr_uninsured", "chr_some_college"}
    assert len(ta_grid()) == 40


def test_fixed_cell_runs_under_the_locked_gate_with_state_effects():
    spec = cell_spec("admin_expense_ratio", "need_index", 1, "fixed")
    assert spec["validator"] == GATE_POLICY
    assert "state" in spec["hypotheses"][1]["absorb"]
    ta = cell_spec("admin_expense_ratio", "need_index", 1, "ta")
    assert "permutation" not in ta["validator"] and "state" not in ta["hypotheses"][1]["absorb"]


def _panel(counties=60, per=8, slope=0.0, seed=0):
    rng = np.random.default_rng(seed)
    need = rng.normal(size=counties)
    burden = rng.normal(size=counties)
    rows = []
    for t in (2020, 2021, 2022):
        for c in range(counties):
            for o in range(per):
                rows.append({"ein": f"{c:03d}{o:03d}", "tax_year": t, "county_fips": f"{c + 1000:05d}",
                             "state": f"S{c % 6}", "category": ["A", "B"][o % 2], "return_type": "990",
                             "has_county": "True", "has_need": "True", "need_index": need[c],
                             "avg_housing_burden": burden[c],
                             "revenue_total": float(np.exp(13 + rng.normal())),
                             "admin_expense_ratio": 0.15 + slope * need[c] + rng.normal(0, 0.04),
                             "program_expense_ratio": 0.8 + rng.normal(0, 0.05),
                             "fundraising_efficiency": abs(0.12 + rng.normal(0, 0.05)),
                             "liabilities_to_assets": abs(0.3 + rng.normal(0, 0.15))})
    return pd.DataFrame(rows)


def test_sweep_reports_every_cell_and_finds_a_planted_relation():
    out = run_sweep(_panel(slope=0.03), scheme="fixed", n_permutations=199)  # min p 0.005: can clear BH over 16
    assert out["n_permutations"] == 199
    assert out["summary"]["cells"] == 8 and len(out["cells"]) == 8 and not out["errors"]
    planted = next(c for c in out["cells"] if c["cell"] == "sweep_admin_expense_ratio_x_need_index")
    assert planted["h1_r"] > 0.3 and planted["significant_after_family_bh"]
    nulls = [c for c in out["cells"] if "admin_expense_ratio_x_need_index" not in c["cell"]]
    assert all(c["verdict"] != "supported" for c in nulls)
