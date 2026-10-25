"""Ratings theory end to end on a synthetic panel with a planted answer, plus spec and check helpers."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from theorylab.catalog import REPO_ROOT
from theorylab.checks import assumption_5_check, part9_coverage
from theorylab.sensitivity import ANCHOR_VARIANTS, run_anchor_sensitivity
from theorylab.spec import SpecError, load_spec, spec_columns, validate_spec
from theorylab.theory_run import available_columns, run_spec, select_population

SPEC_PATH = REPO_ROOT / "specs" / "ratings_fairness.json"


@pytest.fixture(scope="module")
def spec():
    s = load_spec(SPEC_PATH, available_columns())
    s = copy.deepcopy(s)
    s["validator"]["n_permutations"] = 99     # keep the tests fast; min p = 0.01
    s["robustness"] = []
    return s


def _panel(penalty=-1.0, size_gap=0.0, counties=80, per=12, seed=0):
    """Panel-shaped frame. High-need = top quarter of counties, spread over 10 states; the rating
    is built from the four ratios so anchor variants re-score it, then shifted by `penalty` in
    high-need counties."""
    rng = np.random.default_rng(seed)
    high = np.zeros(counties, dtype=bool)
    high[: counties // 4] = True
    rows = []
    for t in (2018, 2019, 2020, 2021, 2022):
        for c in range(counties):
            for o in range(per):
                log_rev = 13 + size_gap * high[c] + rng.normal()
                rows.append({
                    "ein": f"{c:04d}{o:05d}", "tax_year": t, "county_fips": f"{c + 1000:05d}",
                    "state": f"S{c % 10}", "category": ["Health", "Education", "Human Services"][o % 3],
                    "return_type": "990" if o < per - 1 else "990EZ",
                    "has_county": "True", "has_need": "True",
                    "high_need": "True" if high[c] else "False",
                    "revenue_total": float(np.exp(log_rev)),
                    "program_expense_ratio": float(np.clip(0.78 + 0.02 * (log_rev - 13)
                                                           + rng.normal(0, 0.05), 0, 1)),
                    "admin_expense_ratio": float(np.clip(0.15 + rng.normal(0, 0.04), 0, 1)),
                    "fundraising_efficiency": float(abs(0.12 + rng.normal(0, 0.05))),
                    "liabilities_to_assets": float(abs(0.3 + rng.normal(0, 0.15))),
                    "_shift": penalty * high[c],
                })
    df = pd.DataFrame(rows)
    df["standard_rating"] = 7.0 + 0.6 * (np.log(df["revenue_total"]) - 13) + df.pop("_shift") \
        + rng.normal(0, 1.0, len(df))
    return df


def test_committed_spec_is_valid_and_uses_panel_columns():
    s = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert validate_spec(s, available_columns()) == []
    assert {"standard_rating", "high_need", "county_fips", "category", "log_revenue"} <= spec_columns(s)


def test_spec_errors_are_reported_not_repaired(tmp_path):
    s = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    s["hypotheses"][1]["controls"] = ["made_up_column"]
    s["expected_sign"] = 0
    s["validator"]["holdout"] = {"by": "state", "folds": 1}
    errors = validate_spec(s, available_columns())
    assert any("made_up_column" in e for e in errors)
    assert any("expected_sign" in e for e in errors)
    assert any("holdout.folds" in e for e in errors)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec(path, available_columns())


def test_population_keeps_full_990_filers_with_county_and_need(spec):
    df = _panel()
    df.loc[0, "has_need"] = "False"
    df.loc[1, "standard_rating"] = np.nan
    df["log_revenue"] = np.log(df["revenue_total"])
    pop, log = select_population(df, spec)
    assert (pop["return_type"] == "990").all()
    assert log["rows_out"] == int((df["return_type"] == "990").sum()) - 2
    assert log["exposed_units_by_year"] == {str(t): 20 for t in (2018, 2019, 2020, 2021, 2022)}


def test_planted_penalty_is_supported(spec):
    report = run_spec(_panel(penalty=-1.0), spec)
    assert report["verdict"] == "supported"
    assert report["H1"]["effect"] == pytest.approx(-1.0, abs=0.25)
    assert report["H2"]["fit"]["effect"] == pytest.approx(-1.0, abs=0.25)
    assert report["H2"]["permutation"]["observed"] == pytest.approx(report["H2"]["fit"]["effect"], rel=1e-6)
    assert report["replication"]["replicated"] is True
    assert report["replication"]["by"] == "state"
    assert report["replication_year"]["holdout_year"] == 2022     # still reported, for information
    assert report["population"]["after_return_type"] < report["population"]["rows_in"]


def test_no_penalty_is_unsupported(spec):
    assert run_spec(_panel(penalty=0.0), spec)["verdict"] == "unsupported"


def test_higher_ratings_in_high_need_counties_is_reversed(spec):
    assert run_spec(_panel(penalty=1.0), spec)["verdict"] == "reversed"


def test_gap_explained_by_size_is_weak(spec):
    # High-need organizations are smaller, and size alone drives the rating.
    report = run_spec(_panel(penalty=0.0, size_gap=-1.5), spec)
    assert report["H1"]["effect"] < -0.5
    assert abs(report["H2"]["fit"]["standardized_effect"]) < 0.1
    assert report["verdict"] == "weak"


def test_anchor_sensitivity_reports_every_variant(spec):
    variants = {k: ANCHOR_VARIANTS[k] for k in ("base", "program_full_at_90pct", "four_ratios_required")}
    out = run_anchor_sensitivity(_panel(penalty=-1.0), spec, variants=variants, n_permutations=99)
    assert set(out["variants"]) == set(variants)
    assert out["base_verdict"] == "supported"
    assert out["variants"]["program_full_at_90pct"]["anchors"]["program_expense_ratio"] == [0.65, 0.9]
    # the planted penalty is added after the anchors are applied in _panel, so re-scoring removes it
    assert out["variants"]["program_full_at_90pct"]["H1_effect"] > -0.5
    assert out["verdict_stable"] is False


def test_anchor_variants_only_move_full_marks_anchors():
    base = ANCHOR_VARIANTS["base"][0]
    for name, (anchors, _) in ANCHOR_VARIANTS.items():
        for ratio, (zero, _full) in anchors.items():
            assert zero == base[ratio][0], name


def test_part9_coverage_counts_return_types():
    p01 = pd.DataFrame({"ein": ["1", "2", "3", "4"], "tax_year": [2018] * 4,
                        "return_type": ["990", "990", "990EZ", "990PF"]})
    p01["tax_year"] = p01["tax_year"].astype("Int64")
    p09 = pd.DataFrame({"ein": ["1", "3", "4", "5"], "tax_year": [2018] * 4,
                        "return_type": ["990", "990EZ", "990PF", "990"],
                        "p09_total_expense": [100.0, 50.0, 80.0, 10.0],
                        "p09_program_expense": [80.0, 0.0, 0.0, 10.0],
                        "p09_management_expense": [15.0, 0.0, 0.0, 0.0],
                        "p09_fundraising_expense": [5.0, 0.0, 0.0, 0.0]})
    p09["tax_year"] = p09["tax_year"].astype("Int64")
    out = part9_coverage(p01, p09)
    y = out["by_tax_year"]["2018"]
    assert y["panel_rows_with_p09_by_p01_return_type"] == {"990": 1, "990EZ": 1, "990PF": 1}
    assert y["p09_rows_without_p01"] == 1
    assert y["full_990_without_p09"] == 1
    assert y["usable_split_share_by_p01_return_type"]["990"] == 1.0
    assert y["usable_split_share_by_p01_return_type"]["990EZ"] == 0.0
    assert out["panel_rows_with_p09_not_full_990"] == 2


def test_assumption_5_status():
    ok = assumption_5_check({"fit_seconds": 120.0, "converged": True}, 300_000, 2_800)
    slow = assumption_5_check({"fit_seconds": 900.0, "converged": True}, 300_000, 2_800)
    bad = assumption_5_check({"fit_seconds": 900.0, "converged": False}, 300_000, 2_800)
    assert ok["status"] == "Holding (measured)" and ok["holds"]
    assert slow["status"] == "Weakened (measured)" and not slow["holds"]
    assert bad["status"] == "Broken (measured)"
