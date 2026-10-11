import math

import pandas as pd
import pytest

from theorylab.ratios import (RATING_ANCHORS, RATIO_COLUMNS, add_ratios, add_revenue_growth,
                              functional_split_reported, rating_scores, revenue_concentration,
                              safe_ratio, standard_rating)


def _frame(**overrides):
    base = {
        "revenue_total": [1000.0],
        "expenses_total": [900.0],
        "contributions": [600.0],
        "program_revenue": [300.0],
        "net_assets_eoy": [500.0],
        "p09_total_expense": [900.0],
        "p09_program_expense": [720.0],
        "p09_management_expense": [135.0],
        "p09_fundraising_expense": [45.0],
        "total_assets_eoy": [2000.0],
        "total_liabilities_eoy": [500.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_safe_ratio_undefined_cases_are_nan():
    r = safe_ratio(pd.Series([1.0, 1.0, 1.0, None]), pd.Series([2.0, 0.0, -5.0, 4.0]))
    assert r.iloc[0] == 0.5
    assert r.iloc[1:].isna().all()


def test_safe_ratio_allows_negative_denominator_when_asked():
    r = safe_ratio(pd.Series([1.0]), pd.Series([-4.0]), positive_den=False)
    assert r.iloc[0] == -0.25


def test_add_ratios_values():
    out = add_ratios(_frame()).iloc[0]
    assert out["program_expense_ratio"] == pytest.approx(0.8)
    assert out["admin_expense_ratio"] == pytest.approx(0.15)
    assert out["fundraising_expense_ratio"] == pytest.approx(0.05)
    assert out["fundraising_efficiency"] == pytest.approx(45 / 600)
    assert out["operating_margin"] == pytest.approx(0.1)
    assert out["equity_ratio"] == pytest.approx(0.5)
    assert out["liabilities_to_assets"] == pytest.approx(0.25)
    # shares 0.6 / 0.3 / 0.1
    assert out["revenue_concentration"] == pytest.approx(0.36 + 0.09 + 0.01)
    assert set(RATIO_COLUMNS) <= set(out.index)


def test_add_ratios_zero_expenses_and_revenue():
    out = add_ratios(_frame(p09_total_expense=[0.0], revenue_total=[0.0],
                            contributions=[0.0], program_revenue=[0.0])).iloc[0]
    assert math.isnan(out["program_expense_ratio"])
    assert math.isnan(out["operating_margin"])
    assert math.isnan(out["equity_ratio"])
    assert math.isnan(out["fundraising_efficiency"])
    assert math.isnan(out["revenue_concentration"])


def test_add_ratios_missing_column_raises():
    with pytest.raises(KeyError):
        add_ratios(_frame().drop(columns=["total_assets_eoy"]))


def test_revenue_concentration_bounds():
    hhi = revenue_concentration(pd.Series([100.0, 50.0, 100.0]),
                                pd.Series([0.0, 50.0, None]),
                                pd.Series([100.0, 100.0, 100.0]))
    assert hhi.iloc[0] == pytest.approx(1.0)
    assert hhi.iloc[1] == pytest.approx(0.5)
    assert math.isnan(hhi.iloc[2])


def test_revenue_concentration_negative_other_revenue_is_floored():
    # investment losses make total < contributions + program revenue
    hhi = revenue_concentration(pd.Series([60.0]), pd.Series([60.0]), pd.Series([100.0]))
    assert hhi.iloc[0] == pytest.approx(0.5)


def test_revenue_growth_requires_consecutive_years():
    df = pd.DataFrame({
        "ein": ["000000001", "000000001", "000000001", "000000002", "000000002"],
        "tax_year": pd.array([2021, 2018, 2019, 2019, 2020], dtype="Int64"),
        "revenue_total": [300.0, 100.0, 150.0, 0.0, 50.0],
    })
    out = add_revenue_growth(df)
    assert list(out.index) == list(df.index)
    growth = out["revenue_growth"]
    assert math.isnan(growth.iloc[0])           # 2021 follows 2019: gap
    assert math.isnan(growth.iloc[1])           # first year
    assert growth.iloc[2] == pytest.approx(0.5)  # 2019 vs 2018
    assert math.isnan(growth.iloc[4])           # previous revenue 0


# --- audit of the TA formulas: zero, negative and missing inputs ---

def test_unreported_functional_split_is_missing_not_zero():
    # Filers that are not 501(c)(3)/(c)(4) may leave Part IX columns B-D empty or zero.
    out = add_ratios(_frame(p09_program_expense=[0.0], p09_management_expense=[0.0],
                            p09_fundraising_expense=[0.0])).iloc[0]
    assert math.isnan(out["program_expense_ratio"])      # the TA formula gave 0.0
    assert math.isnan(out["admin_expense_ratio"])
    assert math.isnan(out["fundraising_expense_ratio"])
    assert math.isnan(out["fundraising_efficiency"])     # 0 would read as "free fundraising"
    assert out["liabilities_to_assets"] == pytest.approx(0.25)   # Part X is unaffected


def test_blank_fundraising_counts_as_zero_when_the_rest_adds_up():
    out = add_ratios(_frame(p09_program_expense=[765.0], p09_management_expense=[135.0],
                            p09_fundraising_expense=[None])).iloc[0]
    assert out["program_expense_ratio"] == pytest.approx(0.85)
    assert out["fundraising_expense_ratio"] == 0.0
    assert out["fundraising_efficiency"] == 0.0


def test_functional_split_tolerance():
    ok = functional_split_reported(
        total=pd.Series([900.0, 900.0, 900.0, 0.0, None, -900.0]),
        program=pd.Series([720.0, 720.0, None, 0.0, 720.0, -720.0]),
        management=pd.Series([135.0, 135.0, None, 0.0, 135.0, -135.0]),
        fundraising=pd.Series([40.0, 30.0, None, 0.0, 45.0, -45.0]))
    # off by 5 of 900 (0.6%): fine. off by 15 (1.7%): not. all blank, zero, missing or
    # negative totals: not.
    assert ok.tolist() == [True, False, False, False, False, False]


def test_impossible_values_are_missing():
    # management is negative, so program exceeds total even though the three add up
    out = add_ratios(_frame(p09_program_expense=[1000.0], p09_management_expense=[-145.0],
                            p09_fundraising_expense=[45.0],
                            total_liabilities_eoy=[-500.0])).iloc[0]
    assert math.isnan(out["program_expense_ratio"])
    assert math.isnan(out["admin_expense_ratio"])
    assert out["fundraising_expense_ratio"] == pytest.approx(0.05)
    assert math.isnan(out["liabilities_to_assets"])

    out = add_ratios(_frame(p09_program_expense=[765.0], p09_management_expense=[180.0],
                            p09_fundraising_expense=[-45.0])).iloc[0]
    assert math.isnan(out["fundraising_expense_ratio"])
    assert math.isnan(out["fundraising_efficiency"])


def test_real_negatives_are_kept():
    # a deficit year and negative net assets are real, not errors
    out = add_ratios(_frame(expenses_total=[1200.0], net_assets_eoy=[-300.0],
                            total_liabilities_eoy=[2600.0])).iloc[0]
    assert out["operating_margin"] == pytest.approx(-0.2)
    assert out["equity_ratio"] == pytest.approx(-0.3)
    assert out["liabilities_to_assets"] == pytest.approx(1.3)   # insolvent, above 1 is allowed


def test_missing_and_text_inputs():
    out = add_ratios(_frame(total_assets_eoy=[None], net_assets_eoy=[None],
                            contributions=["600"], p09_total_expense=["900"])).iloc[0]
    assert math.isnan(out["liabilities_to_assets"])
    assert math.isnan(out["equity_ratio"])
    assert out["fundraising_efficiency"] == pytest.approx(45 / 600)
    assert out["program_expense_ratio"] == pytest.approx(0.8)

    out = add_ratios(_frame(contributions=[0.0])).iloc[0]
    assert math.isnan(out["fundraising_efficiency"])            # nothing raised: undefined
    out = add_ratios(_frame(contributions=[-10.0])).iloc[0]
    assert math.isnan(out["fundraising_efficiency"])
    # a negative source is floored at 0: shares are 0 / 0.3 / 0.7
    assert out["revenue_concentration"] == pytest.approx(0.09 + 0.49)


def test_revenue_growth_edge_cases():
    df = pd.DataFrame({
        "ein": ["1", "1", "2", "2", "3", "3"],
        "tax_year": pd.array([2019, 2020, 2019, 2020, 2019, 2020], dtype="Int64"),
        "revenue_total": [-50.0, 100.0, None, 100.0, 200.0, -100.0],
    })
    growth = add_revenue_growth(df)["revenue_growth"]
    assert math.isnan(growth.iloc[1])               # previous revenue negative
    assert math.isnan(growth.iloc[3])               # previous revenue missing
    assert growth.iloc[5] == pytest.approx(-1.5)    # falling into negative revenue is real


# --- standard rating (docs/ratings_theory.md) ---

def _ratios(**overrides):
    base = {"program_expense_ratio": [0.75], "admin_expense_ratio": [0.25],
            "fundraising_efficiency": [0.225], "liabilities_to_assets": [0.5]}
    base.update(overrides)
    return pd.DataFrame(base)


def test_rating_scores_are_linear_between_anchors():
    s = rating_scores(_ratios()).iloc[0]
    assert s["score_program_expense_ratio"] == pytest.approx(5.0)
    assert s["score_admin_expense_ratio"] == pytest.approx(5.0)
    assert s["score_fundraising_efficiency"] == pytest.approx(5.0)
    assert s["score_liabilities_to_assets"] == pytest.approx(5.0)


@pytest.mark.parametrize("ratio", list(RATING_ANCHORS))
def test_rating_scores_stop_at_the_anchors(ratio):
    zero, full = RATING_ANCHORS[ratio]
    step = 0.2 if full > zero else -0.2
    values = [zero, full, zero - step, full + step]
    df = pd.concat([_ratios(**{ratio: [v]}) for v in values], ignore_index=True)
    assert rating_scores(df)[f"score_{ratio}"].tolist() == [0.0, 10.0, 0.0, 10.0]


def test_standard_rating_is_the_mean_of_the_scores():
    df = _ratios(program_expense_ratio=[0.85], admin_expense_ratio=[0.15],
                 fundraising_efficiency=[0.35], liabilities_to_assets=[0.5])
    assert standard_rating(df).iloc[0] == pytest.approx((10 + 10 + 0 + 5) / 4)


def test_standard_rating_needs_three_of_four_ratios():
    three = _ratios(fundraising_efficiency=[None], program_expense_ratio=[0.85])
    assert standard_rating(three).iloc[0] == pytest.approx((10 + 5 + 5) / 3)
    two = _ratios(fundraising_efficiency=[None], liabilities_to_assets=[None])
    assert math.isnan(standard_rating(two).iloc[0])
    assert standard_rating(two, min_components=2).iloc[0] == pytest.approx(5.0)


def test_standard_rating_accepts_other_anchors():
    strict = {**RATING_ANCHORS, "program_expense_ratio": (0.75, 0.95)}
    assert standard_rating(_ratios(), anchors=strict).iloc[0] == pytest.approx((0 + 5 + 5 + 5) / 4)


def test_add_ratios_includes_the_rating():
    out = add_ratios(_frame()).iloc[0]
    # program 0.80 -> 7.5, admin 0.15 -> 10, fundraising 0.075 -> 10, liabilities 0.25 -> 7.5
    assert out["standard_rating"] == pytest.approx(8.75)
    assert out["rating_components"] == 4

    no_split = add_ratios(_frame(p09_program_expense=[0.0], p09_management_expense=[0.0],
                                 p09_fundraising_expense=[0.0])).iloc[0]
    assert math.isnan(no_split["standard_rating"])   # only the balance sheet ratio is known
    assert no_split["rating_components"] == 1
