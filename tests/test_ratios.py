import math

import pandas as pd
import pytest

from theorylab.ratios import (RATIO_COLUMNS, add_ratios, add_revenue_growth,
                              revenue_concentration, safe_ratio)


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
