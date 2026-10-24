"""Executor methods on synthetic data with a known answer."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from theorylab.executor import (Residualizer, correlation, demean, fixed_effects, fwl_coefficient,
                                lagged_outcome, multilevel, to_float, two_stage_group_means)


def _grouped(n_groups=60, per_group=40, beta=2.0, seed=1):
    """y = beta * x + group effect + noise, with x correlated with the group effect."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    a = rng.normal(0, 2, n_groups)
    x = 0.8 * a[g] + rng.normal(0, 1, len(g))
    y = beta * x + 3 * a[g] + rng.normal(0, 1, len(g))
    return pd.DataFrame({"y": y, "x": x, "g": [f"{i:05d}" for i in g]})


def test_to_float_reads_panel_booleans():
    s = pd.Series(["True", "False", None, "true"])
    assert to_float(s).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(to_float(s).iloc[2])
    assert to_float(pd.Series([True, False])).tolist() == [1.0, 0.0]
    assert to_float(pd.Series([1, 2])).dtype == "float64"


def test_correlation_recovers_r():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 0.5 * x + rng.normal(scale=np.sqrt(0.75), size=5000)
    r = correlation(pd.DataFrame({"x": x, "y": y}), "y", "x")
    assert r.effect == pytest.approx(0.5, abs=0.04)
    assert r.p_value < 1e-10
    assert r.n_obs == 5000


def test_spearman_is_one_for_a_monotone_transform():
    x = np.linspace(1, 10, 50)
    r = correlation(pd.DataFrame({"x": x, "y": np.exp(x)}), "y", "x", method="spearman")
    assert r.effect == pytest.approx(1.0)


def test_demean_one_factor_is_exact():
    codes = [np.array([0, 0, 1, 1, 1])]
    out, sweeps = demean(np.array([1.0, 3.0, 2.0, 4.0, 6.0]), codes)
    assert out.tolist() == pytest.approx([-1, 1, -2, 0, 2])
    assert sweeps == 1


def test_fixed_effects_removes_group_confounding():
    df = _grouped()
    naive = fixed_effects(df, "y", "x")
    fe = fixed_effects(df, "y", "x", absorb=["g"], cluster="g")
    assert abs(naive.effect - 2.0) > 0.5            # the group effect biases plain OLS
    assert fe.effect == pytest.approx(2.0, abs=0.08)
    assert fe.p_value < 1e-10
    assert fe.n_groups == 60
    assert fe.details["absorbed_levels"] == {"g": 60}


def test_fixed_effects_matches_dummy_ols_with_two_factors_and_clusters():
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame({
        "year": rng.integers(2018, 2023, n),
        "sector": rng.choice(list("ABCD"), n),
        "county": rng.integers(0, 30, n),
        "x": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    df["y"] = (1.5 * df["x"] - 0.7 * df["c"] + df["year"].map(lambda t: 0.3 * (t - 2018))
               + df["sector"].map({"A": 0, "B": 1, "C": -1, "D": 2}) + rng.normal(size=n))
    fe = fixed_effects(df, "y", "x", controls=["c"], absorb=["year", "sector"], cluster="county")

    dummies = pd.get_dummies(df[["year", "sector"]].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[["x", "c"]], dummies], axis=1))
    ref = sm.OLS(df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": df["county"]})
    assert fe.effect == pytest.approx(ref.params["x"], rel=1e-7)
    assert fe.details["controls"]["c"]["coef"] == pytest.approx(ref.params["c"], rel=1e-7)
    assert fe.std_error == pytest.approx(ref.bse["x"], rel=1e-6)


def test_fixed_effects_classical_se_matches_ols():
    df = _grouped(n_groups=10, per_group=30)
    fe = fixed_effects(df, "y", "x", absorb=["g"])
    X = sm.add_constant(pd.concat([df[["x"]], pd.get_dummies(df["g"], drop_first=True, dtype=float)],
                                  axis=1))
    ref = sm.OLS(df["y"], X).fit()
    assert fe.effect == pytest.approx(ref.params["x"], rel=1e-8)
    assert fe.std_error == pytest.approx(ref.bse["x"], rel=1e-6)
    assert fe.p_value == pytest.approx(ref.pvalues["x"], rel=1e-4)


def test_fixed_effects_drops_incomplete_rows_and_rejects_constant_exposure():
    df = _grouped(n_groups=5, per_group=10)
    df.loc[0, "x"] = np.nan
    assert fixed_effects(df, "y", "x").n_obs == 49
    df["const"] = 1.0
    with pytest.raises(ValueError):
        fixed_effects(df, "y", "const", absorb=["g"])


def test_multilevel_recovers_slope_and_group_variance():
    rng = np.random.default_rng(3)
    groups, per = 80, 25
    g = np.repeat(np.arange(groups), per)
    u = rng.normal(0, 1.0, groups)
    x = rng.normal(size=len(g))
    y = 1.5 * x + u[g] + rng.normal(0, 1.0, len(g))
    df = pd.DataFrame({"y": y, "x": x, "county": g.astype(str)})
    r = multilevel(df, "y", "x", group="county")
    assert r.effect == pytest.approx(1.5, abs=0.08)
    assert r.details["converged"] is True
    assert r.details["random_intercept_variance"] == pytest.approx(1.0, abs=0.45)
    assert r.details["residual_variance"] == pytest.approx(1.0, abs=0.15)
    assert r.n_groups == groups
    assert r.details["fit_seconds"] >= 0


def test_two_stage_group_means_recovers_a_group_level_effect():
    rng = np.random.default_rng(5)
    counties, per = 400, 15
    g = np.repeat(np.arange(counties), per)
    exposed = (rng.random(counties) < 0.25).astype(float)
    size = rng.normal(size=len(g))
    y = -1.0 * exposed[g] + 0.5 * size + rng.normal(0, 0.3, counties)[g] + rng.normal(size=len(g))
    df = pd.DataFrame({"y": y, "x": exposed[g], "size": size, "county": g})
    r = two_stage_group_means(df, "y", "x", controls=["size"], groups=["county"])
    assert r.effect == pytest.approx(-1.0, abs=0.2)
    assert r.n_groups == counties
    assert r.p_value < 1e-6


def test_lagged_outcome_uses_only_the_filing_one_year_earlier():
    rng = np.random.default_rng(11)
    rows = []
    for e in range(400):
        x_prev = None
        for t in (2018, 2019, 2020, 2021, 2022):
            x = rng.normal()
            y = (0.5 * x_prev if x_prev is not None else 0.0) + rng.normal(scale=0.5)
            rows.append({"ein": f"{e:09d}", "tax_year": t, "x": x, "y": y})
            x_prev = x
    df = pd.DataFrame(rows)
    r = lagged_outcome(df, "y", "x", include_lagged_outcome=False)
    assert r.effect == pytest.approx(0.5, abs=0.06)
    assert r.details["rows_with_lagged_filing"] == 400 * 4

    gap = df[~((df["ein"] == "000000000") & (df["tax_year"] == 2020))]
    r2 = lagged_outcome(gap, "y", "x")
    # dropping 2020 for one EIN removes its 2020 and 2021 lagged rows
    assert r2.details["rows_with_lagged_filing"] == 400 * 4 - 2


def test_residualizer_matches_fixed_effects_coefficient():
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"year": rng.integers(0, 4, n), "cat": rng.integers(0, 6, n),
                       "c": rng.normal(size=n), "x": rng.normal(size=n)})
    df["y"] = 0.8 * df["x"] + df["c"] + df["year"] + rng.normal(size=n)
    fe = fixed_effects(df, "y", "x", controls=["c"], absorb=["year", "cat"])
    res = Residualizer.from_panel(df, controls=["c"], absorb=["year", "cat"])
    assert fwl_coefficient(res(df["y"]), res(df["x"])) == pytest.approx(fe.effect, rel=1e-6)
