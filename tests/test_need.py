import math

import pandas as pd
import pytest

from theorylab.catalog import load_catalog
from theorylab.need import (NEED_COLUMNS, NORP_NEED_COLUMNS, acs_rows_to_frame, acs_url,
                            add_norp_need, aggregate_tract_need, load_norp_tracts, to_need_table)


@pytest.fixture
def acs_source():
    return load_catalog()["acs_county"]


@pytest.fixture
def norp_source():
    return load_catalog()["norp_disadvantaged"]


def _tracts(rows):
    cols = ["Geoid", "County Fips", "Population", "Avg Housing Burden", "Food Desert Pct"]
    return pd.DataFrame(rows, columns=cols)


def _rows():
    header = ["NAME", "B01003_001E", "B19013_001E", "B17001_001E", "B17001_002E",
              "B23025_003E", "B23025_005E", "state", "county"]
    return [
        header,
        # name, pop, income, pov universe, below pov, labor force, unemployed, state, county
        ["A County, Georgia", "1000", "30000", "1000", "300", "500", "50", "13", "001"],
        ["B County, Georgia", "1000", "60000", "1000", "100", "500", "20", "13", "003"],
        ["C County, Georgia", "1000", "90000", "1000", "50", "500", "10", "13", "005"],
        ["D County, Georgia", "1000", "-666666666", "0", "0", "500", "15", "13", "007"],
    ]


def test_acs_url_requests_contracted_variables(acs_source):
    url = acs_url(acs_source, 2021)
    assert url.startswith("https://api.census.gov/data/2021/acs/acs5?")
    assert "for=county:*" in url
    for var in acs_source.fields:
        assert var in url


def test_sentinels_become_missing(acs_source):
    df = acs_rows_to_frame(_rows(), acs_source, 2021)
    assert math.isnan(df.loc[3, "B19013_001E"])
    assert df["acs_year"].eq(2021).all()


def test_need_table_rates_and_index(acs_source):
    need = to_need_table(acs_rows_to_frame(_rows(), acs_source, 2021), acs_source)
    assert list(need.columns) == list(NEED_COLUMNS)
    need = need.set_index("county_fips")
    assert need.loc["13001", "poverty_rate"] == pytest.approx(0.3)
    assert need.loc["13001", "unemployment_rate"] == pytest.approx(0.1)
    assert math.isnan(need.loc["13007", "poverty_rate"])     # universe 0
    assert math.isnan(need.loc["13007", "need_index"])       # income missing
    # poorest county has the highest need index, richest the lowest
    assert need.loc["13001", "need_index"] > need.loc["13003", "need_index"] > need.loc["13005", "need_index"]
    assert bool(need.loc["13001", "high_need"]) is True
    assert bool(need.loc["13005", "high_need"]) is False
    assert pd.isna(need.loc["13007", "high_need"])


def test_tract_need_is_population_weighted(norp_source):
    tracts = _tracts([
        ["13001000100", "13001", 3000, 40.0, 1.0],
        ["13001000200", "13001", 1000, 20.0, 0.0],
        ["13003000100", "13003", 500, 10.0, 0.5],
    ])
    out = aggregate_tract_need(tracts, norp_source).set_index("county_fips")
    assert out.loc["13001", "avg_housing_burden"] == pytest.approx(35.0)   # not the plain mean, 30
    assert out.loc["13001", "food_desert_share"] == pytest.approx(0.75)
    assert out.loc["13001", "norp_tracts"] == 2
    assert out.loc["13001", "norp_population"] == 4000
    assert out.loc["13003", "avg_housing_burden"] == pytest.approx(10.0)


def test_tract_need_pads_county_fips(norp_source):
    # The NORP file stores County Fips without its leading zero (9001 = Fairfield, CT).
    tracts = _tracts([["9001010101", "9001", 100, 30.0, 0.0],
                      ["1001020100", "1001", 100, 25.0, 1.0]])
    out = aggregate_tract_need(tracts, norp_source)
    assert out["county_fips"].tolist() == ["01001", "09001"]


def test_tract_need_skips_zero_population_and_missing_values(norp_source):
    tracts = _tracts([
        ["13001000100", "13001", 1000, 30.0, None],   # no food desert value
        ["13001000200", "13001", 1000, 10.0, 1.0],
        ["13001000300", "13001", 0, 50.0, 0.0],       # nobody lives here
        ["13003000100", "13003", 0, 50.0, 0.0],       # county with no population at all
    ])
    out = aggregate_tract_need(tracts, norp_source).set_index("county_fips")
    assert out.loc["13001", "avg_housing_burden"] == pytest.approx(20.0)
    assert out.loc["13001", "food_desert_share"] == pytest.approx(1.0)   # only the tract with a value
    assert math.isnan(out.loc["13003", "avg_housing_burden"])
    assert math.isnan(out.loc["13003", "food_desert_share"])


def test_load_norp_tracts_keeps_codes_as_text(norp_source, tmp_path):
    path = tmp_path / "disadvantaged_communities.csv"
    path.write_text("ID,Geoid,County Fips,Population,Avg Housing Burden,Food Desert Pct,Dac Score\n"
                    "1,01001020100,1001,4300,24.3,0E-10,18.1\n", encoding="utf-8")
    tracts = load_norp_tracts(path, norp_source)
    assert sorted(tracts.columns) == sorted(norp_source.columns())
    assert tracts.loc[0, "Geoid"] == "01001020100"
    assert tracts.loc[0, "County Fips"] == "1001"
    assert tracts.loc[0, "Food Desert Pct"] == 0.0


def test_add_norp_need_repeats_snapshot_for_every_year(acs_source, norp_source):
    acs = pd.concat([acs_rows_to_frame(_rows(), acs_source, y) for y in (2021, 2022)])
    need = to_need_table(acs, acs_source)
    county = aggregate_tract_need(_tracts([["13001000100", "13001", 100, 33.0, 0.25]]), norp_source)
    out = add_norp_need(need, county)
    assert list(out.columns) == list(NEED_COLUMNS) + list(NORP_NEED_COLUMNS)
    assert len(out) == len(need)
    a = out[out["county_fips"] == "13001"]
    assert a["acs_year"].tolist() == [2021, 2022]
    assert a["avg_housing_burden"].tolist() == [33.0, 33.0]
    assert a["food_desert_share"].tolist() == [0.25, 0.25]
    # counties absent from the NORP table stay in the need table with missing values
    assert out.loc[out["county_fips"] == "13003", "avg_housing_burden"].isna().all()
