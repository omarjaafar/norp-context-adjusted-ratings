import math

import pandas as pd
import pytest

from theorylab.catalog import load_catalog
from theorylab.need import NEED_COLUMNS, acs_rows_to_frame, acs_url, to_need_table


@pytest.fixture
def acs_source():
    return load_catalog()["acs_county"]


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
