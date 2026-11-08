import math

import pandas as pd
import pytest

from theorylab.catalog import load_catalog
from theorylab.need import (INCOME_BRACKET_COLUMNS, NEED_COLUMNS, NORP_NEED_COLUMNS,
                            acs_rows_to_frame, acs_url, add_norp_need, aggregate_tract_need,
                            ct_region_rows, grouped_median, load_norp_tracts, to_need_table,
                            town_region_map, vintage_correlation)


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


def test_acs_url_can_ask_for_connecticut_towns():
    url = acs_url(load_catalog()["acs_ct_towns"], 2019,
                  geo_for="county subdivision:*", geo_in="state:09")
    assert "for=county%20subdivision:*" in url
    assert "in=state:09" in url
    assert "B19001_017E" in url


def test_grouped_median_interpolates_inside_the_bracket():
    counts = pd.DataFrame([
        [10, 10, 20, 0],     # half = 20, reached at the top of bracket 2
        [10, 10, 40, 40],    # half = 50, three quarters of the way through bracket 3
        [0, 0, 1, 9],        # median in the open-ended top bracket
        [0, 0, 0, 0],        # nobody
    ])
    med = grouped_median(counts, edges=(0, 100, 200, 400))
    assert med.iloc[0] == pytest.approx(200.0)
    assert med.iloc[1] == pytest.approx(350.0)
    assert math.isnan(med.iloc[2])
    assert math.isnan(med.iloc[3])


def _towns(rows):
    cols = ["statefp", "countyfp", "cousub", "acs_year", "population", "poverty_universe",
            "below_poverty", "civilian_labor_force", "unemployed", "lo", "hi"]
    df = pd.DataFrame(rows, columns=cols)
    for c in INCOME_BRACKET_COLUMNS:
        df[c] = 0
    df[INCOME_BRACKET_COLUMNS[0]] = df.pop("lo")     # under $10,000
    df[INCOME_BRACKET_COLUMNS[11]] = df.pop("hi")    # $75,000 to $99,999
    return df


def test_towns_sum_to_planning_regions():
    # 2022 nests towns under planning regions (110, 120); 2018 under legacy counties (003, 001).
    towns = _towns([
        ["09", "110", "01080", 2022, 100, 100, 10, 50, 5, 10, 30],
        ["09", "110", "02060", 2022, 300, 300, 30, 150, 5, 10, 30],
        ["09", "120", "08000", 2022, 500, 500, 100, 250, 50, 40, 10],
        ["09", "120", "00000", 2022, 0, 0, 0, 0, 0, 0, 0],            # water, no town
        ["09", "003", "01080", 2018, 90, 90, 9, 40, 4, 10, 30],
        ["09", "003", "02060", 2018, 310, 300, 21, 160, 12, 10, 30],
        ["09", "001", "08000", 2018, 480, 480, 96, 240, 24, 40, 10],
        ["09", "001", "00000", 2018, 0, 0, 0, 0, 0, 0, 0],
    ])
    region_of_town = town_region_map(towns[towns["acs_year"] == 2022])
    assert region_of_town.to_dict() == {"01080": "09110", "02060": "09110", "08000": "09120"}

    out = ct_region_rows(towns, region_of_town).set_index(["county_fips", "acs_year"])
    capitol = out.loc[("09110", 2018)]
    assert capitol["population"] == 400
    assert capitol["below_poverty"] == 30
    assert capitol["unemployed"] == 16
    # 20 households under $10k and 60 in $75k-$100k: the median is a third into the upper bracket
    assert capitol["median_household_income"] == pytest.approx(75_000 + 25_000 / 3, abs=1)
    # 40 of 50 households under $10k: the median is five eighths into the first bracket
    assert out.loc[("09120", 2018), "median_household_income"] == pytest.approx(6_250)
    assert len(out) == 4


def test_estimated_rows_do_not_move_published_rows(acs_source):
    acs = acs_rows_to_frame(_rows(), acs_source, 2021)
    published = to_need_table(acs, acs_source)
    extra = pd.DataFrame({"county_fips": ["13999"], "acs_year": [2021], "population": [1000],
                          "median_household_income": [20000.0], "poverty_universe": [1000.0],
                          "below_poverty": [600.0], "civilian_labor_force": [500.0],
                          "unemployed": [200.0]})
    both = to_need_table(acs, acs_source, estimated=extra)
    assert list(both.columns) == list(NEED_COLUMNS)
    kept = both[~both["acs_estimated"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, published, check_dtype=False)
    added = both[both["acs_estimated"]].iloc[0]
    assert added["county_fips"] == "13999"
    assert added["poverty_rate"] == pytest.approx(0.6)
    # poorer than every published county, so it scores above all of them
    assert added["need_index"] > published["need_index"].max()
    assert bool(added["high_need"]) is True


def test_vintage_correlation_uses_counties_present_in_both_years():
    need = pd.DataFrame({
        "county_fips": ["01", "02", "03", "01", "02", "03", "04"],
        "acs_year": [2018, 2018, 2018, 2019, 2019, 2019, 2019],
        "poverty_rate": [0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.9],
    })
    out = vintage_correlation(need, ["poverty_rate"], [(2018, 2019)])
    assert out == {"2018_vs_2019": {"counties": 3, "poverty_rate": 1.0}}


# --- County Health Rankings (CP4, D8) --------------------------------------

from theorylab.need import (CHR_COLUMNS, CHR_CRIME_RELEASE, add_chr_need,  # noqa: E402
                            chr_by_tax_year, chr_range_violations, chr_release_for,
                            read_chr_release)

_CHR_LABELS = ["State FIPS Code", "County FIPS Code", "5-digit FIPS Code", "State Abbreviation",
               "Name", "Poor or Fair Health raw value", "Uninsured raw value",
               "Some College raw value", "High School Completion raw value",
               "Violent Crime raw value"]
_CHR_CODES = ["statecode", "countycode", "fipscode", "state", "county", "v002_rawvalue",
              "v085_rawvalue", "v069_rawvalue", "v168_rawvalue", "v043_rawvalue"]


def _write_chr(path, rows, drop=()):
    keep = [i for i, c in enumerate(_CHR_CODES) if c not in drop]
    lines = [",".join(_CHR_LABELS[i] for i in keep), ",".join(_CHR_CODES[i] for i in keep)]
    lines += [",".join(str(r[i]) for i in keep) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def _chr_rows(shift=0.0):
    # state row (countycode 000) must be dropped; codes keep leading zeros
    return [["13", "000", "13000", "GA", "Georgia", 0.2, 0.1, 0.6, 0.9, 300],
            ["13", "001", "13001", "GA", "A County", 0.25 + shift, 0.12, 0.5, 0.85, 400],
            ["09", "001", "9001", "CT", "Fairfield County", 0.11 + shift, 0.07, 0.7, 0.92, 150],
            ["09", "110", "09110", "CT", "Capitol Planning Region", 0.13 + shift, 0.06, 0.68, 0.9, ""]]


def test_chr_release_rule_lags_by_measure():
    # survey/model measures: data year = tax year (release - 3)
    assert chr_release_for("chr_poor_fair_health", 2018) == 2021
    assert chr_release_for("chr_uninsured", 2022) == 2025
    # ACS 5-year measures: window ends in the tax year, as in D4 (release - 2)
    assert chr_release_for("chr_some_college", 2018) == 2020
    assert chr_release_for("chr_hs_completion", 2022) == 2024
    # crime is frozen: one release for every year
    assert {chr_release_for("chr_violent_crime_rate", y) for y in range(2018, 2023)} == {
        CHR_CRIME_RELEASE}


def test_read_chr_release_uses_codes_drops_state_rows_and_pads_fips(tmp_path):
    df = read_chr_release(_write_chr(tmp_path / "a.csv", _chr_rows()), 2024)
    assert list(df["fipscode"]) == ["13001", "09001", "09110"]
    assert (df["chr_release"] == 2024).all()
    assert df.loc[df["fipscode"] == "09110", "v043_rawvalue"].isna().all()


def test_read_chr_release_missing_measure_is_nan(tmp_path):
    # v168 starts in the 2021 release, v043 stops after 2022
    df = read_chr_release(_write_chr(tmp_path / "a.csv", _chr_rows(),
                                     drop=("v168_rawvalue", "v043_rawvalue")), 2020)
    assert df["v168_rawvalue"].isna().all() and df["v043_rawvalue"].isna().all()


def test_chr_by_tax_year_takes_each_column_from_its_release(tmp_path):
    releases = {r: read_chr_release(_write_chr(tmp_path / f"{r}.csv", _chr_rows(shift=r / 1e4)), r)
                for r in (2020, 2021, 2022, 2024)}
    extract = pd.concat(releases.values(), ignore_index=True)
    wide = chr_by_tax_year(extract, [2018])
    a = wide[wide["county_fips"] == "13001"].iloc[0]
    assert a["acs_year"] == 2018
    assert a["chr_poor_fair_health"] == pytest.approx(0.25 + 2021 / 1e4)  # release 2021
    assert a["chr_violent_crime_rate"] == 400                              # release 2022
    assert list(wide.columns) == ["county_fips", "acs_year", *CHR_COLUMNS]


def test_chr_shares_are_in_unit_interval_and_violations_are_counted():
    wide = pd.DataFrame({"chr_poor_fair_health": [0.2, 1.2], "chr_uninsured": [0.1, -0.1],
                         "chr_some_college": [0.5, 0.6], "chr_hs_completion": [0.9, 0.9],
                         "chr_violent_crime_rate": [450.0, 1200.0]})
    v = chr_range_violations(wide)
    assert v == {"chr_poor_fair_health": 1, "chr_uninsured": 1, "chr_some_college": 0,
                 "chr_hs_completion": 0, "chr_violent_crime_rate": 0}


def test_add_chr_need_ct_regions_match_only_when_published_and_rerun_is_safe(tmp_path):
    need = pd.DataFrame({"county_fips": ["13001", "09110", "09001"], "acs_year": [2022] * 3,
                         "need_index": [0.1, 0.2, 0.3]})
    # legacy-only release: the planning region 09110 gets nothing
    legacy = [r for r in _chr_rows() if r[2] != "09110"]
    extract = pd.concat([read_chr_release(_write_chr(tmp_path / f"{r}.csv", legacy), r)
                         for r in (2022, 2024, 2025)], ignore_index=True)
    out = add_chr_need(need, chr_by_tax_year(extract, [2022]))
    assert out.set_index("county_fips").loc["09110", list(CHR_COLUMNS)].isna().all()
    assert out.set_index("county_fips").loc["09001", "chr_uninsured"] == pytest.approx(0.07)
    # running twice replaces, never duplicates, the CHR columns
    again = add_chr_need(out, chr_by_tax_year(extract, [2022]))
    assert list(again.columns) == list(out.columns)
    pd.testing.assert_frame_equal(again, out)
    pd.testing.assert_frame_equal(out[list(need.columns)], need)


def test_chr_catalog_block_is_valid():
    src = load_catalog()["chr_county"]
    assert set(src.fields.values()) == set(CHR_COLUMNS)
    assert src.row_bounds(2025) == (3152, 3152)
