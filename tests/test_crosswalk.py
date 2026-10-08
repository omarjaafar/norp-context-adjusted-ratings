import pandas as pd
import pytest

from theorylab.crosswalk import (build_ein_crosswalk, build_reference_index, county_key,
                                 match_rate, resolve_county)
from theorylab.ids import normalize_ein


@pytest.fixture
def ref():
    # Same columns as the Census national_county2020.txt reference file.
    rows = [
        ("FL", "12", "086", "Miami-Dade County"),
        ("FL", "12", "027", "DeSoto County"),
        ("CT", "09", "001", "Fairfield County"),
        ("MO", "29", "510", "St. Louis city"),
        ("MO", "29", "189", "St. Louis County"),
        ("VA", "51", "510", "Alexandria city"),
        ("NM", "35", "013", "Doña Ana County"),
    ]
    df = pd.DataFrame(rows, columns=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"])
    df["county_fips"] = df["STATEFP"] + df["COUNTYFP"]
    return df


def test_normalize_ein():
    out = normalize_ein(pd.Series(["12-3456789", "1234567", None, "abc", "1234567890"]))
    assert out.iloc[0] == "123456789"
    assert out.iloc[1] == "001234567"
    assert out.iloc[2:].isna().all()


@pytest.mark.parametrize("a,b", [
    ("Miami-Dade County", "MIAMI DADE"),
    ("DeSoto Parish", "De Soto"),
    ("Saint Louis County", "St. Louis"),
    ("Doña Ana County", "DONA ANA"),
    ("Capitol Planning Region", "Capitol"),
])
def test_county_key_equivalences(a, b):
    assert county_key(a) == county_key(b)


def test_county_key_keeps_independent_cities_apart():
    assert county_key("St. Louis city") != county_key("St. Louis County")


def test_resolve_county_methods(ref):
    index, state_fips, collisions = build_reference_index(ref)
    assert collisions == []
    assert resolve_county("FL", "MIAMI-DADE", index, state_fips) == ("12086", "name")
    assert resolve_county("FL", "DADE", index, state_fips) == ("12086", "alias")
    assert resolve_county("FL", "DE SOTO", index, state_fips) == ("12027", "name")
    assert resolve_county("CT", "FAIRFIELD", index, state_fips) == ("09001", "name")
    assert resolve_county("CT", "Capitol Planning Region", index, state_fips) == ("09110", "ct_planning_region")
    assert resolve_county("MO", "ST LOUIS CITY", index, state_fips) == ("29510", "name")
    assert resolve_county("MO", "ST LOUIS", index, state_fips) == ("29189", "name")
    assert resolve_county("VA", "ALEXANDRIA", index, state_fips) == ("51510", "name_city")
    assert resolve_county("MO", "189", index, state_fips) == ("29189", "fips_code")
    assert resolve_county("FL", "12086", index, state_fips) == ("12086", "fips_code")
    assert resolve_county("FL", "NOWHERE", index, state_fips) == (None, "unresolved")
    assert resolve_county("FL", None, index, state_fips) == (None, "missing")


def test_build_ein_crosswalk_prefers_resolved_row(ref):
    ngo = pd.DataFrame({
        "EIN": ["12-0000001", "120000001", "120000002", "120000003", None],
        "STATE": ["FL", "FL", "ct", "FL", "FL"],
        "COUNTY": ["NOWHERE", "DADE", "Capitol Planning Region", None, "DADE"],
        "CATEGORY": ["Human Services", "Human Services", "Education", "Arts", "Arts"],
    })
    xw, diag = build_ein_crosswalk(ngo, ref)
    xw = xw.set_index("ein")
    assert len(xw) == 3
    assert xw.loc["120000001", "county_fips"] == "12086"
    assert xw.loc["120000002", "county_fips"] == "09110"
    assert pd.isna(xw.loc["120000003", "county_fips"])
    assert diag["rows_with_valid_ein"] == 4
    assert diag["eins_resolved"] == 2
    assert diag["resolved_share_by_state"]["FL"] == pytest.approx(0.5)
    assert diag["resolved_share_by_state"]["CT"] == pytest.approx(1.0)


def test_match_rate(ref):
    xw = pd.DataFrame({"ein": ["000000001", "000000002"], "county_fips": ["12086", None]})
    r = match_rate(pd.Series(["1", "2", "3", "1"]), xw)
    assert r == {"unique_eins": 3, "with_county": 1, "match_rate": round(1 / 3, 4)}
