import pandas as pd
import pytest

from theorylab.catalog import ContractError, check_contract, load_catalog
from theorylab.panel import _longest_run, assumption_4_check, build_panel, prepare_part


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def _p01(src):
    rows = [
        # ein, tax_year, rev, exp, fundr, contr, prog, sal, na_boy, na_eoy
        ("12-0000001", "2018", "100", "90", "5", "60", "30", "40", "50", "60"),
        ("120000001", "2019", "120", "100", "5", "70", "40", "40", "60", "80"),
        ("120000001", "2020", "150", "110", "5", "80", "60", "40", "80", "120"),
        ("120000001", "2020", "999", "110", "5", "80", "60", "40", "80", "120"),  # amended: kept
        ("120000002", "2019", "200", "250", "0", "200", "0", "10", "50", "0"),
        ("", "2019", "1", "1", "1", "1", "1", "1", "1", "1"),                     # no EIN
    ]
    return pd.DataFrame(rows, columns=src.columns())


def _p09(src):
    rows = [("120000001", "2018", "90", "72", "13.5", "4.5"),
            ("120000001", "2019", "100", "80", "15", "5"),
            ("120000002", "2019", "250", "200", "50", "0")]
    return pd.DataFrame(rows, columns=src.columns())


def _p10(src):
    rows = [("120000001", "2019", "400", "100")]
    return pd.DataFrame(rows, columns=src.columns())


def test_catalog_loads_and_names_are_canonical(catalog):
    assert {"nccs_p01", "nccs_p09", "nccs_p10", "norp_ngo", "census_counties", "acs_county"} <= set(catalog)
    for name in ("nccs_p01", "nccs_p09", "nccs_p10"):
        src = catalog[name]
        assert src.years == (2018, 2019, 2020, 2021, 2022)
        assert src.keys == {"ein": "ORG_EIN", "tax_year": "TAX_YEAR"}
        assert "{year}" in src.origin and "{year}" in src.extract


def test_check_contract_reports_and_raises(catalog):
    src = catalog["nccs_p09"]
    report = check_contract(_p09(src), src)
    assert report["rows"] == 3
    assert report["violations"]  # 3 rows is far below the expected range
    with pytest.raises(ContractError):
        check_contract(_p09(src).drop(columns=["F9_09_EXP_TOT_MGMT"]), src)


def test_prepare_part_dedupes_and_drops_missing_keys(catalog):
    part, log = prepare_part(_p01(catalog["nccs_p01"]), catalog["nccs_p01"])
    assert log == {"rows_in": 6, "dropped_missing_keys": 1, "dropped_duplicate_keys": 1, "rows_out": 4}
    row = part[(part["ein"] == "120000001") & (part["tax_year"] == 2020)].iloc[0]
    assert row["revenue_total"] == 999.0


def test_build_panel_joins_and_logs(catalog):
    p01, _ = prepare_part(_p01(catalog["nccs_p01"]), catalog["nccs_p01"])
    p09, _ = prepare_part(_p09(catalog["nccs_p09"]), catalog["nccs_p09"])
    p10, _ = prepare_part(_p10(catalog["nccs_p10"]), catalog["nccs_p10"])
    crosswalk = pd.DataFrame({"ein": ["120000001", "120000002"],
                              "county_fips": ["13001", None],
                              "state": ["GA", "GA"],
                              "category": ["Human Services", "Education"]})
    need = pd.DataFrame({"county_fips": ["13001", "13001"], "acs_year": [2018, 2019],
                         "acs_county_name": ["A County, Georgia"] * 2,
                         "poverty_rate": [0.3, 0.28], "need_index": [1.2, 1.1],
                         "high_need": [True, True]})

    panel, log = build_panel(p01, p09, p10, crosswalk, need)
    assert len(panel) == 4
    by = log["by_tax_year"]
    assert by["2018"] == {"rows": 1, "with_p09": 1, "with_p10": 0, "with_county": 1, "with_need": 1,
                          "county_match_rate": 1.0, "need_match_rate": 1.0}
    assert by["2019"]["rows"] == 2 and by["2019"]["with_county"] == 1 and by["2019"]["with_p10"] == 1
    assert by["2020"]["with_need"] == 0
    assert log["eins_with_3plus_consecutive_linked_years"] == 1

    r2019 = panel[(panel["ein"] == "120000001") & (panel["tax_year"] == 2019)].iloc[0]
    assert r2019["program_expense_ratio"] == pytest.approx(0.8)
    assert r2019["liabilities_to_assets"] == pytest.approx(0.25)
    assert r2019["revenue_growth"] == pytest.approx(0.2)
    assert r2019["poverty_rate"] == pytest.approx(0.28)

    a4 = assumption_4_check(log, years=(2018, 2019, 2020))
    assert a4["county_match_rate_by_year"] == {"2018": 1.0, "2019": 0.5, "2020": 1.0}
    assert a4["holds"] is False


def test_longest_run():
    assert _longest_run([2018, 2019, 2021, 2022, 2020]) == 5
    assert _longest_run([2018, 2020, 2022]) == 1
    assert _longest_run([]) == 0
