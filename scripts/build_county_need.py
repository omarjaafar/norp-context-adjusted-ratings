"""Checkpoint 2: county need table (ACS 5-year 2018-2022 + NORP tract need).

Pulls the contracted ACS variables for every county and year, computes
poverty and unemployment rates and a per-year need index with pandas, then
adds the population-weighted county means of NORP's tract-level housing
burden and food desert share. Connecticut's planning regions, which the ACS
only publishes from 2022, are rebuilt for the earlier years from town-level
ACS and checked against the published 2022 regions. Writes
data/derived/county_need.csv plus data/output/county_need_report.json. The
raw API responses are cached in data/raw/acs/ (git-ignored) so a rerun does
not hit the API again.

Usage:
    python scripts/build_county_need.py --norp-raw-dir <path to NORP data/raw>
        [--years 2018 2022] [--api-key KEY]
    (or set CENSUS_API_KEY; the Census API rejects requests without a key)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT, check_contract, load_catalog  # noqa: E402
from theorylab.need import (NORP_NEED_COLUMNS, acs_rows_to_frame, add_norp_need,  # noqa: E402
                            aggregate_tract_need, ct_region_rows, fetch_acs_rows,
                            load_norp_tracts, to_canonical, to_need_table, town_region_map,
                            vintage_correlation)

CACHE = REPO_ROOT / "data" / "raw" / "acs"
REPORT = REPO_ROOT / "data" / "output" / "county_need_report.json"


CT_TOWNS = {"geo_for": "county subdivision:*", "geo_in": "state:09"}


def load_year(source, year: int, api_key: str | None, name: str = "county", **geo) -> list:
    cache = CACHE / f"acs5_{name}_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = fetch_acs_rows(source, year, api_key, **geo)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def ct_region_report(towns, region_of_town, regions, published, estimated):
    """Measure the town -> planning-region rebuild and check it against the published 2022 rows."""
    real = towns[towns["cousub"] != "00000"]
    unmapped = real[~real["cousub"].isin(region_of_town.index)]
    counts = ["population", "poverty_universe", "below_poverty", "civilian_labor_force", "unemployed"]
    both = regions[regions["acs_year"] == 2022].merge(
        published[published["acs_year"] == 2022], on="county_fips", suffixes=("_towns", ""))
    income_err = (both["median_household_income_towns"] / both["median_household_income"] - 1).abs()
    report = {
        "method": "sum of town (county subdivision) ACS rows; median income interpolated from B19001",
        "towns_mapped": int(len(region_of_town)),
        "regions": int(region_of_town.nunique()),
        "towns_by_year": {str(k): int(v) for k, v in real.groupby("acs_year").size().items()},
        "towns_unmapped_by_year": {str(k): int(v) for k, v in
                                   unmapped.groupby("acs_year").size().items()},
        "estimated_rows_by_year": {str(k): int(v) for k, v in
                                   estimated.groupby("acs_year").size().items()},
        "check_against_published_2022": {
            "regions_compared": int(len(both)),
            "max_abs_count_difference": {c: int((both[f"{c}_towns"] - both[c]).abs().max())
                                         for c in counts},
            "median_income_mean_abs_pct_error": round(float(income_err.mean()) * 100, 2),
            "median_income_max_abs_pct_error": round(float(income_err.max()) * 100, 2),
        },
    }
    violations = []
    if len(unmapped):
        violations.append(f"{len(unmapped)} Connecticut town rows have no planning region")
    if len(both) != region_of_town.nunique():
        violations.append("2022 planning regions rebuilt from towns do not line up with the ACS")
    return report, violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int)
    ap.add_argument("--api-key", default=os.environ.get("CENSUS_API_KEY"))
    ap.add_argument("--norp-raw-dir", required=True)
    args = ap.parse_args()

    catalog = load_catalog()
    source = catalog["acs_county"]
    norp_src = catalog["norp_disadvantaged"]
    tract_path = Path(args.norp_raw_dir) / norp_src.origin
    if not tract_path.exists():
        print(f"no NORP tract table at {tract_path}", file=sys.stderr)
        return 1

    frames, contracts = [], {}
    years = list(args.years or source.years)
    for year in years:
        df = acs_rows_to_frame(load_year(source, year, args.api_key), source, year)
        contracts[str(year)] = check_contract(df, source, label=f"acs_county-{year}")
        frames.append(df)

    acs = pd.concat(frames, ignore_index=True)

    # Connecticut planning regions for the years the ACS still used legacy counties.
    town_src = catalog["acs_ct_towns"]
    towns = []
    for year in town_src.years:
        df = acs_rows_to_frame(load_year(town_src, year, args.api_key, "ct_towns", **CT_TOWNS),
                               town_src, year)
        contracts[f"ct_towns-{year}"] = check_contract(df, town_src, label=f"acs_ct_towns-{year}")
        towns.append(df)
    towns = to_canonical(pd.concat(towns, ignore_index=True), town_src)
    region_of_town = town_region_map(towns[towns["acs_year"] == 2022])
    regions = ct_region_rows(towns, region_of_town)
    published = to_need_table(acs, source)
    has_regions = set(published.loc[published["county_fips"].isin(set(region_of_town)), "acs_year"])
    estimated = regions[~regions["acs_year"].isin(has_regions)]
    need = to_need_table(acs, source, estimated=estimated)
    ct_report, ct_violations = ct_region_report(towns, region_of_town, regions, published, estimated)

    tracts = load_norp_tracts(tract_path, norp_src)
    norp_contract = check_contract(tracts, norp_src)
    county_norp = aggregate_tract_need(tracts, norp_src)
    need = add_norp_need(need, county_norp)
    matched = need["county_fips"].isin(county_norp["county_fips"])
    pop = pd.to_numeric(tracts["Population"], errors="coerce")
    flag = pd.to_numeric(tracts["Food Desert Pct"], errors="coerce")

    out = source.extract_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    need.to_csv(out, index=False)

    ct = need[need["county_fips"].str.startswith("09")]
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "output": out.relative_to(REPO_ROOT).as_posix(),
        "contracts": contracts,
        "rows_by_year": {str(k): int(v) for k, v in need.groupby("acs_year").size().items()},
        "null_rates": {c: round(float(need[c].isna().mean()), 4)
                       for c in ("median_household_income", "poverty_rate",
                                 "unemployment_rate", "need_index")},
        "high_need_counties_by_year": {str(k): int(v) for k, v in
                                       need.groupby("acs_year")["high_need"].sum().items()},
        "connecticut_codes_by_year": {str(k): sorted(g["county_fips"].tolist())
                                      for k, g in ct.groupby("acs_year")},
        "ct_planning_regions": ct_report,
        # How much one ACS 5-year vintage differs from the next (published rows only).
        "vintage_correlation": vintage_correlation(
            need[~need["acs_estimated"]], ["poverty_rate", "unemployment_rate", "need_index"],
            [*zip(years, years[1:]), (years[0], years[-1])]),
        "norp_need": {
            "contract": norp_contract,
            "tracts_zero_population": int((pop == 0).sum()),
            "food_desert_flagged_tract_share": round(float(flag.mean()), 4),
            "counties": int(len(county_norp)),
            "counties_null": {c: int(county_norp[c].isna().sum()) for c in NORP_NEED_COLUMNS},
            "need_rows_matched_by_year": {str(k): int(v) for k, v in
                                          matched.groupby(need["acs_year"]).sum().items()},
            "need_rows_unmatched_by_state_by_year": {
                str(k): {s: int(n) for s, n in g.str[:2].value_counts().sort_index().items()}
                for k, g in need.loc[~matched, "county_fips"].groupby(need["acs_year"])},
            "null_rates": {c: round(float(need[c].isna().mean()), 4) for c in NORP_NEED_COLUMNS},
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["rows_by_year"], indent=2))
    violations = [v for c in (*contracts.values(), norp_contract) for v in c["violations"]]
    violations += ct_violations
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
