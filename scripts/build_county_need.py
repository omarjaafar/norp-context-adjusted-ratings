"""Checkpoint 2: county need table (ACS 5-year 2018-2022 + NORP tract need).

Pulls the contracted ACS variables for every county and year, computes
poverty and unemployment rates and a per-year need index with pandas, then
adds the population-weighted county means of NORP's tract-level housing
burden and food desert share. Writes data/derived/county_need.csv plus
data/output/county_need_report.json. The raw API responses are cached in
data/raw/acs/ (git-ignored) so a rerun does not hit the API again.

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
                            aggregate_tract_need, fetch_acs_rows, load_norp_tracts,
                            to_need_table)

CACHE = REPO_ROOT / "data" / "raw" / "acs"
REPORT = REPO_ROOT / "data" / "output" / "county_need_report.json"


def load_year(source, year: int, api_key: str | None) -> list:
    cache = CACHE / f"acs5_county_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = fetch_acs_rows(source, year, api_key)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


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
    for year in args.years or source.years:
        df = acs_rows_to_frame(load_year(source, year, args.api_key), source, year)
        contracts[str(year)] = check_contract(df, source, label=f"acs_county-{year}")
        frames.append(df)

    need = to_need_table(pd.concat(frames, ignore_index=True), source)

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
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
