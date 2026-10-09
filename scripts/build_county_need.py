"""Checkpoint 2: county need table (ACS 5-year, 2018-2022).

Pulls the contracted ACS variables for every county and year, computes
poverty and unemployment rates and a per-year need index with pandas, and
writes data/derived/county_need.csv plus data/output/county_need_report.json.
The raw API responses are cached in data/raw/acs/ (git-ignored) so a rerun
does not hit the API again.

Usage:
    python scripts/build_county_need.py [--years 2018 2022] [--api-key KEY]
    (or set CENSUS_API_KEY; the API also works without a key at this volume)
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
from theorylab.need import acs_rows_to_frame, fetch_acs_rows, to_need_table  # noqa: E402

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
    args = ap.parse_args()

    source = load_catalog()["acs_county"]
    frames, contracts = [], {}
    for year in args.years or source.years:
        df = acs_rows_to_frame(load_year(source, year, args.api_key), source, year)
        contracts[str(year)] = check_contract(df, source, label=f"acs_county-{year}")
        frames.append(df)

    need = to_need_table(pd.concat(frames, ignore_index=True), source)
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
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["rows_by_year"], indent=2))
    violations = [v for c in contracts.values() for v in c["violations"]]
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
