"""Checkpoint 1 feasibility check.

Answers one question before we commit to the plan: is there enough
organization-level financial data, joined to counties, to build naive
ratio ratings and a context-adjusted (multilevel) rating?

Reads the NORP raw files and writes data/output/feasibility_report.json.
All numbers come from pandas; nothing here is estimated by hand or by an LLM.

Usage:
    python scripts/feasibility_check.py --data-dir <path to NORP data/raw>
"""

import argparse
import datetime
import glob
import json
from pathlib import Path

import pandas as pd

F9_FILE = "F9_P01_T00_SUMMARY_2022.csv"
NGO_GLOB = "ngos_full/NGOs_with_categories.part*.csv.gz"
NCCS_FILE = "nccs_crosswalk_economic.csv"

# 990 Part I fields we plan to build ratios from
FIELDS = {
    "mission": "F9 01 Act Gvrn Act Mission",
    "revenue_cy": "F9 01 Rev Tot Cy",
    "revenue_py": "F9 01 Rev Tot Py",
    "contributions_cy": "F9 01 Rev Contr Tot Cy",
    "program_revenue_cy": "F9 01 Rev Prog Tot Cy",
    "expenses_cy": "F9 01 Exp Tot Cy",
    "fundraising_exp_cy": "F9 01 Exp Fundr Tot Cy",
    "salaries_cy": "F9 01 Exp Sal Etc Cy",
    "employees": "F9 01 Act Gvrn Empl Tot",
    "volunteers": "F9 01 Act Gvrn Vol Tot",
    "net_assets_boy": "F9 01 Nafb Tot Boy",
    "net_assets_eoy": "F9 01 Nafb Tot Eoy",
}


def fill_rates(df):
    return {k: round(float(df[c].notna().mean()), 4) for k, c in FIELDS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="data/output/feasibility_report.json")
    args = ap.parse_args()
    raw = Path(args.data_dir)

    f9 = pd.read_csv(raw / F9_FILE, dtype=str, low_memory=False)
    f9_one = f9.drop_duplicates("Org Ein")

    ngo_parts = sorted(glob.glob(str(raw / NGO_GLOB)))
    ngo = pd.concat(
        pd.read_csv(p, usecols=["EIN", "STATE", "COUNTY", "CATEGORY"], dtype=str)
        for p in ngo_parts
    )

    matched = f9_one.merge(ngo, left_on="Org Ein", right_on="EIN", how="inner")
    matched["county_key"] = matched["STATE"] + "|" + matched["COUNTY"]
    per_county = matched.groupby("county_key").size()

    full990 = matched[matched["Return Type"] == "990"]
    nccs = pd.read_csv(raw / NCCS_FILE, dtype=str)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "f9_rows": len(f9),
            "f9_unique_eins": int(f9["Org Ein"].nunique()),
            "f9_return_types": f9["Return Type"].value_counts().to_dict(),
            "f9_tax_years": f9["Tax Year"].value_counts().to_dict(),
            "ngo_rows": len(ngo),
            "ngo_parts_read": len(ngo_parts),
            "nccs_county_rows": len(nccs),
        },
        "join": {
            "f9_eins_matched_to_ngo_table": len(matched),
            "match_rate_of_f9_eins": round(len(matched) / len(f9_one), 4),
            "full_990_filers_matched": len(full990),
            "distinct_counties_with_a_filer": int(per_county.size),
            "counties_with_10plus_filers": int((per_county >= 10).sum()),
            "median_filers_per_county": float(per_county.median()),
        },
        "field_fill_rates_all_matched": fill_rates(matched),
        "field_fill_rates_full_990_only": fill_rates(full990),
        "full_990_with_mission_text": int(full990[FIELDS["mission"]].notna().sum()),
        "top_categories_matched": matched["CATEGORY"].value_counts().head(10).to_dict(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["join"], indent=2))


if __name__ == "__main__":
    main()
