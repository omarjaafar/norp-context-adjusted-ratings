"""Checkpoint 1 probe: which multi-year 990 tables are available from NCCS?

The plan needs more than the single-year Part I extract in the NORP Metabase:
Part IX (functional expenses, for the program expense ratio), Part X (balance
sheet, for liabilities-to-assets and working capital), and several tax years
(Charity Navigator-style ratings average three years; outcome checks need
later years). This script checks, without downloading full files, that each
table/year exists on the NCCS efile bucket, how large it is, and which of the
columns we plan to use are present.

Writes data/output/nccs_efile_probe.json.

Usage:
    python scripts/probe_nccs_efile.py
"""

import datetime
import json
import urllib.request
from pathlib import Path

BASE = "https://nccs-efile.s3.us-east-1.amazonaws.com/public/efile_v2_1"
YEARS = range(2018, 2023)
TABLES = {
    "F9-P01-T00-SUMMARY": ["F9_01_REV_TOT_CY", "F9_01_EXP_TOT_CY", "F9_01_EXP_FUNDR_TOT_CY"],
    "F9-P09-T00-EXPENSES": ["F9_09_EXP_TOT_TOT", "F9_09_EXP_TOT_PROG",
                            "F9_09_EXP_TOT_MGMT", "F9_09_EXP_TOT_FUNDR"],
    "F9-P10-T00-BALANCE-SHEET": ["F9_10_ASSET_TOT_EOY", "F9_10_LIAB_TOT_EOY"],
}
HEADER_BYTES = 20000


def probe(url, wanted):
    head = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(head, timeout=30) as r:
            size = int(r.headers.get("Content-Length", 0))
    except Exception as e:  # missing table/year is a result, not a crash
        return {"exists": False, "error": str(e)}
    rng = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES}"})
    with urllib.request.urlopen(rng, timeout=30) as r:
        header = r.read().decode("utf-8", "replace").splitlines()[0].split(",")
    return {
        "exists": True,
        "size_mb": round(size / 1e6, 1),
        "n_columns": len(header),
        "wanted_columns_present": {c: c in header for c in wanted},
    }


def main():
    out = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "base_url": BASE, "tables": {}}
    for table, wanted in TABLES.items():
        out["tables"][table] = {}
        for y in YEARS:
            res = probe(f"{BASE}/{table}-{y}.CSV", wanted)
            out["tables"][table][str(y)] = res
            print(table, y, res.get("size_mb"), res.get("exists"))
    path = Path("data/output/nccs_efile_probe.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
