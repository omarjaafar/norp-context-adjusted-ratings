"""Checkpoint 1 probe: which multi-year 990 tables are available from NCCS?

The plan needs more than the single-year Part I extract in the NORP Metabase:
Part IX (functional expenses, for the program expense ratio), Part X (balance
sheet, for liabilities-to-assets and working capital), and several tax years
(Charity Navigator-style ratings average three years; outcome checks need
later years). This script checks, without downloading full files, that each
table/year exists on the NCCS efile bucket, how large it is, and which of the
columns we plan to use are present.

CP2: the wanted columns now come from catalog/catalog.yaml (every contracted
column of nccs_p01 / nccs_p09 / nccs_p10), so the probe verifies the whole
data contract, not just the CP1 subset.

Writes data/output/nccs_efile_probe.json.

Usage:
    python scripts/probe_nccs_efile.py
"""

import datetime
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import load_catalog  # noqa: E402

BASE = "https://nccs-efile.s3.us-east-1.amazonaws.com/public/efile_v2_1"
YEARS = range(2018, 2023)
CATALOG_SOURCES = ("nccs_p01", "nccs_p09", "nccs_p10")


def wanted_columns():
    """{table name: contracted raw columns}, read from the catalog."""
    catalog = load_catalog()
    out = {}
    for name in CATALOG_SOURCES:
        src = catalog[name]
        table = src.origin.rsplit("/", 1)[-1].replace("-{year}.CSV", "")
        out[table] = src.columns()
    return out


HEADER_BYTES = 60000


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
    for table, wanted in wanted_columns().items():
        out["tables"][table] = {}
        for y in YEARS:
            res = probe(f"{BASE}/{table}-{y}.CSV", wanted)
            out["tables"][table][str(y)] = res
            missing = [c for c, ok in res.get("wanted_columns_present", {}).items() if not ok]
            print(table, y, res.get("size_mb"), res.get("exists"), "missing:", missing or "none")
    path = Path("data/output/nccs_efile_probe.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
