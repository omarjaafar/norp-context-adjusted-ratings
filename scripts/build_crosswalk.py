"""Checkpoint 2: EIN -> county FIPS crosswalk with per-year match rates.

Reads the NORP NGO table, resolves each (STATE, COUNTY) against the Census
2020 county reference (downloaded once to data/reference/), and writes
data/derived/ein_county_crosswalk.csv.gz plus data/output/crosswalk_report.json.
When NCCS extracts exist, the crosswalk keeps only EINs that file in them and
the report gives the match rate for each extract year (Assumption 4).

Usage:
    python scripts/build_crosswalk.py --norp-raw-dir <path to NORP data/raw>
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT, check_contract, load_catalog  # noqa: E402
from theorylab.crosswalk import build_ein_crosswalk, load_census_counties, match_rate  # noqa: E402
from theorylab.ids import normalize_ein  # noqa: E402

OUT = REPO_ROOT / "data" / "derived" / "ein_county_crosswalk.csv.gz"
REPORT = REPO_ROOT / "data" / "output" / "crosswalk_report.json"


def ensure_census_reference(source) -> Path:
    path = source.extract_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(source.origin_for(), headers={"User-Agent": "theorylab-cp2"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norp-raw-dir", required=True)
    args = ap.parse_args()
    raw = Path(args.norp_raw_dir)
    catalog = load_catalog()

    census_src = catalog["census_counties"]
    ref = load_census_counties(ensure_census_reference(census_src))
    ref_contract = check_contract(ref, census_src)

    ngo_src = catalog["norp_ngo"]
    parts = sorted(glob.glob(str(raw / ngo_src.origin)))
    if not parts:
        print(f"no NGO table parts matching {raw / ngo_src.origin}", file=sys.stderr)
        return 1
    ngo = pd.concat((pd.read_csv(p, usecols=ngo_src.columns(), dtype=str) for p in parts),
                    ignore_index=True)
    ngo_contract = check_contract(ngo, ngo_src)

    xw, diagnostics = build_ein_crosswalk(ngo, ref)

    per_year, nccs_eins = {}, set()
    p01 = catalog["nccs_p01"]
    for year in p01.years:
        path = p01.extract_path(year)
        if not path.exists():
            continue
        eins = pd.read_csv(path, usecols=[p01.keys["ein"]], dtype=str)[p01.keys["ein"]]
        per_year[str(year)] = match_rate(eins, xw)
        nccs_eins.update(normalize_ein(eins).dropna().unique())

    if nccs_eins:
        xw_out = xw[xw["ein"].isin(nccs_eins)]
        scope = "EINs that appear in the committed NCCS Part I extracts"
    else:
        xw_out = xw
        scope = "all EINs in the NORP NGO table (no NCCS extracts found)"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    xw_out.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "reference": {"source": census_src.origin_for(), "contract": ref_contract},
        "ngo_table": {"parts_read": len(parts), "contract": ngo_contract},
        "crosswalk": diagnostics,
        "output": {"path": OUT.relative_to(REPO_ROOT).as_posix(), "scope": scope,
                   "rows": int(len(xw_out)), "size_mb": round(OUT.stat().st_size / 1e6, 2)},
        "nccs_p01_match_rate_by_extract_year": per_year,
        "florida_resolved_share": diagnostics["resolved_share_by_state"].get("FL"),
        "connecticut_resolved_share": diagnostics["resolved_share_by_state"].get("CT"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("nccs_p01_match_rate_by_extract_year",
                                             "florida_resolved_share",
                                             "connecticut_resolved_share")}, indent=2))
    violations = ref_contract["violations"] + ngo_contract["violations"]
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
