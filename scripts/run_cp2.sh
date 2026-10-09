#!/usr/bin/env bash
# Rebuild every Checkpoint 2 output from a bare clone.
#
#   bash scripts/run_cp2.sh <path to NORP data/raw> [<raw NCCS download dir>]
#
# Inputs that are NOT committed:
#   - NORP raw data (NGO table parts; needed by the crosswalk): from the NORP Metabase
#     (norpp.cc.gatech.edu) or the data/raw folder of the Summer 2026 NORP repo
#     https://github.gatech.edu/IEC-Summer-26/NORP-Food-Assistance-Need-Capacity-Gap-Explorer
#   - Full NCCS efile CSVs (~4 GB): downloaded by step 2 from the public NCCS bucket.
#   - ACS county estimates: pulled from the Census API by step 4 (set CENSUS_API_KEY if rate-limited).
# Everything else (extracts, crosswalk, need table, reports) is committed, so steps 2-5
# can be skipped to inspect results without running anything.
set -euo pipefail

NORP_RAW="${1:?usage: run_cp2.sh <NORP data/raw> [<NCCS raw dir>]}"
NCCS_RAW="${2:-${NCCS_RAW_DIR:-data/raw/nccs}}"

cd "$(dirname "$0")/.."

echo "== 1/6 unit tests";            python -m pytest -q
echo "== 2/6 NCCS extracts";         python scripts/build_nccs_extracts.py --raw-dir "$NCCS_RAW"
echo "== 3/6 EIN -> county";         python scripts/build_crosswalk.py --norp-raw-dir "$NORP_RAW"
echo "== 4/6 county need (ACS)";     python scripts/build_county_need.py
echo "== 5/6 organization-year panel"; python scripts/build_panel.py
echo "== 6/6 feasibility (CP1/A5)";  python scripts/feasibility_check.py --data-dir "$NORP_RAW"
echo "done: see data/extracts/nccs/MANIFEST.json and data/output/*.json"
