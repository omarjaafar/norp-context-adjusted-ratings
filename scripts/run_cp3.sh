#!/usr/bin/env bash
# Checkpoint 3: Executor + Validator on the committed data layer.
#
#   bash scripts/run_cp3.sh            # full run (999 permutations for the ratings theory)
#   QUICK=1 bash scripts/run_cp3.sh    # 199 permutations everywhere
#
# Needs only committed inputs: the panel is rebuilt from the CP2 extracts, crosswalk
# and need table if it is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

PERM_ARGS=()
if [[ "${QUICK:-0}" == "1" ]]; then PERM_ARGS=(--n-permutations 199); fi

echo "== 1/7 unit tests";                      python -m pytest -q
if [[ ! -f data/derived/org_year_panel.csv.gz ]]; then
  echo "== panel missing: rebuilding";         python scripts/build_panel.py
fi
echo "== 2/7 Part IX coverage by return type"; python scripts/part9_coverage.py
echo "== 3/7 Assumption 5 (MixedLM, 2022)";    python scripts/measure_assumption5.py
echo "== 4/7 ratings theory end to end";       python scripts/run_theory.py ${PERM_ARGS[@]+"${PERM_ARGS[@]}"}
echo "== 5/7 ratings theory by tax year";      python scripts/run_by_year.py ${PERM_ARGS[@]+"${PERM_ARGS[@]}"}
echo "== 6/7 rating anchor sensitivity";       python scripts/anchor_sensitivity.py
echo "== 7/7 done: data/output/{part9_coverage,assumption_5,anchor_sensitivity}.json, data/output/theory_runs/"
