"""CP4 ratio x need sweep (O7). Needs the panel (scripts/build_panel.py).

Parts (each writes data/output/sweep/<part>.json):
  main       the reported sweep: 4 rating ratios x need_index, avg_housing_burden (+ chr_* if
             present), fixed scheme (whole-county shuffle, state FE), 999 permutations
  ta_before  the TA's 40-cell grid with the TA's cell as written (county-year shuffle, no
             state FE), 199 permutations
  ta_after   the same 40 cells and signs under the fixed scheme, 199 permutations
Then data/output/sweep.json collects the main result and the before/after counts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.sweep import cell_spec, main_grid, run_sweep, ta_grid  # noqa: E402
from theorylab.theory_run import available_columns, load_panel, spec_columns, write_report  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "output" / "sweep"
PARTS = {"main": ("fixed", None), "ta_before": ("ta", 199), "ta_after": ("fixed", 199)}


def summarize() -> dict:
    parts = {p: json.loads((OUT_DIR / f"{p}.json").read_text()) for p in PARTS
             if (OUT_DIR / f"{p}.json").exists()}
    out = {"parts": {p: {"scheme": d["scheme"], "n_permutations": d["n_permutations"],
                         "seconds": d.get("seconds"), **d["summary"]} for p, d in parts.items()}}
    if "main" in parts:
        out["main_cells"] = [{k: c[k] for k in ("cell", "verdict", "h1_r", "h2_std_effect",
                                                 "family_q_h1", "family_q_h2")}
                             for c in parts["main"]["cells"]]
    if {"ta_before", "ta_after"} <= set(parts):
        before = {c["cell"]: c for c in parts["ta_before"]["cells"]}
        after = {c["cell"]: c for c in parts["ta_after"]["cells"]}
        out["before_after"] = {
            "cells_significant_before": sum(bool(c["significant_after_family_bh"]) for c in before.values()),
            "cells_significant_after": sum(bool(c["significant_after_family_bh"]) for c in after.values()),
            "still_significant_after": sorted(k for k in before if before[k]["significant_after_family_bh"]
                                              and after.get(k, {}).get("significant_after_family_bh")),
        }
    path = REPO_ROOT / "data" / "output" / "sweep.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS), action="append",
                    help="repeatable; default: all three")
    ap.add_argument("--permutations", type=int, default=None, help="override the part's default")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        grids = {"main": main_grid(available_columns()), "ta_before": ta_grid(), "ta_after": ta_grid()}
        todo = args.part or list(PARTS)
        cols = set()
        for p in todo:
            for r, n, s in grids[p]:
                cols |= spec_columns(cell_spec(r, n, s, PARTS[p][0]))
        panel = load_panel(columns=sorted(cols))
        for p in todo:
            scheme, n_perm = PARTS[p]
            t0 = time.time()
            out = run_sweep(panel, grids[p], scheme, args.permutations or n_perm)
            out["seconds"] = round(time.time() - t0, 1)
            write_report(out, OUT_DIR / f"{p}.json")
            print(p, json.dumps(out["summary"]), f"{out['seconds']} s", flush=True)
    print(json.dumps(summarize(), indent=2))


if __name__ == "__main__":
    main()
