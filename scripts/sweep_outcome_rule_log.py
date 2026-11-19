"""The outcome rule's log for each main sweep cell (CP5, O5).

The CP5 sweep rerun (data/output/sweep/main.json, run at 466f21a) applied the heavy-tail rule
but its cell rows did not keep the rule's log; sweep._cell_row records it from now on. The
rule depends only on the cell's population, so this rebuilds each population (no
permutations, a few minutes) and writes what the rule did: SD / robust SD, whether it
clipped, the clip points and rows clipped.

    python scripts/sweep_outcome_rule_log.py   -> data/output/sweep/outcome_rule_log.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.sweep import cell_spec, main_grid  # noqa: E402
from theorylab.theory_run import (add_derived, available_columns, load_panel,  # noqa: E402
                                  select_population, spec_columns)

OUT = REPO_ROOT / "data" / "output" / "sweep" / "outcome_rule_log.json"


def main() -> int:
    grid = main_grid(available_columns())
    specs = [cell_spec(r, n, s) for r, n, s in grid]
    cols = sorted(set().union(*(spec_columns(s) for s in specs)) | {"ein"})
    panel = load_panel(columns=cols)
    cells = {}
    for spec in specs:
        _, log = select_population(add_derived(panel, spec_columns(spec)), spec)
        cells[spec["theory_id"]] = {"rows": log["rows_out"], **log["outcome_rule"]}
    clipped = sorted(k for k, v in cells.items() if v["applied"])
    out = {"rule": specs[0]["validator"]["outcome_rule"], "cells": len(cells),
           "clipped": clipped, "not_clipped": sorted(set(cells) - set(clipped)), "by_cell": cells}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(clipped)} of {len(cells)} cells clipped -> {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
