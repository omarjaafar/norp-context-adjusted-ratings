"""Checkpoint 3: run a theory spec end to end on the organization-year panel.

    python scripts/run_theory.py                                   # specs/ratings_fairness.json
    python scripts/run_theory.py --spec specs/<theory>.json --n-permutations 199

Needs data/derived/org_year_panel.csv.gz (python scripts/build_panel.py).
Writes data/output/theory_runs/<theory_id>.json with the verdict and every
number behind it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.spec import load_spec, spec_columns  # noqa: E402
from theorylab.theory_run import (PANEL_PATH, RUNS_DIR, available_columns, load_panel,  # noqa: E402
                                  run_spec, write_report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "specs" / "ratings_fairness.json"))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--n-permutations", type=int, default=None,
                    help="override validator.n_permutations (e.g. 199 for a quick run)")
    ap.add_argument("--no-robustness", action="store_true")
    args = ap.parse_args()

    spec = load_spec(args.spec, available_columns())
    panel = load_panel(args.panel, columns=spec_columns(spec) | {"ein"})
    report = run_spec(panel, spec, n_permutations=args.n_permutations,
                      include_robustness=not args.no_robustness)
    out = write_report(report, RUNS_DIR / f"{spec['theory_id']}.json")
    print(f"{spec['theory_id']}: {report['verdict']} ({'; '.join(report['reasons'])})")
    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
