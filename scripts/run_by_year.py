"""Checkpoint 3: a theory spec's H1 and H2 for each tax year, and whether the gap shifts over time.

    python scripts/run_by_year.py                                  # specs/ratings_fairness.json
    python scripts/run_by_year.py --spec specs/<theory>.json --n-permutations 199

Needs data/derived/org_year_panel.csv.gz (python scripts/build_panel.py).
Writes data/output/theory_runs/<theory_id>_by_year.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.spec import load_spec, spec_columns  # noqa: E402
from theorylab.theory_run import PANEL_PATH, RUNS_DIR, available_columns, load_panel, write_report  # noqa: E402
from theorylab.trends import run_by_year  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "specs" / "ratings_fairness.json"))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--n-permutations", type=int, default=None,
                    help="override validator.n_permutations (e.g. 199 for a quick run)")
    args = ap.parse_args()

    spec = load_spec(args.spec, available_columns())
    panel = load_panel(args.panel, columns=spec_columns(spec) | {"ein"})
    report = run_by_year(panel, spec, n_permutations=args.n_permutations)
    out = write_report(report, RUNS_DIR / f"{spec['theory_id']}_by_year.json")
    for year, r in report["by_year"].items():
        print(f"{year}  H1 {r['H1']['standardized_effect']:+.3f} SD (q {r['H1']['q_value']:.3g})  "
              f"H2 {r['H2']['standardized_effect']:+.3f} SD (q {r['H2']['q_value']:.3g})")
    t = report["trend"]
    print(f"gap change per year {t['gap_change_per_year']:+.4f} (SE {t['std_error']:.4f}, p {t['p_value']:.3g})")
    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
