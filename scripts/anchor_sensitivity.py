"""Checkpoint 3: does the ratings-theory verdict survive other rating anchors?

    python scripts/anchor_sensitivity.py                 # 199 permutations per variant
    python scripts/anchor_sensitivity.py --n-permutations 999

Reruns specs/ratings_fairness.json once per variant in
theorylab/sensitivity.py::ANCHOR_VARIANTS (each "full marks" anchor moved one
at a time, plus a four-ratios-required rating) and writes
data/output/anchor_sensitivity.json.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.sensitivity import run_anchor_sensitivity  # noqa: E402
from theorylab.spec import load_spec, spec_columns  # noqa: E402
from theorylab.theory_run import (PANEL_PATH, RATING_INPUTS, available_columns,  # noqa: E402
                                  load_panel, write_report)

OUT = REPO_ROOT / "data" / "output" / "anchor_sensitivity.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "specs" / "ratings_fairness.json"))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--n-permutations", type=int, default=199)
    args = ap.parse_args()

    spec = load_spec(args.spec, available_columns())
    panel = load_panel(args.panel, columns=spec_columns(spec) | {"ein"} | set(RATING_INPUTS))
    report = run_anchor_sensitivity(panel, spec, n_permutations=args.n_permutations)
    report["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    report["n_permutations"] = args.n_permutations
    write_report(report, OUT)
    for name, row in report["variants"].items():
        print(f"{name:28s} {row['verdict']:12s} H1 {row['H1_standardized']:+.3f} SD  "
              f"H2 {row['H2_standardized']:+.3f} SD")
    print(f"verdict stable: {report['verdict_stable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
