"""Checkpoint 3: negative and positive controls through the Validator gate on the real panel.

    python scripts/run_controls.py                       # 20 shuffled seeds, 199 permutations
    python scripts/run_controls.py --seeds 10            # if short on time
    python scripts/run_controls.py --family              # CP4: 8 placebos + 1 planted, one BH family

Negative control: each county gets another county's whole exposure history (a placebo),
so every run should be unsupported. Positive controls: a gap is planted at 0.25 SD (should
be supported) and 0.05 SD (below the 0.1 SD floor, should be unsupported). See
theorylab/controls.py.

Needs data/derived/org_year_panel.csv.gz (python scripts/build_panel.py).
Writes data/output/controls.json (--family: data/output/controls_family.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.controls import family_controls, run_controls  # noqa: E402
from theorylab.spec import load_spec, spec_columns  # noqa: E402
from theorylab.theory_run import PANEL_PATH, available_columns, load_panel, write_report  # noqa: E402

OUT = REPO_ROOT / "data" / "output" / "controls.json"
OUT_FAMILY = REPO_ROOT / "data" / "output" / "controls_family.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "specs" / "ratings_fairness.json"))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--n-permutations", type=int, default=199)
    ap.add_argument("--seeds", type=int, default=20, help="number of shuffled negative-control runs")
    ap.add_argument("--family", action="store_true",
                    help="8 placebo theories + 1 planted 0.25 SD theory through one BH family")
    ap.add_argument("--placebos", type=int, default=8)
    args = ap.parse_args()

    spec = load_spec(args.spec, available_columns())
    panel = load_panel(args.panel, columns=spec_columns(spec) | {"ein"})
    if args.family:
        report = family_controls(panel, spec, placebo_seeds=range(2000, 2000 + args.placebos),
                                 n_permutations=args.n_permutations)
        write_report(report, OUT_FAMILY)
        for r in report["theories"]:
            print(f"{r['theory_id']:16s} within {r['verdict_within_theory']:12s} family "
                  f"{r['verdict_family']:12s} H1 {r['H1_standardized']:+.3f} SD q {r['H1_family_q']:.3g}"
                  f"  H2 {r['H2_standardized']:+.3f} SD q {r['H2_family_q']:.3g}")
        print(f"family false positives: {report['family_false_positives']} of {report['placebos']}; "
              f"planted survives: {report['planted']['survives']}")
        return 0
    seeds = range(1000, 1000 + args.seeds)
    report = run_controls(panel, spec, negative_seeds=seeds, n_permutations=args.n_permutations)
    write_report(report, OUT)
    neg = report["negative_control"]
    for r in neg["runs"]:
        print(f"negative seed {r['seed']}  {r['verdict']:12s} H1 {r['H1_standardized']:+.3f} SD "
              f"(q {r['H1_q']:.3g})  H2 {r['H2_standardized']:+.3f} SD (q {r['H2_q']:.3g})")
    print(f"negative: {neg['false_positives']} false positives in {neg['n_runs']} runs "
          f"{neg['verdict_counts']}")
    for p in report["positive_controls"]:
        print(f"positive {p['planted_effect_sd']:.2f} SD  {p['verdict']:12s} "
              f"(expected {p['expected_verdict']})  H1 {p['H1_standardized']:+.3f} SD  "
              f"H2 {p['H2_standardized']:+.3f} SD  replicated {p['replicated']}")
    print(f"all passed: {report['all_passed']}")
    print(f"wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
