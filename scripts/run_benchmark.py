"""CP5: the 20-entry benchmark (10 literature theories, 5 positive and 5 negative controls).

    python scripts/run_benchmark.py --replay                      # cached LLM responses, full gate (999)
    python scripts/run_benchmark.py --screen 199 --rerun-q 0.25   # screen at 199, rerun near-misses at 999
    python scripts/run_benchmark.py --only ratings_fairness,pos_poverty_programs

Needs the panel (scripts/build_panel.py) and cached responses (--replay) or ANTHROPIC_API_KEY.
Writes data/output/benchmark_cp5.json (metrics: plan validity, control recovery, false-positive
rate) and one report and spec per entry in data/output/theory_runs/cp5/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.benchmark_run import OUT_PATH, load_entries, run_benchmark  # noqa: E402
from theorylab.llm import LLMClient  # noqa: E402
from theorylab.orchestrator import Orchestrator  # noqa: E402
from theorylab.theory_run import RUNS_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="use only cached LLM responses")
    ap.add_argument("--variant", choices=("text", "paraphrase"), default="text")
    ap.add_argument("--permutations", type=int, default=None, help="default: the gate policy's 999")
    ap.add_argument("--screen", type=int, default=None,
                    help="first pass at this many permutations; near-misses rerun at the full count")
    ap.add_argument("--rerun-q", type=float, default=0.25)
    ap.add_argument("--only", default=None, help="comma-separated entry ids")
    args = ap.parse_args()

    entries = load_entries()
    if args.only:
        wanted = set(args.only.split(","))
        entries = [e for e in entries if e["id"] in wanted]
    orch = Orchestrator(LLMClient(replay_only=args.replay), by_year=False, drill_by="state",
                        n_permutations=args.permutations)
    result = run_benchmark(orch, None, entries, variant=args.variant, n_permutations=args.permutations,
                           screen_permutations=args.screen, rerun_q=args.rerun_q,
                           out_dir=RUNS_DIR / "cp5")
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    for row in result["table"]:
        print(f"{row['id']:30s} {row['kind']:17s} {row['verdict']:12s} "
              f"H1 {row['H1_std']:+.3f} q {row['H1_family_q']:.3g}  H2 {row['H2_std']:+.3f} "
              f"q {row['H2_family_q']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
