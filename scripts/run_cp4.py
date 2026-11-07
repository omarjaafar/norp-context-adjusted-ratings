"""CP4: first 5 theories end to end, plus the ratio x need sweep. Needs the panel (scripts/build_panel.py)
and either cached LLM responses (--replay) or ANTHROPIC_API_KEY."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.llm import LLMClient
from theorylab.orchestrator import Orchestrator
from theorylab.sweep import run_sweep
from theorylab.benchmark import load_theories, runnable
from theorylab.theory_run import RUNS_DIR, available_columns, load_panel, write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--permutations", type=int, default=None, help="default: the gate policy's 999")
    ap.add_argument("--no-by-year", action="store_true", help="skip the per-year results and trend")
    args = ap.parse_args()
    columns = available_columns()
    theories = [t["text"] for t in load_theories() if runnable(t, columns)]
    llm = LLMClient(replay_only=args.replay)
    orch = Orchestrator(llm, n_permutations=args.permutations, by_year=not args.no_by_year)
    summary = orch.run_theories(theories, RUNS_DIR / "cp4")  # loads only the columns the specs need
    print(json.dumps(summary, indent=2))
    if args.sweep:
        write_report(run_sweep(load_panel()), RUNS_DIR / "cp4" / "ratio_need_sweep.json")


if __name__ == "__main__":
    main()
