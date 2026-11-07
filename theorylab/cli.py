"""`python -m theorylab.cli test "<theory>"` (CP4): runs the chain for one theory."""

from __future__ import annotations

import argparse
import json

from theorylab.llm import LLMClient
from theorylab.orchestrator import Orchestrator
from theorylab.theory_run import RUNS_DIR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="theorylab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test", help="test one theory in plain language")
    t.add_argument("theory")
    t.add_argument("--replay", action="store_true", help="offline: use only cached LLM responses")
    t.add_argument("--no-scout", action="store_true", help="Planner reads catalog fields directly")
    t.add_argument("--drill", default="state", choices=["state", "sector", "size", "none"])
    t.add_argument("--permutations", type=int, default=None)
    t.add_argument("--no-by-year", action="store_true", help="skip the per-year results and trend")
    args = ap.parse_args(argv)
    orch = Orchestrator(LLMClient(replay_only=args.replay), use_scout=not args.no_scout,
                        drill_by=None if args.drill == "none" else args.drill,
                        n_permutations=args.permutations, by_year=not args.no_by_year)
    summary = orch.run_theories([args.theory], out_dir=RUNS_DIR)
    print(json.dumps(summary, indent=2))
    return 1 if summary["plan_failures"] or summary["run_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
