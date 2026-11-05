"""CP4: first 5 theories end to end, plus the ratio x need sweep. Needs the panel (scripts/build_panel.py)
and either cached LLM responses (--replay) or ANTHROPIC_API_KEY."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT
from theorylab.llm import LLMClient
from theorylab.orchestrator import Orchestrator
from theorylab.sweep import run_sweep
from theorylab.theory_run import RUNS_DIR, load_panel, write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    theories = json.loads((REPO_ROOT / "benchmark" / "cp4_theories.json").read_text(encoding="utf-8"))
    llm = LLMClient(replay_only=args.replay)
    panel = load_panel()
    summary = Orchestrator(llm, panel).run_theories(theories, RUNS_DIR / "cp4")
    print(json.dumps(summary, indent=2))
    if args.sweep:
        write_report(run_sweep(panel), RUNS_DIR / "cp4" / "ratio_need_sweep.json")


if __name__ == "__main__":
    main()
