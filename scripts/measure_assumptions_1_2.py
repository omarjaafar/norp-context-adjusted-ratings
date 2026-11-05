"""First measurement of Assumptions 1 and 2 (CP4).

A1: share of the benchmark theories the Planner turns into a spec that passes the schema check.
A2: cost and wall time of one full run of the same theories, from the client's token counts.
Writes data/output/assumptions_1_2.json. Needs ANTHROPIC_API_KEY (or a warm cache for A1 only).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT
from theorylab.llm import LLMClient
from theorylab.orchestrator import Orchestrator
from theorylab.theory_run import available_columns


def main():
    theories = json.loads((REPO_ROOT / "benchmark" / "cp4_theories.json").read_text(encoding="utf-8"))
    llm = LLMClient()
    orch = Orchestrator(llm, panel=None)
    t0, valid, rows = time.time(), 0, []
    for text in theories:
        try:
            spec = orch.plan(text)
            valid += 1
            rows.append({"theory": text[:80], "valid": True, "theory_id": spec["theory_id"]})
        except ValueError as exc:
            rows.append({"theory": text[:80], "valid": False, "error": str(exc)[:300]})
    n = len(theories)
    out = {
        "assumption_1": {"theories": n, "valid_specs": valid, "rate": valid / n,
                         "needed_rate": 17 / 20, "rows": rows,
                         "note": "schema validity only; the manual review (variables, sign, unit) is separate"},
        "assumption_2": {"planning_calls": llm.usage.calls, "cache_hits": llm.usage.cache_hits,
                         "input_tokens": llm.usage.input_tokens, "output_tokens": llm.usage.output_tokens,
                         "planning_cost_usd": llm.usage.cost_usd(), "planning_seconds": time.time() - t0,
                         "projected_20_theories_usd": llm.usage.cost_usd() * 20 / n,
                         "limit_usd": 15},
        "columns_offered": len(available_columns()),
    }
    path = REPO_ROOT / "data" / "output" / "assumptions_1_2.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
