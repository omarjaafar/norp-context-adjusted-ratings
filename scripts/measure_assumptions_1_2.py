"""First measurement of Assumptions 1 and 2 (CP4).

A1: share of theory texts (each theory and its paraphrase) that the Scout and Planner
    turn into a spec that passes the schema check AND agrees with the gold answer
    (outcome, exposure, exposure type, sign, unit; theorylab.benchmark.review_spec).
A2: tokens, cost and wall time of the planning calls, and the cost projected to the
    20-theory benchmark. Everything after the spec is deterministic Python.
Writes data/output/assumptions_1_2.json. Needs ANTHROPIC_API_KEY, or --replay with a
committed cache (the cost is then recomputed from the tokens stored in the cache).
Theories whose gold columns are not in the panel yet are listed as skipped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.benchmark import load_theories, review_spec, runnable  # noqa: E402
from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.llm import LLMClient  # noqa: E402
from theorylab.orchestrator import Orchestrator  # noqa: E402
from theorylab.theory_run import available_columns  # noqa: E402

A1_NEEDED = 17 / 20     # Assumption 1: at least 17 of 20 theories
A2_LIMIT_USD = 15.0     # Assumption 2: a full benchmark run under $15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="use only cached LLM responses")
    args = ap.parse_args()
    llm = LLMClient(replay_only=args.replay)
    orch = Orchestrator(llm, panel=None)
    columns = available_columns()
    rows, skipped, t0 = [], [], time.time()
    for theory in load_theories():
        if not runnable(theory, columns):
            skipped.append(theory["id"])
            continue
        for variant in ("text", "paraphrase"):
            row = {"theory_id": theory["id"], "variant": variant}
            try:
                spec = orch.plan(theory[variant])
                row.update(schema_valid=True, review=review_spec(spec, theory["gold"]),
                           gate_overrides=spec.get("gate_overrides", []))
            except ValueError as exc:
                row.update(schema_valid=False, error=str(exc)[:500])
            row["passes"] = bool(row["schema_valid"] and not row.get("review"))
            rows.append(row)
    seconds = time.time() - t0
    n = len(rows)
    cost = llm.usage.cost_usd(llm.model)
    out = {
        "model": llm.model,
        "replayed": args.replay,
        "assumption_1": {"texts": n, "schema_valid": sum(r["schema_valid"] for r in rows),
                         "pass_schema_and_review": sum(r["passes"] for r in rows),
                         "rate": (sum(r["passes"] for r in rows) / n) if n else None,
                         "needed_rate": A1_NEEDED, "skipped_theories": skipped, "rows": rows},
        "assumption_2": {"live_calls": llm.usage.calls, "cache_hits": llm.usage.cache_hits,
                         "input_tokens": llm.usage.input_tokens + llm.usage.cached_input_tokens,
                         "output_tokens": llm.usage.output_tokens + llm.usage.cached_output_tokens,
                         "planning_cost_usd": cost,
                         "planning_seconds_this_run": seconds,
                         "projected_20_theories_usd": cost / n * 20 if n else None,  # one plan per theory
                         "limit_usd": A2_LIMIT_USD,
                         "note": "cost counts every planning call, cached or not, at the model's list price; "
                                 "the statistics after the spec make no LLM calls"},
    }
    path = REPO_ROOT / "data" / "output" / "assumptions_1_2.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["assumption_1"].items() if k != "rows"}, indent=2))
    print(json.dumps(out["assumption_2"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
