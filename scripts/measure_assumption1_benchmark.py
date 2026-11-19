"""CP5: Assumption 1 on theories the agents have never seen (the 20 benchmark entries).

CP4 measured A1 on the 10 texts the Planner was tuned against, so 9 of 10 was not independent
evidence. Here the Planner and Scout see each of the 20 benchmark entries (text and
paraphrase, 40 texts) for the first time; nothing in the prompts or code was changed after
reading them. For each text: schema-valid, agrees with the gold answer (benchmark.review_spec),
and the H2 controls the Planner chose (CP4 open question: the review did not check them), listed
for a person to read. The first pass is the number; later runs go to a separate file.

Writes data/output/assumption_1_unseen.json. Needs ANTHROPIC_API_KEY, or --replay.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.benchmark import review_spec  # noqa: E402
from theorylab.benchmark_run import load_entries  # noqa: E402
from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.llm import LLMClient  # noqa: E402
from theorylab.orchestrator import Orchestrator  # noqa: E402

A1_NEEDED = 0.85  # Assumption 1: under 85% means a template-based planner for those theory types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "output" / "assumption_1_unseen.json"))
    args = ap.parse_args()
    llm = LLMClient(replay_only=args.replay)
    orch = Orchestrator(llm, panel=None)
    rows = []
    for e in load_entries():
        for variant in ("text", "paraphrase"):
            row = {"id": e["id"], "kind": e["kind"], "variant": variant}
            try:
                spec = orch.plan(e[variant])
                h2 = next(h for h in spec["hypotheses"] if h["id"] == "H2")
                row.update(schema_valid=True, review=review_spec(spec, e["gold"]),
                           h2_controls=h2.get("controls", []), h2_absorb=h2.get("absorb", []),
                           gate_overrides=spec.get("gate_overrides", []))
            except ValueError as exc:
                row.update(schema_valid=False, error=str(exc)[:500])
            row["passes"] = bool(row["schema_valid"] and not row.get("review"))
            rows.append(row)
    n = len(rows)
    by_kind = {k: {"passes": sum(r["passes"] for r in rows if r["kind"] == k),
                   "texts": sum(r["kind"] == k for r in rows)}
               for k in ("literature", "positive_control", "negative_control")}
    out = {"model": llm.model, "replayed": args.replay, "texts": n,
           "schema_valid": sum(r["schema_valid"] for r in rows),
           "pass_schema_and_review": sum(r["passes"] for r in rows),
           "rate": sum(r["passes"] for r in rows) / n, "needed_rate": A1_NEEDED, "by_kind": by_kind,
           "usage": vars(llm.usage).copy(), "cost_usd": llm.usage.cost_usd(llm.model), "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
