"""CP5 V4: rerun committed theory specs under the current gate, with no LLM call.

    python scripts/rerun_specs.py                                  # cp4 -> cp5_reruns
    python scripts/rerun_specs.py --from cp4 --to cp5_reruns [--permutations 999]

Why not `run_cp4.py --replay`: the LLM cache is keyed on the exact prompt, and the
prompts changed twice since CP4: the TA import (f825052) added `outcome_rule` to
specs/ratings_fairness.json, which is the example spec in every Planner prompt, and
CP5 V3 added `nonprofit_density` to the column list the Scout sees. So every CP4
cache key misses. The committed <id>.spec.json files ARE the Planner's CP4 output,
so rerunning them is the same theory with the same plan; only the gate is new.

Each spec goes through spec.lock_gate (so the CP5 outcome rule applies), then the same
Orchestrator.execute as a planned theory (run_spec, drill-down for survivors, per-year
results and trend), then one BH family over the reruns, as run_cp4.py did. Writes
<id>.json, <id>.spec.json and summary.json (with seconds per theory) to
data/output/theory_runs/<to>/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.gate import apply_family  # noqa: E402
from theorylab.llm import LLMClient  # noqa: E402
from theorylab.orchestrator import Orchestrator, _trend_summary  # noqa: E402
from theorylab.spec import lock_gate, validate_spec  # noqa: E402
from theorylab.theory_run import RUNS_DIR, available_columns, write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="cp4")
    ap.add_argument("--to", dest="dst", default="cp5_reruns")
    ap.add_argument("--permutations", type=int, default=None, help="default: the gate policy's 999")
    ap.add_argument("--no-by-year", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated theory ids (e.g. a README quickstart row)")
    args = ap.parse_args()
    src, dst = RUNS_DIR / args.src, RUNS_DIR / args.dst
    if dst.resolve() == src.resolve():
        raise SystemExit("--to must differ from --from; the source runs would be overwritten")
    only = set(args.only.split(",")) if args.only else None
    columns = available_columns()
    specs = []
    for path in sorted(src.glob("*.spec.json")):
        if only and path.name.removesuffix(".spec.json") not in only:
            continue
        spec, overrides = lock_gate(json.loads(path.read_text(encoding="utf-8")))
        spec["rerun_of"] = f"theory_runs/{args.src}/{path.name}"
        spec["rerun_gate_changes"] = [o["field"] for o in overrides]
        errors = validate_spec(spec, columns)
        if errors:
            raise SystemExit(f"{path.name}: {errors}")
        specs.append(spec)
    orch = Orchestrator(LLMClient(replay_only=True), use_scout=False,
                        n_permutations=args.permutations, by_year=not args.no_by_year)
    panel = orch.panel_for(specs)
    reports, seconds = [], {}
    for spec in specs:
        t0 = time.time()
        reports.append(orch.execute(spec, panel))
        seconds[spec["theory_id"]] = round(time.time() - t0, 1)
        print(f"{spec['theory_id']:34s} {reports[-1]['verdict']:11s} ({seconds[spec['theory_id']]} s)",
              flush=True)
    family = apply_family(reports, {s["theory_id"]: int(s["expected_sign"]) for s in specs})
    dst.mkdir(parents=True, exist_ok=True)
    for spec, rep in zip(specs, family):
        write_report(rep, dst / f"{rep['theory_id']}.json")
        (dst / f"{rep['theory_id']}.spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    summary = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "rerun_of": f"theory_runs/{args.src}", "llm_calls": 0, "seconds_per_theory": seconds,
               "theories": [{"theory_id": r["theory_id"], "verdict": r["verdict"],
                             "verdict_within_theory": r["verdict_within_theory"],
                             "outcome_rule_applied": r["population"]["outcome_rule"].get("applied"),
                             **_trend_summary(r.get("by_year"))} for r in family]}
    write_report(summary, dst / "summary.json")
    for t in summary["theories"]:
        print(f"{t['theory_id']:34s} family verdict {t['verdict']:11s} rule applied {t['outcome_rule_applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
