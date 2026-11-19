"""CP4 V6: check that every committed theory verdict follows from its own numbers.

For each run in data/output/theory_runs/cp4/ (<id>.json with <id>.spec.json):
  - gate:     was it run under the current gate policy (spec.GATE_POLICY: alpha, floor,
              999 permutations, whole county histories, studentized statistic)? A run
              from before 7ab7337 is "stale" and must be rerun.
  - family:   recompute the family q-values from the stored permutation p-values of
              all runs together (gate.apply_family), and compare with the stored q's.
  - verdict:  recompute the verdict from those numbers and compare with the stored one.
  - gold:     does the spec agree with the theory's gold answer (benchmark.review_spec)?
  - by year:  which years pass, and the trend p (reported, never changes a verdict).

Writes data/output/theory_runs/cp4/review.json and prints one line per theory, ready
for the report (V6).

CP5 V7: a folder written by run_benchmark (it holds benchmark.json) is reviewed as a
benchmark: gold answers come from benchmark/cp5_theories.json by entry id, the q-values
are recomputed in the two BH families (literature; controls) with
benchmark_run.apply_families, and each report must say it was judged in its own
family. Entries left at the screening count (--screen) are not flagged for it; the
ones rerun at full count must be at the policy's count. CP5 reruns of CP4 theories
(theory_runs/cp5_reruns) are reviewed like CP4, with each CP4 verdict alongside.

Usage:
    python scripts/review_theory_runs.py [--dir data/output/theory_runs/cp4]
    python scripts/review_theory_runs.py --dir data/output/theory_runs/cp5      # the benchmark
    python scripts/review_theory_runs.py --dir data/output/theory_runs/cp5_reruns
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.benchmark import load_theories, review_spec  # noqa: E402
from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.gate import apply_family  # noqa: E402
from theorylab.spec import GATE_POLICY  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "data" / "output" / "theory_runs" / "cp4"


def gate_problems(report: dict) -> list:
    v = report["validator"]
    want = GATE_POLICY
    out = []
    for key, got in (("alpha", v.get("alpha")), ("effect_floor_sd", v.get("effect_floor_sd")),
                     ("n_permutations", v.get("n_permutations"))):
        if got != want[key]:
            out.append(f"{key} {got} (policy {want[key]})")
    perm = want["permutation"]
    if bool(v.get("permutation_keep_paths")) != perm["keep_paths"]:
        out.append("not whole county histories")
    if bool(v.get("permutation_studentized")) != perm.get("studentized", False):
        out.append("raw-coefficient permutation (before 7ab7337): rerun")
    return out


def gold_for(spec: dict, theories: list) -> tuple[str | None, dict | None]:
    """The benchmark theory a run belongs to. The Planner rewrites the theory text, so
    match on the exact text first, then on the exposure (unique per theory), then the outcome."""
    for key in ("text", "exposure", "outcome"):
        for t in theories:
            if key == "text" and spec.get("theory") in (t["text"], t.get("paraphrase")):
                return t["id"], t["gold"]
            if key != "text" and spec.get(key) == t["gold"][key]:
                return t["id"], t["gold"]
    return None, None


def _benchmark_context(run_dir: Path):
    """(kinds, screened ids, benchmark entries) for a run_benchmark folder, else None."""
    path = run_dir / "benchmark.json"
    if not path.exists():
        return None
    from theorylab.benchmark_run import load_entries
    bench = json.loads(path.read_text(encoding="utf-8"))
    entries = load_entries()
    kinds = {e["id"]: e["kind"] for e in entries}
    screen = (bench.get("permutations") or {}).get("screen")
    rerun = set(bench.get("rerun_at_full_permutations") or [])
    screened = {i for i in kinds if screen and i not in rerun}
    return kinds, screened, entries, screen


def review(run_dir: Path) -> dict:
    bench = _benchmark_context(run_dir)
    theories = bench[2] if bench else load_theories()
    pairs = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name.endswith(".spec.json") or path.name in ("summary.json", "review.json", "benchmark.json"):
            continue
        spec_path = path.with_name(path.stem + ".spec.json")
        if not spec_path.exists():
            continue
        pairs.append((json.loads(path.read_text()), json.loads(spec_path.read_text())))
    if not pairs:
        raise SystemExit(f"no runs with specs in {run_dir}")
    reports = [r for r, _ in pairs]
    signs = {r["theory_id"]: int(s["expected_sign"]) for r, s in pairs}
    # recompute from the stored permutation p-values; the stored q's are what the run claimed
    unjudged = [{**r, "verdict": r.get("verdict_within_theory", r["verdict"])} for r in reports]
    if bench:
        from theorylab.benchmark_run import FAMILIES, apply_families
        kinds = bench[0]
        by_id = apply_families({r["theory_id"]: r for r in unjudged}, signs, kinds)
        redone = [by_id[r["theory_id"]] for r in reports]
    else:
        redone = apply_family(unjudged, signs)
    rows = []
    for (r, s), f in zip(pairs, redone):
        if bench:
            gid = r["theory_id"]
            gold = next((e["gold"] for e in theories if e["id"] == gid), None)
        else:
            gid, gold = gold_for(s, theories)
        problems = gate_problems(r)
        if bench and gid in bench[1]:  # left at the screening count by design
            problems = [x for x in problems if not x.startswith(f"n_permutations {bench[3]} ")]
        extra = {}
        if bench:
            fam = next(n for n, members in FAMILIES.items() if bench[0][gid] in members)
            extra = {"kind": bench[0][gid], "family_expected": fam,
                     "family_stored": r["validator"].get("family"),
                     "own_family": r["validator"].get("family") == fam}
        cp4 = (REPO_ROOT / "data" / "output" / "theory_runs" / "cp4" / f"{r['theory_id']}.json")
        if "cp5_reruns" in run_dir.as_posix() and cp4.exists():
            c = json.loads(cp4.read_text(encoding="utf-8"))
            extra["cp4"] = {"verdict": c["verdict"], "H1_standardized": c["H1"]["standardized_effect"],
                            "H2_standardized": c["H2"]["fit"]["standardized_effect"],
                            "outcome_sd": c.get("outcome_sd")}
            extra["outcome_rule"] = r["population"].get("outcome_rule")
        stored_q = (r["H1"].get("family_q_value"), r["H2"].get("family_q_value"))
        q_ok = all(a is not None and abs(a - b) < 1e-9 for a, b in
                   zip(stored_q, (f["H1"]["family_q_value"], f["H2"]["family_q_value"])))
        by = r.get("by_year") or {}
        rows.append({
            "theory_id": r["theory_id"], "gold_id": gid,
            "verdict": r["verdict"], "verdict_recomputed": f["verdict"],
            "verdict_follows": r["verdict"] == f["verdict"], "reasons": f["reasons"],
            "family_q_matches": q_ok,
            "H1_standardized": r["H1"]["standardized_effect"], "H1_family_q": f["H1"]["family_q_value"],
            "H2_standardized": r["H2"]["fit"]["standardized_effect"],
            "H2_family_q": f["H2"]["family_q_value"],
            "replicated": r["replication"]["replicated"],
            "folds_with_expected_sign": r["replication"].get("folds_with_expected_sign"),
            "gate_problems": problems, **extra,
            "gold_disagreements": review_spec(s, gold) if gold else ["no gold answer found"],
            "years_H1_holds": by.get("years_H1_holds"), "trend_p": (by.get("trend") or {}).get("p_value"),
        })
    out = {"run_dir": (run_dir.relative_to(REPO_ROOT).as_posix() if run_dir.is_relative_to(REPO_ROOT)
                       else str(run_dir)), "family_size": 2 * len(rows),
            "all_follow": all(x["verdict_follows"] and x["family_q_matches"] for x in rows),
            "all_current_gate": not any(x["gate_problems"] for x in rows), "theories": rows}
    if bench:
        out["family_size"] = {n: 2 * sum(x["family_expected"] == n for x in rows) for n in FAMILIES}
        out["all_in_own_family"] = all(x["own_family"] for x in rows)
    return out


def line(x: dict) -> str:
    ok = "Follows from the numbers." if x["verdict_follows"] and x["family_q_matches"] else \
        f"DOES NOT follow: recomputed {x['verdict_recomputed']}, q match {x['family_q_matches']}."
    gate = "" if not x["gate_problems"] else f" Gate: {'; '.join(x['gate_problems'])}."
    gold = "" if not x["gold_disagreements"] else f" Spec vs gold: {'; '.join(x['gold_disagreements'])}."
    trend = f", trend p {x['trend_p']:.2f}" if x.get("trend_p") is not None else ""
    if "cp4" in x:
        c = x["cp4"]
        trend += (f". CP4: {c['verdict']}, H1 {c['H1_standardized']:+.3f} SD, H2 {c['H2_standardized']:+.3f} SD"
                  f" (CP4 verdict kept)")
    if "own_family" in x and not x["own_family"]:
        gate += f" Judged in {x['family_stored']!r}, should be {x['family_expected']!r}."
    return (f"- **{x['gold_id'] or x['theory_id']}** (`{x['theory_id']}`): {x['verdict']}. {ok} "
            f"H1 {x['H1_standardized']:+.3f} SD (family q {x['H1_family_q']:.3g}), H2 "
            f"{x['H2_standardized']:+.3f} SD (q {x['H2_family_q']:.3g}), replicated "
            f"{x['replicated']} ({x['folds_with_expected_sign']}/5 folds); years passing H1 "
            f"{x['years_H1_holds'] or 'none'}{trend}.{gate}{gold}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    run_dir = Path(args.dir).resolve()
    out = review(run_dir)
    (run_dir / "review.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for x in out["theories"]:
        print(line(x))
    print(f"\nall verdicts follow: {out['all_follow']}; all on the current gate: {out['all_current_gate']}"
          + (f"; every entry in its own family: {out['all_in_own_family']}" if "all_in_own_family" in out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
