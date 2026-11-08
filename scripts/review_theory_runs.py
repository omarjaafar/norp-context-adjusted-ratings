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

Usage:
    python scripts/review_theory_runs.py [--dir data/output/theory_runs/cp4]
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


def review(run_dir: Path) -> dict:
    theories = load_theories()
    pairs = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name.endswith(".spec.json") or path.name in ("summary.json", "review.json"):
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
    redone = apply_family([{**r, "verdict": r.get("verdict_within_theory", r["verdict"])}
                           for r in reports], signs)
    rows = []
    for (r, s), f in zip(pairs, redone):
        gid, gold = gold_for(s, theories)
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
            "gate_problems": gate_problems(r),
            "gold_disagreements": review_spec(s, gold) if gold else ["no gold answer found"],
            "years_H1_holds": by.get("years_H1_holds"), "trend_p": (by.get("trend") or {}).get("p_value"),
        })
    return {"run_dir": run_dir.relative_to(REPO_ROOT).as_posix(), "family_size": 2 * len(rows),
            "all_follow": all(x["verdict_follows"] and x["family_q_matches"] for x in rows),
            "all_current_gate": not any(x["gate_problems"] for x in rows), "theories": rows}


def line(x: dict) -> str:
    ok = "Follows from the numbers." if x["verdict_follows"] and x["family_q_matches"] else \
        f"DOES NOT follow: recomputed {x['verdict_recomputed']}, q match {x['family_q_matches']}."
    gate = "" if not x["gate_problems"] else f" Gate: {'; '.join(x['gate_problems'])}."
    gold = "" if not x["gold_disagreements"] else f" Spec vs gold: {'; '.join(x['gold_disagreements'])}."
    trend = f", trend p {x['trend_p']:.2f}" if x.get("trend_p") is not None else ""
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
    print(f"\nall verdicts follow: {out['all_follow']}; all on the current gate: {out['all_current_gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
