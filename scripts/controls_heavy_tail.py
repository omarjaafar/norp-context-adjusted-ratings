"""CP5 V2 and V6: controls on heavy-tailed ratios, with and without the outcome rule.

V2 (CP4 feedback F4, audit X3). For two heavy-tailed outcomes, on the real panel:
    pos_unemployment_margin   operating_margin      x unemployment_rate   sign -1
    pos_poverty_leverage      liabilities_to_assets x poverty_rate        sign +1
four runs each: {0.25 SD planted, nothing planted} x {rule on, rule off}. The effect is
planted in the RAW outcome before the rule, at 0.25 x the outcome's SD after the rule on
the unplanted population (benchmark_run.prepare_population, docs/validator.md 2.8).
The finding to show: missed without the rule, recovered with it, and the unplanted
version stays unsupported. -> data/output/controls_heavy_tail.json

V6 (--family, audit X7). For the same two outcomes under the rule: 8 placebo theories
(exposure replaced by another county's whole history) and the planted one through one
BH family, as in CP4's controls_family.json. Shows the rule does not create false
positives. -> data/output/controls_family_heavy_tail.json

Specs are the sweep cells (sweep.cell_spec, fixed scheme, gate locked), so the gate is the
one every theory meets. Times are written into the output.

    python scripts/controls_heavy_tail.py [--n-permutations 199]
    python scripts/controls_heavy_tail.py --family [--placebos 8]
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.benchmark_run import prepare_population  # noqa: E402
from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.executor import to_jsonable  # noqa: E402
from theorylab.gate import apply_family  # noqa: E402
from theorylab.spec import spec_columns  # noqa: E402
from theorylab.sweep import cell_spec  # noqa: E402
from theorylab.theory_run import load_panel, run_spec  # noqa: E402

CONTROLS = [("pos_unemployment_margin", "operating_margin", "unemployment_rate", -1),
            ("pos_poverty_leverage", "liabilities_to_assets", "poverty_rate", 1)]
EFFECT_SD = 0.25
OUT = REPO_ROOT / "data" / "output" / "controls_heavy_tail.json"
OUT_FAMILY = REPO_ROOT / "data" / "output" / "controls_family_heavy_tail.json"


def summary(r: dict) -> dict:
    rule = r["population"].get("outcome_rule", {})
    return {"verdict": r["verdict"], "reasons": r["reasons"],
            "H1_standardized": r["H1"]["standardized_effect"], "H1_p": r["H1"]["permutation"]["p_value"],
            "H2_standardized": r["H2"]["fit"]["standardized_effect"], "H2_p": r["H2"]["permutation"]["p_value"],
            "replicated": r["replication"]["replicated"], "outcome_sd": r["outcome_sd"],
            "rule_applied": bool(rule.get("applied")), "sd_over_robust_sd_after": rule.get("sd_over_robust_sd_after")}


def no_rule(spec: dict) -> dict:
    s = copy.deepcopy(spec)
    s["validator"].pop("outcome_rule", None)
    return s


def entry(kind: str, **control) -> dict:
    return {"id": kind, "kind": kind, "control": control}


def run_v2(panel, n_perm: int) -> dict:
    out = {}
    for cid, outcome, exposure, sign in CONTROLS:
        spec = cell_spec(outcome, exposure, sign, n_permutations=n_perm)
        rows = {}
        for planted in (True, False):
            e = entry("positive_control", effect_sd=EFFECT_SD) if planted else entry("literature")
            pop, prep = prepare_population(e, spec, panel)  # raw outcome; planted before the rule
            for rule in (True, False):
                t0 = time.time()
                r = run_spec(pop, spec if rule else no_rule(spec), n_permutations=n_perm,
                             include_robustness=False)
                key = f"{'planted' if planted else 'no_effect'}_{'rule' if rule else 'no_rule'}"
                rows[key] = {**summary(r), "seconds": round(time.time() - t0, 1)}
                print(f"{cid:24s} {key:20s} {r['verdict']:11s} H1 {r['H1']['standardized_effect']:+.3f} SD"
                      f"  H2 {r['H2']['fit']['standardized_effect']:+.3f} SD  ({rows[key]['seconds']} s)", flush=True)
            if planted:
                rows["planting"] = prep["planted"]
        rows["finding_holds"] = (rows["planted_no_rule"]["verdict"] != "supported"
                                 and rows["planted_rule"]["verdict"] == "supported"
                                 and rows["no_effect_rule"]["verdict"] == "unsupported")
        out[cid] = {"outcome": outcome, "exposure": exposure, "expected_sign": sign, **rows}
    return out


def run_v6(panel, n_perm: int, placebos: int) -> dict:
    out = {}
    for cid, outcome, exposure, sign in CONTROLS:
        spec = cell_spec(outcome, exposure, sign, n_permutations=n_perm)
        reports, kinds, t0 = [], {}, time.time()
        for seed in range(2000, 2000 + placebos):
            pop, _ = prepare_population(entry("negative_control", placebo_seed=seed), spec, panel)
            r = run_spec(pop, spec, n_permutations=n_perm, include_robustness=False)
            r["theory_id"] = f"placebo_{seed}"
            reports.append(r)
            kinds[r["theory_id"]] = "placebo"
            print(f"{cid:24s} placebo {seed} {r['verdict']}", flush=True)
        pop, prep = prepare_population(entry("positive_control", effect_sd=EFFECT_SD), spec, panel)
        r = run_spec(pop, spec, n_permutations=n_perm, include_robustness=False)
        r["theory_id"] = f"planted_{EFFECT_SD:g}sd"
        reports.append(r)
        kinds[r["theory_id"]] = "planted"
        family = apply_family(reports, {tid: sign for tid in kinds})
        rows = [{"theory_id": f["theory_id"], "kind": kinds[f["theory_id"]],
                 "verdict_within_theory": f["verdict_within_theory"], "verdict_family": f["verdict"],
                 "H1_standardized": f["H1"]["standardized_effect"], "H1_family_q": f["H1"]["family_q_value"],
                 "H2_standardized": f["H2"]["fit"]["standardized_effect"], "H2_family_q": f["H2"]["family_q_value"],
                 "replicated": f["replication"]["replicated"],
                 "rule_applied": bool(f["population"]["outcome_rule"].get("applied"))} for f in family]
        pl = [x for x in rows if x["kind"] == "placebo"]
        planted_row = next(x for x in rows if x["kind"] == "planted")
        out[cid] = {"outcome": outcome, "exposure": exposure, "expected_sign": sign,
                    "family_size_hypotheses": 2 * len(rows), "placebos": len(pl),
                    "family_false_positives": sum(x["verdict_family"] != "unsupported" for x in pl),
                    "planted_survives": planted_row["verdict_family"] == "supported",
                    "planting": prep["planted"], "seconds": round(time.time() - t0, 1), "theories": rows}
        print(f"{cid}: {out[cid]['family_false_positives']} of {len(pl)} placebos not unsupported; "
              f"planted survives: {out[cid]['planted_survives']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=199)
    ap.add_argument("--family", action="store_true", help="V6: 8 placebos + 1 planted per outcome, one BH family")
    ap.add_argument("--placebos", type=int, default=8)
    args = ap.parse_args()
    cols = set()
    for _, o, x, s in CONTROLS:
        cols |= spec_columns(cell_spec(o, x, s))
    panel = load_panel(columns=sorted(cols | {"ein"}))
    t0 = time.time()
    body = (run_v6(panel, args.n_permutations, args.placebos) if args.family
            else run_v2(panel, args.n_permutations))
    report = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "what": ("V6: placebo family on heavy-tailed outcomes under the rule" if args.family else
                       "V2: heavy-tailed positive controls, planted raw at 0.25 post-rule SD, with and without the rule"),
              "n_permutations": args.n_permutations, "effect_sd": EFFECT_SD, "controls": body,
              "seconds_total": round(time.time() - t0, 1)}
    path = OUT_FAMILY if args.family else OUT
    path.write_text(json.dumps(to_jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)} in {report['seconds_total']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
