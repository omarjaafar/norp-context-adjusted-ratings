"""CP4 V5 (CP3 feedback F4, audit W12): does the one-level drill-down find the right stratum?

Nothing on the real panel survives the gate, so the drill-down never runs on real
data. This script plants known effects into the ratings theory's real population
and checks what drill_down reports:

  A. state:   a 0.3 SD rating penalty in high-need counties of one large state only
              -> drill by state should name that state and no other
  B. sector:  a 0.3 SD penalty in high-need counties for one NORP category only
              -> drill by sector should name that category and no other
  C. national: a 0.25 SD penalty everywhere -> drill by state, sector and size: every
              tested stratum should have the expected sign

It also runs the national gate on A and B, to see whether an effect confined to one
stratum would be drilled at all (the drill-down only runs for survivors), and
counts the states skipped by the 30-county rule (drilldown.MIN_UNITS).

For A and B the drill-down is called with the verdict set to "weak" whatever the
national verdict was, so that the stratum search itself is tested.

Writes data/output/drilldown_check.json.

Usage:
    python scripts/drilldown_check.py [--n-permutations 199]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.controls import population  # noqa: E402
from theorylab.drilldown import MIN_UNITS, drill_down  # noqa: E402
from theorylab.executor import to_float, to_jsonable  # noqa: E402
from theorylab.spec import load_spec, lock_gate, spec_columns  # noqa: E402
from theorylab.theory_run import available_columns, load_panel, run_spec  # noqa: E402

OUT = REPO_ROOT / "data" / "output" / "drilldown_check.json"


def plant(pop: pd.DataFrame, spec: dict, effect_sd: float, where: pd.Series) -> pd.DataFrame:
    outcome, sign = spec["outcome"], int(spec["expected_sign"])
    y = to_float(pop[outcome])
    exposed = to_float(pop[spec["exposure"]]).eq(1.0) & where
    out = pop.copy()
    out[outcome] = y + sign * effect_sd * float(y.std(ddof=1)) * exposed
    return out


def national(pop, spec, n_perm) -> dict:
    r = run_spec(pop, spec, n_permutations=n_perm, include_robustness=False)
    return {"verdict": r["verdict"], "H1_standardized": r["H1"]["standardized_effect"],
            "H1_q": r["H1"]["q_value"], "H2_standardized": r["H2"]["fit"]["standardized_effect"],
            "H2_q": r["H2"]["q_value"], "replicated": r["replication"]["replicated"]}


def summarize(d: dict, target: str | None, sign: int) -> dict:
    strata = d.get("strata", [])
    out = {"family_size": d.get("family_size"), "survivors": d.get("survivors"),
           "skipped": len(d.get("skipped", [])),
           "tested_with_expected_sign": sum(s["same_sign"] for s in strata),
           "tested": len(strata)}
    if target is not None:
        out["target"] = target
        out["target_tested"] = any(s["level"] == target for s in strata)
        out["finds_target_only"] = d.get("survivors") == [target]
        t = next((s for s in strata if s["level"] == target), None)
        out["target_stratum"] = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=199)
    ap.add_argument("--effect-sd", type=float, default=0.3)
    args = ap.parse_args()
    n_perm = args.n_permutations

    spec = lock_gate(load_spec(REPO_ROOT / "specs" / "ratings_fairness.json", available_columns()))[0]
    panel = load_panel(columns=spec_columns(spec) | {"ein", "state"})
    pop, pop_log = population(panel, spec)
    sign = int(spec["expected_sign"])
    exposed = to_float(pop[spec["exposure"]]).eq(1.0)

    # the state and the category with the most high-need counties / exposed rows
    counties_by_state = pop.groupby("state")["county_fips"].nunique()
    hn_counties = pop[exposed].groupby("state")["county_fips"].nunique()
    target_state = str(hn_counties.idxmax())
    target_cat = str(pop.loc[exposed, "category"].value_counts().idxmax())

    t0 = time.perf_counter()
    result = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "theory_id": spec["theory_id"], "n_permutations": n_perm,
              "permutation": spec["validator"].get("permutation"),
              "min_units": MIN_UNITS, "planted_effect_sd": args.effect_sd,
              "population_rows": pop_log["rows_out"],
              "states": int(counties_by_state.size),
              "states_below_min_units": sorted(counties_by_state[counties_by_state < MIN_UNITS]
                                               .index.astype(str).tolist()),
              "rows_in_states_below_min_units": int(
                  pop["state"].isin(counties_by_state[counties_by_state < MIN_UNITS].index).sum()),
              "scenarios": {}}
    result["states_skipped_by_min_units"] = len(result["states_below_min_units"])

    def weak(report_like):  # force the survivor rule open, to test the search itself
        return {**report_like, "verdict": "weak"}

    scen = {
        "A_one_state": (target_state, "state", plant(pop, spec, args.effect_sd,
                                                     pop["state"].astype(str) == target_state)),
        "B_one_sector": (target_cat, "sector", plant(pop, spec, args.effect_sd,
                                                     pop["category"].astype(str) == target_cat)),
    }
    for name, (target, by, planted) in scen.items():
        nat = national(planted, spec, n_perm)
        d = drill_down(planted, spec, weak(nat), by, n_permutations=n_perm)
        result["scenarios"][name] = {"by": by, "national_gate": nat,
                                     "would_be_drilled": nat["verdict"] in ("supported", "weak"),
                                     **summarize(d, target, sign), "drill_down": d}
        print(name, target, json.dumps(summarize(d, target, sign), default=str)[:400], flush=True)

    planted = plant(pop, spec, 0.25, pd.Series(True, index=pop.index))
    nat = national(planted, spec, n_perm)
    result["scenarios"]["C_national"] = {"national_gate": nat, "by": {}}
    for by in ("state", "sector", "size"):
        d = drill_down(planted, spec, weak(nat), by, n_permutations=n_perm)
        result["scenarios"]["C_national"]["by"][by] = {**summarize(d, None, sign), "drill_down": d}
        print("C", by, json.dumps(summarize(d, None, sign), default=str)[:300], flush=True)

    result["seconds"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(to_jsonable(result), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
