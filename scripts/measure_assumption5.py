"""Checkpoint 3: measure Assumption 5 (MixedLM fit time and convergence).

    python scripts/measure_assumption5.py                 # 2022 full-990 filers
    python scripts/measure_assumption5.py --year 2021 --threshold-seconds 600

Fits standard_rating ~ high_need + log_revenue + C(category) with a random
intercept per county on one tax year of the ratings-theory population, times
it, and records convergence. If the fit misses the threshold or does not
converge, it also fits the two fallbacks the register names (state fixed
effects with county random intercepts, and the two-stage county-mean model).
Writes data/output/assumption_5.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.checks import A5_THRESHOLD_SECONDS, assumption_5_check  # noqa: E402
from theorylab.executor import multilevel, to_jsonable, two_stage_group_means  # noqa: E402
from theorylab.spec import load_spec, spec_columns  # noqa: E402
from theorylab.theory_run import (PANEL_PATH, add_derived, available_columns,  # noqa: E402
                                  load_panel, select_population)

OUT = REPO_ROOT / "data" / "output" / "assumption_5.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "specs" / "ratings_fairness.json"))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--threshold-seconds", type=float, default=A5_THRESHOLD_SECONDS)
    args = ap.parse_args()

    spec = load_spec(args.spec, available_columns())
    spec = {**spec, "population": {**spec["population"], "years": [args.year]}}
    cols = spec_columns(spec) | {"ein", "state"}
    panel = add_derived(load_panel(args.panel, columns=cols), cols)
    pop, pop_log = select_population(panel, spec)
    outcome, exposure = spec["outcome"], spec["exposure"]

    fit = multilevel(pop, outcome, exposure, controls=["log_revenue"], categorical=["category"],
                     group="county_fips")
    check = assumption_5_check(fit.details, fit.n_obs, fit.n_groups, args.threshold_seconds)
    fallbacks = {}
    if not check["holds"]:
        try:
            fallbacks["state_fe_county_random_intercept"] = multilevel(
                pop, outcome, exposure, controls=["log_revenue"], categorical=["category", "state"],
                group="county_fips").to_dict()
        except Exception as exc:  # recorded, the two-stage model still runs
            fallbacks["state_fe_county_random_intercept"] = {"error": f"{type(exc).__name__}: {exc}"}
    fallbacks["two_stage_county_means"] = two_stage_group_means(
        pop, outcome, exposure, controls=["log_revenue"], absorb=["category"],
        groups=["county_fips"]).to_dict()

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor()},
        "year": args.year,
        "population": pop_log,
        "assumption_5": check,
        "fit": fit.to_dict(),
        "fallbacks": fallbacks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(check), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
