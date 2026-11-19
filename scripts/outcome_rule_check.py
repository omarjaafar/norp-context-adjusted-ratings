"""CP5 V1: what the heavy-tailed outcome rule does to each of the 10 panel outcomes.

On full-990 filers pooled over 2018-2022 (the scale the rule clips on, D-b), for every
outcome: rows, share of zeros, SD / robust SD and the variance share of the top 0.1% of
rows, before and after the rule. The "after" numbers come from theorylab.outcomes itself,
so this file shows what the gate sees. Bounded outcomes (D-a) are exempt; for them the file
also shows what the ratio test alone would have done (the TA version clipped the fundraising
share at about 0.31). Descriptive only: no verdict depends on it.

Writes data/output/outcome_rule_check.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.outcomes import apply_outcome_rule, bounded_outcomes, robust_sd, tail_ratio  # noqa: E402
from theorylab.spec import GATE_POLICY  # noqa: E402
from theorylab.theory_run import load_panel  # noqa: E402

OUTCOMES = ["program_expense_ratio", "admin_expense_ratio", "fundraising_expense_ratio",
            "fundraising_efficiency", "operating_margin", "equity_ratio", "liabilities_to_assets",
            "revenue_concentration", "standard_rating", "revenue_growth"]
OUT_PATH = REPO_ROOT / "data" / "output" / "outcome_rule_check.json"


def describe(y: np.ndarray) -> dict:
    dev2 = (y - y.mean()) ** 2
    top = max(1, len(y) // 1000)
    return {"sd": float(np.std(y, ddof=1)), "robust_sd": robust_sd(y), "sd_over_robust_sd": tail_ratio(y),
            "variance_share_top_0.1pct_rows": float(np.sort(dev2)[-top:].sum() / dev2.sum()),
            "min": float(y.min()), "p01": float(np.quantile(y, 0.01)), "median": float(np.median(y)),
            "p99": float(np.quantile(y, 0.99)), "max": float(y.max())}


def main() -> int:
    t0 = time.time()
    rule = GATE_POLICY["outcome_rule"]
    panel = load_panel(columns=["return_type", "tax_year"] + OUTCOMES)
    panel = panel[panel["return_type"] == "990"]
    out = {"population": "full-990 filers, 2018-2022 pooled, outcome not missing",
           "rule": rule, "bounded": bounded_outcomes(), "outcomes": {}}
    for name in OUTCOMES:
        pop = panel[[name]].dropna()
        y = pop[name].astype(float).to_numpy()
        after_pop, log = apply_outcome_rule(pop, name, rule)
        entry = {"rows": int(len(y)), "share_zero": float((y == 0).mean()),
                 "before": describe(y), "rule_log": log,
                 "after": describe(after_pop[name].astype(float).to_numpy())}
        if name in bounded_outcomes():
            # what the ratio test alone would do, i.e. the TA version without D-a
            _, unexempt = apply_outcome_rule(pop.rename(columns={name: "_y"}), "_y", rule)
            entry["without_exemption"] = {k: unexempt.get(k) for k in
                                          ("applied", "sd_over_robust_sd", "lower", "upper", "rows_clipped")}
        out["outcomes"][name] = entry
    out["seconds"] = round(time.time() - t0, 1)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'outcome':28s} {'rows':>8s} {'zeros':>6s} {'ratio before':>13s} {'ratio after':>12s}  rule")
    for name, e in out["outcomes"].items():
        log = e["rule_log"]
        status = ("exempt (bounded)" if "bounded" in log else
                  f"clipped [{log['lower']:.3g}, {log['upper']:.3g}]"
                  + (" still > 3" if log.get("still_above_threshold_after") else "") if log["applied"] else "untouched")
        print(f"{name:28s} {e['rows']:8d} {e['share_zero']:6.1%} {e['before']['sd_over_robust_sd']:13,.2f} "
              f"{e['after']['sd_over_robust_sd']:12,.2f}  {status}")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} in {out['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
