"""Rating anchor sensitivity (CP3 open check from CP2).

The zero anchors of the standard rating come from published standards (BBB
Wise Giving Alliance 8 and 9, and liabilities equal to assets). The full-marks
anchors are our judgment (docs/ratings_theory.md, section 1). This module
reruns a theory spec with each full-marks anchor moved, one at a time, and
with the rating restricted to organizations that have all four ratios, and
reports whether the verdict survives.
"""

from __future__ import annotations

import pandas as pd

from theorylab.ratios import RATING_ANCHORS, RATING_MIN_COMPONENTS
from theorylab.theory_run import run_spec


def _move_full(ratio: str, full: float) -> dict:
    anchors = dict(RATING_ANCHORS)
    zero, _ = anchors[ratio]
    anchors[ratio] = (zero, full)
    return anchors


# name -> (anchors, min_components). Only "full marks" anchors move; zero anchors are published.
ANCHOR_VARIANTS = {
    "base": (dict(RATING_ANCHORS), RATING_MIN_COMPONENTS),
    "program_full_at_80pct": (_move_full("program_expense_ratio", 0.80), RATING_MIN_COMPONENTS),
    "program_full_at_90pct": (_move_full("program_expense_ratio", 0.90), RATING_MIN_COMPONENTS),
    "admin_full_at_10pct": (_move_full("admin_expense_ratio", 0.10), RATING_MIN_COMPONENTS),
    "admin_full_at_20pct": (_move_full("admin_expense_ratio", 0.20), RATING_MIN_COMPONENTS),
    "fundraising_full_at_5c": (_move_full("fundraising_efficiency", 0.05), RATING_MIN_COMPONENTS),
    "fundraising_full_at_20c": (_move_full("fundraising_efficiency", 0.20), RATING_MIN_COMPONENTS),
    "liabilities_full_at_25pct": (_move_full("liabilities_to_assets", 0.25), RATING_MIN_COMPONENTS),
    "liabilities_full_at_50pct": (_move_full("liabilities_to_assets", 0.50), RATING_MIN_COMPONENTS),
    "four_ratios_required": (dict(RATING_ANCHORS), 4),
}


def run_anchor_sensitivity(panel: pd.DataFrame, spec: dict, variants: dict = ANCHOR_VARIANTS,
                           n_permutations: int | None = None) -> dict:
    """Verdict and effect sizes for each anchor variant; stable = every verdict equals the base one."""
    if spec["outcome"] != "standard_rating":
        raise ValueError("anchor sensitivity applies to specs whose outcome is standard_rating")
    rows = {}
    for name, (anchors, min_components) in variants.items():
        report = run_spec(panel, spec, anchors=anchors, min_components=min_components,
                          n_permutations=n_permutations, include_robustness=False)
        rows[name] = {
            "anchors": {k: list(v) for k, v in anchors.items()},
            "min_components": min_components,
            "verdict": report["verdict"],
            "rows": report["population"]["rows_out"],
            "outcome_sd": report["outcome_sd"],
            "H1_effect": report["H1"]["effect"],
            "H1_standardized": report["H1"]["standardized_effect"],
            "H1_q": report["H1"]["q_value"],
            "H2_effect": report["H2"]["fit"]["effect"],
            "H2_standardized": report["H2"]["fit"]["standardized_effect"],
            "H2_q": report["H2"]["q_value"],
            "replicated": report["replication"]["replicated"],
        }
    base = rows.get("base", next(iter(rows.values())))["verdict"]
    changed = [n for n, r in rows.items() if r["verdict"] != base]
    return {"theory_id": spec["theory_id"], "base_verdict": base, "verdict_stable": not changed,
            "variants_with_other_verdict": changed, "variants": rows}
