"""Ratio x need sweep (CP4; the TA's CP3 suggestion, moved here).

Every ratio in the ratio library is tested against every continuous need measure with
the same spec machinery as any other theory (continuous exposure, county-level
permutation, held-out states). The cells form one BH family (gate.apply_family), so a
cell is reported only when it survives the correction for the whole grid.
"""

from __future__ import annotations

import copy
import itertools

import pandas as pd

from theorylab.gate import apply_family
from theorylab.theory_run import run_spec

RATIOS = {  # ratio -> expected sign of its relation to need (high need = fewer resources)
    "program_expense_ratio": -1, "admin_expense_ratio": 1, "fundraising_expense_ratio": 1,
    "fundraising_efficiency": -1, "operating_margin": -1, "equity_ratio": -1,
    "liabilities_to_assets": 1, "revenue_concentration": 1,
}
NEED_MEASURES = ("need_index", "poverty_rate", "unemployment_rate", "avg_housing_burden",
                 "food_desert_share")


def cell_spec(ratio: str, need: str, sign: int, n_permutations: int = 199, seed: int = 6235) -> dict:
    return {
        "spec_version": 1, "theory_id": f"sweep_{ratio}_x_{need}",
        "theory": f"{ratio} moves with {need} across counties.",
        "exposure_type": "continuous", "outcome": ratio, "exposure": need, "expected_sign": sign,
        "exposure_unit": "county_fips", "exposure_varies_by": "tax_year",
        "population": {"return_type": ["990"], "require_true": ["has_county", "has_need"],
                       "require_present": [ratio, need, "log_revenue", "category", "county_fips"]},
        "hypotheses": [
            {"id": "H1", "claim": f"{ratio} correlates with {need}.", "method": "correlation"},
            {"id": "H2", "claim": "The relation survives size, sector and tax year.",
             "method": "fixed_effects", "controls": ["log_revenue"], "absorb": ["tax_year", "category"],
             "cluster": "county_fips"}],
        "robustness": [],
        "validator": {"alpha": 0.05, "effect_floor_sd": 0.1, "n_permutations": n_permutations,
                      "seed": seed, "holdout": {"by": "state", "folds": 5}},
    }


def run_sweep(panel: pd.DataFrame, ratios=None, needs=NEED_MEASURES, n_permutations: int = 199) -> dict:
    ratios = ratios or RATIOS
    specs, reports = [], []
    for ratio, need in itertools.product(ratios, needs):
        spec = cell_spec(ratio, need, RATIOS.get(ratio, -1), n_permutations)
        try:
            reports.append(run_spec(panel, spec, include_robustness=False))
            specs.append(spec)
        except (ValueError, KeyError) as exc:
            reports.append(None)
            specs.append({"theory_id": spec["theory_id"], "error": str(exc)})
    ok = [r for r in reports if r is not None]
    fam = apply_family(ok, {s["theory_id"]: s["expected_sign"] for s in specs if "error" not in s})
    cells = [{"cell": r["theory_id"], "verdict": r["verdict"], "h1_r": r["H1"]["effect"],
              "h2_std_effect": r["H2"]["fit"]["standardized_effect"],
              "family_q_h1": r["H1"]["family_q_value"], "family_q_h2": r["H2"]["family_q_value"]}
             for r in fam]
    return {"cells": cells, "errors": [s for s in specs if "error" in s],
            "survivors": [c["cell"] for c in cells if c["verdict"] == "supported"]}
