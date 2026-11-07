"""Ratio x need sweep (CP4; the TA's CP2 feedback suggestion).

Each cell tests one rating ratio against one county need measure with the same
spec machinery as any other theory (continuous exposure, held-out states), and all
cells form one BH family (gate.apply_family), so a cell survives only after the
correction for the whole grid. Nulls are reported cell by cell, not only survivors.

Two schemes (CP4 audit W3, W10):
  "fixed" (the one we report): the locked gate policy (spec.GATE_POLICY: whole county
          histories shuffled, 999 permutations) and state fixed effects in H2.
  "ta":   the TA's ai-suggestions/cp4 cell, kept unchanged for the before/after: the
          county-year shuffle withdrawn in CP3 and no state fixed effects.

The main grid is the four ratios inside the standard rating x the ACS need index and
NORP housing burden (the TA's suggested 8 cells), plus every County Health Rankings
measure (chr_*) the panel has. Expected signs come from the rating anchors: the sign
that would lower an organization's rating, i.e. what a ratings penalty in high-need
counties would look like.
"""

from __future__ import annotations

import itertools

import pandas as pd

from theorylab.gate import apply_family
from theorylab.ratios import RATING_ANCHORS
from theorylab.spec import GATE_POLICY, lock_gate
from theorylab.theory_run import run_spec

# ratio -> sign of the change that lowers the rating (anchor scoring 10 above anchor scoring 0 -> -1)
RATING_SIGNS = {r: (-1 if hi > lo else 1) for r, (lo, hi) in RATING_ANCHORS.items()}
MAIN_NEEDS = ("need_index", "avg_housing_burden")

# The TA's grid as written (8 ratios x 5 needs, with its own expected signs).
TA_RATIOS = {
    "program_expense_ratio": -1, "admin_expense_ratio": 1, "fundraising_expense_ratio": 1,
    "fundraising_efficiency": -1, "operating_margin": -1, "equity_ratio": -1,
    "liabilities_to_assets": 1, "revenue_concentration": 1,
}
TA_NEEDS = ("need_index", "poverty_rate", "unemployment_rate", "avg_housing_burden",
            "food_desert_share")
SCHEMES = ("fixed", "ta")


def main_grid(columns) -> list:
    """[(ratio, need, sign)] for the reported sweep; CHR measures join when the panel has them."""
    needs = list(MAIN_NEEDS) + sorted(c for c in columns if c.startswith("chr_"))
    return [(r, n, s) for (r, s), n in itertools.product(RATING_SIGNS.items(), needs)]


def ta_grid() -> list:
    return [(r, n, s) for (r, s), n in itertools.product(TA_RATIOS.items(), TA_NEEDS)]


def cell_spec(ratio: str, need: str, sign: int, scheme: str = "fixed",
              n_permutations: int = 199, seed: int = 6235) -> dict:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    absorb = ["tax_year", "category"] + (["state"] if scheme == "fixed" else [])
    spec = {
        "spec_version": 1, "theory_id": f"sweep_{ratio}_x_{need}",
        "theory": f"{ratio} moves with {need} across counties.",
        "exposure_type": "continuous", "outcome": ratio, "exposure": need, "expected_sign": sign,
        "exposure_unit": "county_fips", "exposure_varies_by": "tax_year",
        "population": {"return_type": ["990"], "require_true": ["has_county", "has_need"],
                       "require_present": [ratio, need, "log_revenue", "category", "county_fips"]
                       + (["state"] if scheme == "fixed" else [])},
        "hypotheses": [
            {"id": "H1", "claim": f"{ratio} correlates with {need}.", "method": "correlation"},
            {"id": "H2", "claim": "The relation survives size, sector, tax year"
                                  + (" and state." if scheme == "fixed" else "."),
             "method": "fixed_effects", "controls": ["log_revenue"], "absorb": absorb,
             "cluster": "county_fips"}],
        "robustness": [],
        "validator": {"alpha": 0.05, "effect_floor_sd": 0.1, "n_permutations": n_permutations,
                      "seed": seed, "holdout": {"by": "state", "folds": 5}},
    }
    if scheme == "fixed":
        spec, _ = lock_gate(spec)
    return spec


def _cell_row(r: dict, alpha: float, floor: float) -> dict:
    sig = [h for h in ("H1", "H2") if r[h]["family_q_value"] < alpha]
    eff = {"H1": r["H1"]["standardized_effect"], "H2": r["H2"]["fit"]["standardized_effect"]}
    return {
        "cell": r["theory_id"], "verdict": r["verdict"], "reasons": r["reasons"],
        "n_rows": r["population"].get("rows_out"), "counties": r["population"].get("units"),
        "h1_r": r["H1"]["effect"], "h1_p": r["H1"]["permutation"]["p_value"],
        "h1_null_sd": r["H1"]["permutation"].get("null_sd"),
        "h2_std_effect": eff["H2"], "h2_p": r["H2"]["permutation"]["p_value"],
        "h2_clustered_p": r["H2"]["fit"].get("p_value"),
        "family_q_h1": r["H1"]["family_q_value"], "family_q_h2": r["H2"]["family_q_value"],
        "significant_after_family_bh": sig,
        "stopped_only_by_floor": bool(sig) and r["verdict"] == "unsupported"
                                 and all(abs(eff[h]) < floor for h in sig),
        "replicated_held_out_states": bool(r["replication"]["replicated"]),
    }


def run_sweep(panel: pd.DataFrame, grid=None, scheme: str = "fixed",
              n_permutations: int | None = None) -> dict:
    """Run every cell, gate them as one BH family, and report every cell (nulls included)."""
    grid = grid if grid is not None else main_grid(panel.columns)
    alpha, floor = GATE_POLICY["alpha"], GATE_POLICY["effect_floor_sd"]
    n_perm = n_permutations or (GATE_POLICY["n_permutations"] if scheme == "fixed" else 199)
    reports, signs, errors = [], {}, []
    for ratio, need, sign in grid:
        spec = cell_spec(ratio, need, sign, scheme, n_perm)
        try:
            reports.append(run_spec(panel, spec, include_robustness=False, n_permutations=n_perm))
            signs[spec["theory_id"]] = sign
        except (ValueError, KeyError) as exc:
            errors.append({"cell": spec["theory_id"], "error": str(exc)})
    fam = apply_family(reports, signs, alpha=alpha, floor=floor) if reports else []
    cells = [_cell_row(r, alpha, floor) for r in fam]
    return {
        "scheme": scheme,
        "permutation_keep_paths": scheme == "fixed",
        "state_fixed_effects": scheme == "fixed",
        "n_permutations": n_perm, "alpha": alpha, "effect_floor_sd": floor,
        "fdr_family": f"all H1 and H2 of the {len(cells)} cells ({2 * len(cells)} hypotheses)",
        "summary": {
            "cells": len(cells),
            "verdicts": {v: sum(c["verdict"] == v for c in cells)
                         for v in ("supported", "weak", "unsupported", "reversed")},
            "cells_with_family_q_below_alpha": sum(bool(c["significant_after_family_bh"]) for c in cells),
            "cells_stopped_only_by_floor": sum(c["stopped_only_by_floor"] for c in cells),
            "max_abs_std_effect": max((max(abs(c["h1_r"]), abs(c["h2_std_effect"])) for c in cells),
                                      default=None),
        },
        "cells": cells, "errors": errors,
        "survivors": [c["cell"] for c in cells if c["verdict"] == "supported"],
    }
