"""Outcome rule for heavy-tailed ratios (CP5, CP4 finding in docs/validator.md 2.8).

CP4 showed that for T2, T3 and T5 the outcome SD is 664x to 12,754x the robust SD,
and more than 99.9% of the variance sits in the top 0.1% of rows. The gate's floor
is 0.1 SD of the outcome, so for those ratios it is out of reach and "unsupported"
only says the test could not see anything.

The rule, fixed here before any rerun and applied by select_population:

  1. Diagnose on the theory's own population: SD / robust SD, with robust SD = IQR / 1.349.
  2. If the ratio exceeds `heavy_tail_ratio` (3.0; the well-behaved CP4 outcomes were 0.9
     and 1.6), winsorize the outcome at the `winsorize` quantiles (0.01, 0.99).
  3. Everything downstream (H1, H2, the floor, replication, per-year tests, drill-down,
     sweep cells) uses the winsorized outcome. Raw numbers stay in the population log.

Well-behaved outcomes (the 0-10 rating, expense shares) pass through untouched, so
CP4 verdicts for T1 and T4 do not move. The rule is part of spec.GATE_POLICY, so the
Planner cannot loosen or switch it off; it is logged as an override if it tries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from theorylab.executor import to_float

DEFAULT_RULE = {"heavy_tail_ratio": 3.0, "winsorize": [0.01, 0.99]}


def robust_sd(y: np.ndarray) -> float:
    q25, q75 = np.percentile(y, [25, 75])
    return float((q75 - q25) / 1.349)


def tail_ratio(y: np.ndarray) -> float:
    """SD over robust SD. Near 1 for a normal outcome; inf when the IQR is zero but the SD is not."""
    y = np.asarray(y, dtype=float)
    sd, rsd = float(np.std(y, ddof=1)), robust_sd(y)
    if rsd == 0:
        return float("inf") if sd > 0 else 1.0
    return sd / rsd


def validate_rule(rule: dict) -> list:
    errors = []
    if not isinstance(rule, dict):
        return ["validator.outcome_rule must be an object"]
    if not float(rule.get("heavy_tail_ratio", 0)) > 1:
        errors.append("outcome_rule.heavy_tail_ratio must be above 1")
    w = rule.get("winsorize")
    if not (isinstance(w, (list, tuple)) and len(w) == 2 and 0 <= w[0] < w[1] <= 1):
        errors.append("outcome_rule.winsorize must be [low, high] quantiles with 0 <= low < high <= 1")
    return errors


def apply_outcome_rule(pop: pd.DataFrame, outcome: str, rule: dict | None) -> tuple[pd.DataFrame, dict]:
    """Winsorize the outcome when it is heavy-tailed. Returns (population, log entry)."""
    if not rule:
        return pop, {"applied": False, "reason": "no outcome rule in the spec"}
    errors = validate_rule(rule)
    if errors:
        raise ValueError("; ".join(errors))
    y = to_float(pop[outcome]).to_numpy()
    ratio = tail_ratio(y)
    log = {"applied": False, "sd_over_robust_sd": ratio, "threshold": float(rule["heavy_tail_ratio"]),
           "raw_sd": float(np.std(y, ddof=1)), "raw_robust_sd": robust_sd(y)}
    if not ratio > float(rule["heavy_tail_ratio"]):
        return pop, log
    lo, hi = np.quantile(y, rule["winsorize"])
    clipped = np.clip(y, lo, hi)
    out = pop.copy()
    out[outcome] = clipped
    log.update(applied=True, quantiles=list(rule["winsorize"]), lower=float(lo), upper=float(hi),
               rows_clipped=int((clipped != y).sum()), sd_after=float(np.std(clipped, ddof=1)),
               sd_over_robust_sd_after=tail_ratio(clipped))
    return out, log
