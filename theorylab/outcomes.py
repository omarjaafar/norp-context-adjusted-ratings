"""Outcome rule for heavy-tailed ratios (CP5, CP4 finding in docs/validator.md 2.8).

CP4 showed that for T2, T3 and T5 the outcome SD is 664x to 12,754x the robust SD,
and more than 99.9% of the variance sits in the top 0.1% of rows. The gate's floor
is 0.1 SD of the outcome, so for those ratios it is out of reach and "unsupported"
only says the test could not see anything.

The rule, applied by select_population (decided 2026-11-19, before any rerun):

  0. D-a: outcomes marked `bounded` in catalog/panel_columns.yaml (the [0, 1] expense
     shares and revenue concentration) are exempt. Their SD is finite and meaningful;
     a share with many zeros has a small IQR, so the ratio test would clip it for no reason.
  1. Diagnose on the theory's own population, pooled over years: SD / robust SD, with
     robust SD = IQR / 1.349.
  2. If the ratio exceeds `heavy_tail_ratio` (3.0), winsorize the outcome at the `winsorize`
     quantiles (0.01, 0.99) of the pooled population (D-b: one scale across years, which
     the per-year and trend tests need).
  3. D-c: clip once. If the ratio is still above the threshold afterwards it is logged
     (`still_above_threshold_after`), not clipped again.
  4. Everything downstream (H1, H2, the floor, replication, per-year tests, drill-down,
     sweep cells) uses the result. Raw numbers stay in the population log.

The rule is part of spec.GATE_POLICY, so the Planner cannot loosen or switch it off; it is
logged as an override if it tries.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from theorylab.executor import to_float

COLUMNS_PATH = Path(__file__).resolve().parent.parent / "catalog" / "panel_columns.yaml"

DEFAULT_RULE = {"heavy_tail_ratio": 3.0, "winsorize": [0.01, 0.99]}


@lru_cache(maxsize=None)
def bounded_outcomes(path: Path = COLUMNS_PATH) -> dict:
    """{column: [low, high]} for every column marked `bounded` in panel_columns.yaml (D-a)."""
    with open(path, encoding="utf-8") as f:
        columns = yaml.safe_load(f)["columns"]
    return {c: list(d["bounded"]) for c, d in columns.items() if isinstance(d, dict) and d.get("bounded")}


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
    bounds = bounded_outcomes().get(outcome)
    if bounds is not None:
        return pop, {"applied": False, "reason": f"bounded outcome {bounds}: exempt (D-a)", "bounded": bounds}
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
               sd_over_robust_sd_after=tail_ratio(clipped), pooled_over_years=True)
    log["still_above_threshold_after"] = bool(log["sd_over_robust_sd_after"] > log["threshold"])
    return out, log
