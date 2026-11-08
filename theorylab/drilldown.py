"""One-level drill-down for survivors (CP4).

A theory that is supported or weak is re-tested inside each level of one stratifier
(state, sector or size tercile). Every stratum gets the same H2 model and the same
county-level permutation test as the whole sample; the strata form one BH family, so
a stratum is only called out when it survives that family. Deterministic code only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from theorylab.executor import Residualizer, fixed_effects, to_float, to_jsonable
from theorylab.theory_run import add_derived, select_population
from theorylab.spec import spec_columns
from theorylab.validator import (UnitPermuter, benjamini_hochberg, gate_statistic,
                                 passes_effect_floor, permutation_test)

STRATIFIERS = {"state": "state", "sector": "category", "size": "size_tercile"}
SURVIVOR_VERDICTS = ("supported", "weak")
MIN_UNITS = 30  # a stratum needs this many exposure units (counties) to be tested


def size_terciles(log_revenue: pd.Series) -> pd.Series:
    return pd.qcut(to_float(log_revenue), 3, labels=["small", "medium", "large"]).astype("string")


def drill_down(panel: pd.DataFrame, spec: dict, report: dict, by: str, *,
               n_permutations: int | None = None, min_units: int = MIN_UNITS) -> dict:
    """Re-test the H2 model inside each level of `by` (state, sector or size)."""
    if by not in STRATIFIERS:
        raise ValueError(f"drill-down by {by!r}; choose one of {sorted(STRATIFIERS)}")
    if report["verdict"] not in SURVIVOR_VERDICTS:
        return {"by": by, "skipped": f"verdict is {report['verdict']}; only survivors are drilled"}
    v = spec["validator"]
    alpha, floor, sign = float(v["alpha"]), float(v["effect_floor_sd"]), int(spec["expected_sign"])
    n_perm = int(n_permutations or v["n_permutations"])
    h2 = next(h for h in spec["hypotheses"] if h["id"] == "H2")
    controls, absorb, cluster = h2.get("controls", []), h2.get("absorb", []), h2.get("cluster")
    exposure, outcome, unit = spec["exposure"], spec["outcome"], spec["exposure_unit"]
    within = spec.get("exposure_varies_by")
    keep_paths = bool((v.get("permutation") or {}).get("keep_paths", False))
    studentized = bool((v.get("permutation") or {}).get("studentized", False))

    df = add_derived(panel, spec_columns(spec))
    pop, _ = select_population(df, spec)
    if spec.get("exposure_type", "binary") == "continuous":
        xs = to_float(pop[exposure])
        pop = pop.assign(**{exposure: (xs - xs.mean()) / xs.std(ddof=1)})
    col = STRATIFIERS[by]
    if by == "size":
        pop = pop.assign(size_tercile=size_terciles(add_derived(pop, ["log_revenue"])["log_revenue"]))
    rows, skipped = [], []
    for level, part in pop.groupby(col, sort=True):
        part = part.reset_index(drop=True)
        if part[unit].nunique() < min_units:
            skipped.append({"level": str(level), "reason": f"fewer than {min_units} {unit} units"})
            continue
        try:
            fit = fixed_effects(part, outcome, exposure, controls=controls,
                                absorb=[a for a in absorb if a != col], cluster=cluster)
            resid = Residualizer.from_panel(part, controls, [a for a in absorb if a != col])
            y_resid = resid(to_float(part[outcome]).to_numpy())
            permuter = UnitPermuter(part, exposure, unit=unit, within=within, keep_paths=keep_paths)
            clusters = pd.factorize(part[cluster], sort=True)[0] if cluster else None
            perm = permutation_test(gate_statistic(y_resid, resid, clusters, studentized), permuter,
                                    n_perm, int(v["seed"]) + 100)
        except ValueError as exc:  # e.g. no exposure variation inside this stratum
            skipped.append({"level": str(level), "reason": str(exc)})
            continue
        rows.append({"level": str(level), "n_obs": fit.n_obs, "units": int(part[unit].nunique()),
                     "effect": fit.effect, "standardized_effect": fit.standardized_effect,
                     "p_value": perm["p_value"]})
    q, _ = benjamini_hochberg([r["p_value"] for r in rows], alpha)
    for r, qv in zip(rows, q):
        r["q_value"] = float(qv)
        r["same_sign"] = bool(np.sign(r["effect"]) == sign)
        r["survives"] = bool(qv < alpha and r["same_sign"]
                             and passes_effect_floor(r["standardized_effect"], floor))
    return to_jsonable({"by": by, "family_size": len(rows), "strata": rows, "skipped": skipped,
                        "survivors": [r["level"] for r in rows if r["survives"]]})
