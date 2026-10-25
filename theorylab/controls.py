"""Controls through the Validator gate on the real panel (CP3; the first CP5 benchmark controls).

    population (theory_run.select_population, the same rows as the theory run)
      -> negative control: replace the exposure with a placebo, another county's
         whole exposure history (UnitPermuter with keep_paths), once per seed, and
         run the full gate. There is no real effect, so every run should be
         "unsupported"; any other verdict is a false positive. The placebo keeps the
         real exposure's persistence over years, so a test that ignores that
         persistence shows up here as too many significant H1 gaps.
      -> positive control: keep the real exposure and plant a known gap in the
         outcome (expected_sign x effect_sd x SD(outcome) in exposed rows), then run
         the full gate. Above the effect floor it should be "supported"; below the
         floor it should be "unsupported", which shows the floor does its job.

Both run run_spec unchanged, so they test the same gate the theory goes through.
The real ratings gap is about -0.02 SD, so a planted gap adds to it; the report
records the measured effect next to the planted one.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from theorylab.executor import to_float, to_jsonable
from theorylab.spec import spec_columns
from theorylab.theory_run import add_derived, run_spec, select_population
from theorylab.validator import UnitPermuter


def _summary(report: dict) -> dict:
    """The numbers a control is judged on, from one run_spec report."""
    rep = report["replication"]
    return {
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "H1_effect": report["H1"]["effect"],
        "H1_standardized": report["H1"]["standardized_effect"],
        "H1_p": report["H1"]["permutation"]["p_value"],
        "H1_q": report["H1"]["q_value"],
        "H2_effect": report["H2"]["fit"]["effect"],
        "H2_standardized": report["H2"]["fit"]["standardized_effect"],
        "H2_q": report["H2"]["q_value"],
        "replicated": rep["replicated"],
        "folds_with_expected_sign": rep.get("folds_with_expected_sign"),
    }


def population(panel: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """The theory's population, selected once so each control run starts from the same rows."""
    return select_population(add_derived(panel, spec_columns(spec)), spec)


def negative_control(pop: pd.DataFrame, spec: dict, seeds, *, n_permutations: int | None = None) -> dict:
    """Run the gate with the exposure replaced by a placebo county history, once per seed."""
    exposure, within = spec["exposure"], spec.get("exposure_varies_by")
    permuter = UnitPermuter(pop, exposure, unit=spec["exposure_unit"], within=within,
                            keep_paths=within is not None)
    runs = []
    for seed in seeds:
        shuffled = pop.copy()
        shuffled[exposure] = permuter.draw(np.random.default_rng(seed))
        report = run_spec(shuffled, spec, n_permutations=n_permutations, include_robustness=False)
        runs.append({"seed": int(seed), **_summary(report)})
    verdicts = [r["verdict"] for r in runs]
    false_positives = [r["seed"] for r in runs if r["verdict"] != "unsupported"]
    return {
        "what": "exposure replaced by another county's whole history (placebo); no real effect",
        "expected_verdict": "unsupported",
        "n_runs": len(runs),
        "verdict_counts": {v: verdicts.count(v) for v in ("supported", "weak", "unsupported", "reversed")},
        "false_positives": len(false_positives),
        "false_positive_seeds": false_positives,
        # How often each piece of the gate fires on its own, before the floor and replication.
        "h1_significant": sum(r["H1_q"] is not None and r["H1_q"] < float(spec["validator"]["alpha"])
                              for r in runs),
        "h1_above_floor": sum(abs(r["H1_standardized"]) >= float(spec["validator"]["effect_floor_sd"])
                              for r in runs),
        "replicated": sum(bool(r["replicated"]) for r in runs),
        "runs": runs,
    }


def positive_control(pop: pd.DataFrame, spec: dict, effect_sd: float, expected_verdict: str, *,
                     n_permutations: int | None = None) -> dict:
    """Run the gate with a gap of effect_sd outcome SDs planted in the exposed rows."""
    outcome, exposure, sign = spec["outcome"], spec["exposure"], int(spec["expected_sign"])
    y = to_float(pop[outcome])
    exposed = to_float(pop[exposure]).eq(1.0)
    shift = sign * effect_sd * float(y.std(ddof=1))
    planted = pop.copy()
    planted[outcome] = y + shift * exposed
    report = run_spec(planted, spec, n_permutations=n_permutations, include_robustness=False)
    out = {"planted_effect_sd": float(effect_sd), "planted_shift": shift,
           "expected_verdict": expected_verdict, **_summary(report)}
    out["passed"] = out["verdict"] == expected_verdict
    return out


def run_controls(panel: pd.DataFrame, spec: dict, *, negative_seeds=range(20),
                 positive=((0.25, "supported"), (0.05, "unsupported")),
                 n_permutations: int | None = None) -> dict:
    """Negative and positive controls for one spec; the report data/output/controls.json holds."""
    pop, pop_log = population(panel, spec)
    if pop.empty:
        raise ValueError("the population is empty")
    neg = negative_control(pop, spec, list(negative_seeds), n_permutations=n_permutations)
    pos = [positive_control(pop, spec, sd, verdict, n_permutations=n_permutations)
           for sd, verdict in positive]
    return to_jsonable({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "theory_id": spec["theory_id"],
        "n_permutations": int(n_permutations or spec["validator"]["n_permutations"]),
        "robustness_models": False,
        "population": pop_log,
        "negative_control": neg,
        "positive_controls": pos,
        "all_passed": neg["false_positives"] == 0 and all(p["passed"] for p in pos),
    })
