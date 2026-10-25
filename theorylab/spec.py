"""Theory specs: the JSON a theory is turned into before anything is estimated.

In CP3 specs are written by hand (specs/ratings_fairness.json). In CP4 the
Planner agent writes them, and validate_spec() is the schema check its output
must pass: every column it names must exist in the panel (or be one of the
derived columns in theory_run.DERIVED), every method must be one the Executor
has, and the Validator settings must be complete. A spec that fails is
rejected, never repaired silently.
"""

from __future__ import annotations

import json
from pathlib import Path

HYPOTHESIS_METHODS = {"H1": ("difference_in_means",), "H2": ("fixed_effects",)}
ROBUSTNESS_METHODS = ("multilevel", "two_stage_group_means", "lagged_outcome", "correlation")
REQUIRED_KEYS = ("spec_version", "theory_id", "theory", "outcome", "exposure", "expected_sign",
                 "exposure_unit", "population", "hypotheses", "validator")
VALIDATOR_KEYS = ("alpha", "effect_floor_sd", "n_permutations", "seed", "holdout")


class SpecError(ValueError):
    """A theory spec is malformed or names something the panel does not have."""


def spec_columns(spec: dict) -> set:
    """Every panel column a spec refers to."""
    cols = {spec.get("outcome"), spec.get("exposure"), spec.get("exposure_unit"),
            spec.get("exposure_varies_by")}
    pop = spec.get("population") or {}
    cols.update(pop.get("require_true", []))
    cols.update(pop.get("require_present", []))
    if pop.get("return_type"):
        cols.add("return_type")
    for item in list(spec.get("hypotheses", [])) + list(spec.get("robustness", [])):
        cols.update(item.get("controls", []))
        cols.update(item.get("absorb", []))
        cols.update(item.get("categorical", []))
        cols.update(c for c in (item.get("cluster"), item.get("group")) if c)
    holdout = (spec.get("validator") or {}).get("holdout")
    if isinstance(holdout, dict) and holdout.get("by"):
        cols.add(holdout["by"])
    cols.add("tax_year")
    return {c for c in cols if c}


def validate_spec(spec: dict, available_columns=None) -> list:
    """List of problems with a spec; empty when it can be run."""
    errors = [f"missing key {k!r}" for k in REQUIRED_KEYS if k not in spec]
    if errors:
        return errors
    if spec["expected_sign"] not in (-1, 1):
        errors.append("expected_sign must be -1 or 1")
    pop = spec["population"]
    if not isinstance(pop, dict):
        errors.append("population must be an object")
    hyps = {h.get("id"): h for h in spec["hypotheses"]}
    for hid, methods in HYPOTHESIS_METHODS.items():
        if hid not in hyps:
            errors.append(f"hypothesis {hid} is missing")
        elif hyps[hid].get("method") not in methods:
            errors.append(f"{hid} method must be one of {methods}, not {hyps[hid].get('method')!r}")
    for item in spec.get("robustness", []):
        if item.get("method") not in ROBUSTNESS_METHODS:
            errors.append(f"robustness {item.get('id')!r}: unknown method {item.get('method')!r}")
    v = spec["validator"]
    errors += [f"validator.{k} is missing" for k in VALIDATOR_KEYS if k not in v]
    if "alpha" in v and not 0 < v["alpha"] < 1:
        errors.append("validator.alpha must be between 0 and 1")
    if "effect_floor_sd" in v and v["effect_floor_sd"] < 0:
        errors.append("validator.effect_floor_sd must be >= 0")
    if "n_permutations" in v and int(v["n_permutations"]) < 19:
        errors.append("validator.n_permutations must be at least 19 (p can reach 0.05)")
    if "holdout" in v:
        h = v["holdout"]
        if not isinstance(h, dict) or not h.get("by"):
            errors.append('validator.holdout must be an object like {"by": "state", "folds": 5}')
        elif not isinstance(h.get("folds"), int) or h["folds"] < 2:
            errors.append("validator.holdout.folds must be an integer of at least 2")
    if available_columns is not None:
        unknown = sorted(spec_columns(spec) - set(available_columns))
        if unknown:
            errors.append(f"columns not in the panel or derived columns: {unknown}")
    return errors


def load_spec(path, available_columns=None) -> dict:
    with open(Path(path), encoding="utf-8") as f:
        spec = json.load(f)
    errors = validate_spec(spec, available_columns)
    if errors:
        raise SpecError(f"{path}: " + "; ".join(errors))
    return spec
