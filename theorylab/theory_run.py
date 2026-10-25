"""Run one theory spec end to end: population -> Executor -> Validator -> verdict (CP3).

    panel (data/derived/org_year_panel.csv.gz)
      -> select_population()            spec.population: return type, joins, complete rows
      -> H1: difference in means        permutation test, counties shuffled within tax year
                                        (whole county histories with validator.permutation.keep_paths)
      -> H2: fixed effects              controls + absorbed factors, clustered SEs,
                                        permutation test on the same coefficient
      -> Benjamini-Hochberg over H1, H2
      -> held-out-state replication     H2 on k folds of states; feeds the verdict
      -> held-out-year replication      H2 in one year; reported for information only
      -> decide_verdict()               supported / weak / unsupported / reversed
      -> robustness models              reported next to the verdict, never change it

The report records every number the verdict rests on, so the CP4 Reporter can
narrate it without computing anything.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from theorylab.catalog import REPO_ROOT
from theorylab.executor import (Residualizer, correlation, fixed_effects, fwl_coefficient,
                                lagged_outcome, multilevel, to_float, to_jsonable,
                                two_stage_group_means)
from theorylab.ratios import RATING_ANCHORS, RATING_MIN_COMPONENTS, standard_rating
from theorylab.spec import SpecError, spec_columns
from theorylab.validator import (GateInput, UnitPermuter, benjamini_hochberg, decide_verdict,
                                 difference_in_means, held_out_group_replication,
                                 held_out_year_replication, permutation_test)

PANEL_PATH = REPO_ROOT / "data" / "derived" / "org_year_panel.csv.gz"
JOIN_LOG = REPO_ROOT / "data" / "output" / "panel_join_log.json"
RUNS_DIR = REPO_ROOT / "data" / "output" / "theory_runs"
PANEL_TEXT_COLUMNS = ("ein", "county_fips", "state", "category", "return_type")


def _log_revenue(df: pd.DataFrame) -> pd.Series:
    rev = to_float(df["revenue_total"])
    return np.log(rev.where(rev > 0))


# Columns a spec may use that the panel does not store: name -> (inputs, function).
DERIVED = {
    "log_revenue": (("revenue_total",), _log_revenue),
}
# The four ratios standard_rating() averages; needed to re-score the rating with other anchors.
RATING_INPUTS = tuple(RATING_ANCHORS)


def panel_columns(join_log: Path = JOIN_LOG) -> list:
    """Columns of the committed panel, from its join log (no need to open the 397 MB file)."""
    with open(join_log, encoding="utf-8") as f:
        return list(json.load(f)["panel"]["columns"])


def available_columns(join_log: Path = JOIN_LOG) -> list:
    return panel_columns(join_log) + list(DERIVED)


def load_panel(path: Path = PANEL_PATH, columns=None) -> pd.DataFrame:
    """Read the panel, only the columns asked for (plus the inputs of derived ones)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with python scripts/build_panel.py")
    usecols = None
    if columns is not None:
        wanted = set(columns)
        for name, (inputs, _) in DERIVED.items():
            if name in wanted:
                wanted.update(inputs)
        wanted -= set(DERIVED)
        usecols = lambda c: c in wanted  # noqa: E731
    dtype = {c: str for c in PANEL_TEXT_COLUMNS}
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)


def add_derived(df: pd.DataFrame, names) -> pd.DataFrame:
    todo = [n for n in names if n in DERIVED and n not in df.columns]
    if not todo:
        return df
    out = df.copy()
    for name in todo:
        out[name] = DERIVED[name][1](out)
    return out


def select_population(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Rows the theory is tested on, with a log of how many each condition kept."""
    pop = spec["population"]
    mask = pd.Series(True, index=df.index)
    log = {"rows_in": int(len(df))}
    if pop.get("return_type"):
        mask &= df["return_type"].isin(pop["return_type"])
        log["after_return_type"] = int(mask.sum())
    if pop.get("years"):
        mask &= to_float(df["tax_year"]).isin([float(y) for y in pop["years"]])
        log["after_years"] = int(mask.sum())
    for col in pop.get("require_true", []):
        mask &= to_float(df[col]).eq(1.0)
        log[f"after_{col}"] = int(mask.sum())
    for col in dict.fromkeys([*pop.get("require_present", []), spec["outcome"], spec["exposure"]]):
        mask &= df[col].notna()
        if col in (spec["outcome"], spec["exposure"]) or col in DERIVED:
            mask &= to_float(df[col]).notna()
        log[f"after_{col}_present"] = int(mask.sum())
    keys = [spec["exposure_unit"], "tax_year", "ein"]
    out = (df.loc[mask].sort_values([k for k in keys if k in df.columns], kind="stable")
           .reset_index(drop=True))
    log["rows_out"] = int(len(out))
    if "ein" in out.columns:
        log["organizations"] = int(out["ein"].nunique())
    log["units"] = int(out[spec["exposure_unit"]].nunique())
    exposed = to_float(out[spec["exposure"]]).eq(1.0)
    log["exposed_rows"] = int(exposed.sum())
    log["exposed_units_by_year"] = {
        str(int(y)): int(g.loc[exposed.loc[g.index], spec["exposure_unit"]].nunique())
        for y, g in out.groupby(to_float(out["tax_year"]))}
    return out, log


def _run_robustness(pop: pd.DataFrame, spec: dict, item: dict) -> dict:
    outcome, exposure = spec["outcome"], spec["exposure"]
    data = pop
    if item.get("years"):
        data = pop[to_float(pop["tax_year"]).isin([float(y) for y in item["years"]])]
    method = item["method"]
    try:
        if method == "multilevel":
            r = multilevel(data, outcome, exposure, controls=item.get("controls", []),
                           categorical=item.get("categorical", []),
                           group=item.get("group", spec["exposure_unit"]))
        elif method == "two_stage_group_means":
            groups = item.get("groups") or [spec["exposure_unit"], spec.get("exposure_varies_by")]
            r = two_stage_group_means(data, outcome, exposure, controls=item.get("controls", []),
                                      absorb=item.get("absorb", []), groups=[g for g in groups if g])
        elif method == "lagged_outcome":
            r = lagged_outcome(data, outcome, exposure, controls=item.get("controls", []),
                               lag=int(item.get("lag", 1)), absorb=item.get("absorb", ["tax_year"]),
                               cluster=item.get("cluster"))
        elif method == "correlation":
            r = correlation(data, outcome, exposure, method=item.get("kind", "pearson"))
        else:
            raise SpecError(f"unknown robustness method {method!r}")
    except Exception as exc:  # a failed robustness model is reported, not fatal
        return {"id": item.get("id"), "method": method, "error": f"{type(exc).__name__}: {exc}"}
    return {"id": item.get("id"), "method": method, "years": item.get("years"),
            "result": r.to_dict(), "sign_matches_expected": int(np.sign(r.effect)) == spec["expected_sign"]}


def run_spec(panel: pd.DataFrame, spec: dict, *, anchors: dict | None = None,
             min_components: int = RATING_MIN_COMPONENTS, n_permutations: int | None = None,
             include_robustness: bool = True) -> dict:
    """Test one theory on the panel. Returns the full report (also the input of write_report)."""
    v = spec["validator"]
    alpha, floor = float(v["alpha"]), float(v["effect_floor_sd"])
    n_perm = int(n_permutations or v["n_permutations"])
    seed = int(v["seed"])
    outcome, exposure, sign = spec["outcome"], spec["exposure"], int(spec["expected_sign"])
    unit, within = spec["exposure_unit"], spec.get("exposure_varies_by")
    hyps = {h["id"]: h for h in spec["hypotheses"]}

    df = panel
    # The panel's standard_rating was built with RATING_ANCHORS; re-score only for other settings.
    rescored = ((anchors is not None and dict(anchors) != dict(RATING_ANCHORS))
                or min_components != RATING_MIN_COMPONENTS)
    rating = {"anchors": {k: list(a) for k, a in (anchors or RATING_ANCHORS).items()},
              "min_components": int(min_components),
              "rescored": rescored}
    if rating["rescored"]:
        if outcome != "standard_rating":
            raise SpecError("anchors only apply to specs whose outcome is standard_rating")
        df = df.copy()
        df["standard_rating"] = standard_rating(df, anchors or RATING_ANCHORS, min_components)
    df = add_derived(df, spec_columns(spec))
    pop, pop_log = select_population(df, spec)
    if pop.empty:
        raise ValueError("the population is empty")

    y = to_float(pop[outcome]).to_numpy()
    x = to_float(pop[exposure]).to_numpy()
    sd_y = float(np.std(y, ddof=1))
    keep_paths = bool((v.get("permutation") or {}).get("keep_paths", False))
    permuter = UnitPermuter(pop, exposure, unit=unit, within=within, keep_paths=keep_paths)

    # H1: the gap exists.
    h1_effect = difference_in_means(y, x)
    h1_perm = permutation_test(lambda xv: difference_in_means(y, xv), permuter, n_perm, seed)
    welch = stats.ttest_ind(y[x == 1], y[x == 0], equal_var=False)

    # H2: the gap survives the controls.
    h2 = hyps["H2"]
    controls, absorb, cluster = h2.get("controls", []), h2.get("absorb", []), h2.get("cluster")

    def fit_h2(frame):
        return fixed_effects(frame, outcome, exposure, controls=controls, absorb=absorb, cluster=cluster)

    h2_fit = fit_h2(pop)
    resid = Residualizer.from_panel(pop, controls, absorb)
    y_resid = resid(y)
    h2_perm = permutation_test(lambda xv: fwl_coefficient(y_resid, resid(xv)), permuter, n_perm, seed + 1)

    q, _ = benjamini_hochberg([h1_perm["p_value"], h2_perm["p_value"]], alpha)
    holdout = v["holdout"]
    replication = held_out_group_replication(pop, fit_h2, group=holdout["by"],
                                             n_folds=int(holdout["folds"]), seed=seed + 2,
                                             alpha=alpha, expected_sign=sign)
    replication_year = (held_out_year_replication(pop, fit_h2, int(v["holdout_year"]), alpha=alpha,
                                                  expected_sign=sign)
                        if v.get("holdout_year") is not None else None)
    gate_h1 = GateInput(h1_effect, h1_effect / sd_y if sd_y > 0 else float("nan"), float(q[0]))
    gate_h2 = GateInput(h2_fit.effect, h2_fit.standardized_effect, float(q[1]))
    verdict = decide_verdict(gate_h1, gate_h2, expected_sign=sign, alpha=alpha, floor=floor,
                             replicated=replication["replicated"])

    robustness = ([_run_robustness(pop, spec, item) for item in spec.get("robustness", [])]
                  if include_robustness else [])

    return to_jsonable({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "theory_id": spec["theory_id"],
        "theory": spec["theory"],
        "verdict": verdict["verdict"],
        "reasons": verdict["reasons"],
        "validator": {"alpha": alpha, "effect_floor_sd": floor, "n_permutations": n_perm,
                      "seed": seed, "holdout": {"by": holdout["by"], "folds": int(holdout["folds"])},
                      "holdout_year_for_information": v.get("holdout_year"),
                      "permutation_unit": unit, "permutation_within": within,
                      "permutation_keep_paths": keep_paths,
                      "fdr_family": ["H1", "H2"]},
        "rating": rating,
        "population": pop_log,
        "outcome_sd": sd_y,
        "H1": {
            "claim": hyps["H1"].get("claim"),
            "method": "difference_in_means",
            "effect": h1_effect,
            "standardized_effect": gate_h1.standardized_effect,
            "mean_exposed": float(y[x == 1].mean()),
            "mean_unexposed": float(y[x == 0].mean()),
            "n_exposed": int((x == 1).sum()),
            "n_unexposed": int((x == 0).sum()),
            "welch_t_p_value_rows_independent": float(welch.pvalue),
            "permutation": h1_perm,
            "q_value": float(q[0]),
        },
        "H2": {
            "claim": h2.get("claim"),
            "method": "fixed_effects",
            "fit": h2_fit.to_dict(),
            "permutation": h2_perm,
            "q_value": float(q[1]),
        },
        "replication": replication,
        "replication_year": replication_year,
        "robustness": robustness,
    })


def write_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    return path
